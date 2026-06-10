"""
Directional ablation — derive the three arm masks (lowest / highest / random) from
ONE cached NRC attribution, so the *selection rule* is the only variable.

Design (see DIR_ABLATION.md):
  The attribution is computed ONCE by main.py's selection phase on the AT checkpoint
  (`--cal_neuron_mrc --norm_mode l2 --contrastive off ...`). That single run writes,
  into one directory <sel>:

    neuron_mrc_list.npy   per-neuron NRC scores  (name, idx, score) for every ALIVE
                          neuron  ->  the cached per-layer score tensor + the shared
                          candidate set S_l  (a neuron is in S_l iff ||W||2>0 & ||g||2>0,
                          which is exactly the >0 filter main.py applies before saving)
    neuron_masks.pth      the canonical "lowest" mask  ->  reused here ONLY as a per-layer
                          SHAPE + BUDGET template: k_l := its trainable channel count,
                          i.e. the main method's OWN per-layer selection size.

  Every arm derives its mask from that one cache. Nothing is ever re-attributed, so the
  per-run wobble of PGD / gradient accumulation that a per-arm re-run would inject can NOT
  leak into the comparison. (This is the whole point: re-attributing per arm pollutes the
  "selection rule" axis with attribution noise.)

Arms (per Conv/Linear `.weight` layer l, choosing exactly k_l channels from the SAME
alive candidate set S_l so the arms are budget-matched down to the neuron):
    lowest   k_l alive neurons with the SMALLEST NRC   == main method (reproduces template)
    highest  k_l alive neurons with the LARGEST  NRC   (deliberately train the units we
                                                        claim are most critical)
    random   k_l alive neurons sampled uniformly w/o replacement (--mask_seed, independent
                                                                  of the training seed)

Mask convention is identical to main.py / gen_mask.py:
    mask == 0  -> trainable        mask == 1  -> frozen
Biases, BN affine params and any 1-D tensor are all-ones (frozen). The three boundary
conventions you asked to pin down fall out automatically because BOTH the candidate set
S_l AND the budget k_l are read back from the main method's own output:
    * classification head: `linear.weight` is 2-D -> participates (its class-rows are
      candidate neurons), identically for all arms; `linear.bias` frozen.
    * BatchNorm affine: 1-D -> frozen branch -> frozen for all arms.
    * N_l < k: k_l comes from the template (= min(k, |S_l|) as main.py computed it),
      so all arms shrink identically.

Candidate-set note: `random` samples from S_l (the ALIVE set the other two arms rank
over), NOT from all N_l channels. This keeps the candidate set byte-identical across arms
-- the stated control -- and guarantees equal per-layer counts even in a degenerate layer
where |S_l| < k. For ResNet18/CIFAR10 essentially every channel is alive, so S_l == all
N_l and this is a no-op in practice; flip --random_pool all to sample from all channels.

Usage (the driver run_dir_ablation.sh passes these for you):
    py derive_mask.py --scores <sel>/neuron_mrc_list.npy --template <sel>/neuron_masks.pth \
       --arm highest --out results/ResNet18_CIFAR10/checkpoint/<dir>/neuron_masks.pth
    py derive_mask.py --scores ... --template ... --arm random --mask_seed 0 --out ...
"""
import argparse
import os

import numpy as np
import torch


def _strip_prefix(name):
    """Mirror main.py._strip_module_prefix (drop DataParallel / torch.compile prefixes)."""
    changed = True
    while changed:
        changed = False
        for pre in ("module.", "_orig_mod."):
            if name.startswith(pre):
                name = name[len(pre):]
                changed = True
    return name


def load_scores(npy_path):
    """neuron_mrc_list.npy -> {layer_name: {neuron_idx: nrc_score}} (the alive set S_l)."""
    rows = np.load(npy_path, allow_pickle=True)
    per_layer = {}
    for name, idx, score in rows:
        per_layer.setdefault(_strip_prefix(str(name)), {})[int(idx)] = float(score)
    return per_layer


def _trainable_count(mask):
    """How many output channels/units are trainable (whole-channel mask == 0) in template."""
    if mask.dim() < 2:
        return 0
    return int((mask.flatten(1).sum(dim=1) == 0).sum().item())


def select_arm(idxs, scores, k, arm, mask_seed):
    """Return the k chosen ORIGINAL channel indices for this arm.

    idxs   : np.ndarray of alive channel indices (sorted ascending)
    scores : np.ndarray of NRC scores aligned with idxs
    """
    if k <= 0 or len(idxs) == 0:
        return np.array([], dtype=int)
    k = min(k, len(idxs))
    if arm == "lowest":
        order = np.argsort(scores, kind="stable")            # smallest NRC first (== main method)
    elif arm == "highest":
        order = np.argsort(-scores, kind="stable")           # largest NRC first
    elif arm == "random":
        g = torch.Generator().manual_seed(int(mask_seed))    # torch RNG: matches gen_mask.py
        order = torch.randperm(len(idxs), generator=g).numpy()
    else:
        raise ValueError(f"unknown --arm={arm!r} (use lowest|highest|random)")
    return idxs[order[:k]]


def build_arm_masks(scores_by_layer, template_masks, arm, mask_seed, k_fallback, random_pool):
    """Build {param_name: mask_tensor} for one arm, using the template for shapes/budget."""
    masks, statistic = {}, {}
    summary = []  # (layer, B, n_alive, k_l, n_selected)

    for name, tmpl in template_masks.items():
        key = _strip_prefix(name)
        m = torch.ones_like(tmpl, dtype=torch.float32)       # frozen everywhere by default

        eligible = name.endswith(".weight") and tmpl.dim() in (2, 4)
        if eligible:
            B = tmpl.shape[0]
            # k_l := the main method's OWN per-layer selection size (template trainable count),
            # so every arm is budget-matched to the neuron. Fallback only if a layer somehow
            # has no template selection but we were asked to pick anyway.
            k_l = _trainable_count(tmpl)
            if k_l == 0 and k_fallback > 0:
                k_l = min(k_fallback, B)

            alive = scores_by_layer.get(key, {})
            if random_pool == "all":
                # sample from ALL channels; alive ones keep their score, dead ones get +inf
                idxs = np.arange(B)
                scores = np.array([alive.get(int(i), np.inf) for i in idxs], dtype=float)
            else:
                idxs = np.array(sorted(alive.keys()), dtype=int)
                scores = np.array([alive[int(i)] for i in idxs], dtype=float)

            sel = select_arm(idxs, scores, k_l, arm, mask_seed)
            if len(sel):
                sel_t = torch.as_tensor(np.asarray(sel, dtype=np.int64))
                if tmpl.dim() == 4:
                    m[sel_t, :, :, :] = 0.0
                else:
                    m[sel_t, :] = 0.0
            summary.append((key, int(B), int(len(idxs) if random_pool != "all"
                                            else int(np.isfinite(scores).sum())),
                            int(k_l), int(len(sel))))

        masks[name] = m
        statistic[name] = [int((m == 0).sum().item()), int(m.numel())]

    return masks, statistic, summary


def main():
    ap = argparse.ArgumentParser(
        description="Derive a directional-ablation arm mask from one cached NRC attribution")
    ap.add_argument("--scores", required=True,
                    help="<sel>/neuron_mrc_list.npy from the single selection run (the cache)")
    ap.add_argument("--template", required=True,
                    help="<sel>/neuron_masks.pth from the same run (shape + per-layer budget)")
    ap.add_argument("--arm", required=True, choices=["lowest", "highest", "random"])
    ap.add_argument("--mask_seed", type=int, default=0,
                    help="seed for the random arm (independent of the training --seed)")
    ap.add_argument("--random_pool", choices=["valid", "all"], default="valid",
                    help="random arm samples from the alive set S_l (valid, default; the "
                         "shared candidate set) or from all N_l channels (all)")
    ap.add_argument("--k_fallback", type=int, default=4,
                    help="per-layer k used ONLY if the template has no selection for a layer")
    ap.add_argument("--out", required=True, help="output neuron_masks.pth for this arm")
    args = ap.parse_args()

    scores_by_layer = load_scores(args.scores)
    tmpl = torch.load(args.template, map_location="cpu")
    template_masks = tmpl["masks"]
    norm_mode = tmpl.get("norm_mode", "?")

    masks, statistic, summary = build_arm_masks(
        scores_by_layer, template_masks, args.arm, args.mask_seed,
        args.k_fallback, args.random_pool)
    masks = {k: v.cpu() for k, v in masks.items()}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "masks": masks,
        "statistic": statistic,
        "arm": args.arm,
        "mask_seed": (args.mask_seed if args.arm == "random" else None),
        "random_pool": args.random_pool,
        # carried through so downstream tooling can confirm every arm shares one attribution
        "norm_mode": norm_mode,
        "derived_from": os.path.abspath(args.scores),
    }, args.out)

    trainable_neurons = sum(t for t, _ in statistic.values())
    trainable_tensors = sum(1 for v in masks.values() if bool((v == 0).any()))
    print(f"[derive_mask] arm={args.arm}"
          + (f"  mask_seed={args.mask_seed}" if args.arm == "random" else "")
          + f"  random_pool={args.random_pool}  norm_mode={norm_mode}")
    print(f"[derive_mask] {len(summary)} weight layers, "
          f"{trainable_tensors} tensors with trainable channels, "
          f"{trainable_neurons} trainable weight elements -> {args.out}")
    # per-layer budget check: k_l (selected) must equal the template's count for every layer
    bad = [(l, k, s) for (l, _B, _na, k, s) in summary if s != k]
    if bad:
        print(f"[derive_mask] WARNING: {len(bad)} layer(s) selected != template budget:")
        for l, k, s in bad[:10]:
            print(f"    {l}: k_template={k} selected={s}")
    else:
        print(f"[derive_mask] OK: every layer's selection size matches the template budget "
              f"(equal-count control holds).")


if __name__ == "__main__":
    main()
