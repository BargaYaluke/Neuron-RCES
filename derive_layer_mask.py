"""
Layer-scope ablation — derive the four arm masks from ONE cached NRC attribution, so the
*pooling scope* (and the *scale* it is ranked on) is the only variable.

Claim under test (paper §3.3): NRC is comparable WITHIN a layer but NOT ACROSS layers, so
selection must proceed layer by layer. We test it by changing only the pool we rank over and
the scale we rank on, at identical TOTAL budget B:

    layerwise    per-layer pool, raw NRC          k lowest per layer       == main method
    global_raw   whole-net pool, raw NRC          globally lowest B        (no cross-layer fix)
    global_z     whole-net pool, within-layer z   globally lowest B        (z = (x-mu_l)/sig_l)
    global_rank  whole-net pool, within-layer rank globally lowest B       (rank = (r+0.5)/N_l)

The two-part argument (see LAYER_ABLATION.md):
  * global_raw is EXPECTED to fail   -> proves "cross-layer NRC is incomparable" is real.
  * global_z / global_rank recover   -> proves the failure is the cross-layer SCALE gap,
                                         not the act of global selection itself.
  z-score keeps each layer's distribution shape (more informative control); rank erases scale
  entirely (most aggressive). If the two agree, the "it's just scale" conclusion is robust.

One attribution, four arms. The NRC scores are cached ONCE by main.py's selection phase
(`--cal_neuron_mrc --norm_mode l2`) into one directory <sel>:

    neuron_mrc_list.npy   (name, idx, score) for every ALIVE neuron (||W||>0 & ||g||>0 -> the
                          >0 filter main.py applies before saving) = the cached per-layer score
                          tensor AND the shared candidate set S_l.
    neuron_masks.pth      reused here ONLY for tensor SHAPES (which params exist, their dims,
                          N_l = out-channel count). Shapes are k-INDEPENDENT, so the directional
                          cache (run at any k) is reusable verbatim; --budget_k sets the budget.

Nothing is ever re-attributed. The per-layer z-score / rank transforms are applied ON TOP of
the cached l2-NRC values at selection time -- they never touch the numerator.

Mask convention is identical to main.py / derive_mask.py:
    mask == 0  -> trainable        mask == 1  -> frozen
Biases, BN affine and any 1-D tensor are all-ones (frozen). Boundary conventions (head
participates as 2-D; BN frozen; extreme occupancy under global selection NOT special-cased)
fall out automatically -- see LAYER_ABLATION.md.

Budget control differs from the directional ablation on purpose: there the budget was pinned
PER LAYER; here only the TOTAL B is pinned and the per-layer occupancy c_l is exactly what we
let vary (and measure -> occupancy.json).

Usage (the driver run_layer_ablation.sh passes these for you):
    py derive_layer_mask.py --scores <sel>/neuron_mrc_list.npy --template <sel>/neuron_masks.pth \
       --arm global_raw  --budget_k 2 --out results/.../<dir>/neuron_masks.pth
    py derive_layer_mask.py --scores ... --template ... --arm global_z   --budget_k 2 --out ...
    py derive_layer_mask.py --scores ... --template ... --arm global_rank --budget_k 2 --out ...
    py derive_layer_mask.py --scores ... --template ... --arm layerwise  --budget_k 2 --out ...
"""
import argparse
import json
import os

import numpy as np
import torch


ARMS = {
    # arm name      : (pool_scope, norm)
    "layerwise":   ("layerwise", "none"),
    "global_raw":  ("global",    "raw"),
    "global_z":    ("global",    "zscore"),
    "global_rank": ("global",    "rank"),
}


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


def _normalize_within_layer(scores, norm, eps=1e-12):
    """Map a layer's raw NRC vector to the scale the arm ranks on (lower = picked first).

    raw     : identity (the cross-layer-incomparable baseline)
    zscore  : (x - mu_l) / (sig_l + eps)  -- keeps distribution shape, removes location+scale
    rank    : (rank_ascending + 0.5) / N_l -- forces every layer onto the same [0,1] grid
              (most aggressive scale removal; the layer's lowest NRC -> smallest quantile)
    """
    scores = np.asarray(scores, dtype=float)
    if norm == "raw":
        return scores.copy()
    if norm == "zscore":
        mu = scores.mean()
        sigma = scores.std()                      # population std (ddof=0); +eps guards sigma==0
        return (scores - mu) / (sigma + eps)
    if norm == "rank":
        n = len(scores)
        if n == 0:
            return scores.copy()
        order = np.argsort(scores, kind="stable")  # ascending: smallest NRC first
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        return (ranks + 0.5) / n                   # midpoint convention; n==1 -> 0.5, no div0
    raise ValueError(f"unknown norm={norm!r}")


def _eligible_layers(scores_by_layer, template_masks):
    """Eligible (Conv/Linear .weight, dim in {2,4}) layers in DEPTH order, with their alive set.

    Depth order = the order params appear in the template (== model.named_parameters() order,
    which is forward/definition order). This is the depth axis for any occupancy figure drawn
    later from occupancy.json (plotting is intentionally not part of this pipeline).
    """
    layers = []
    depth = 0
    for name, tmpl in template_masks.items():
        if not (name.endswith(".weight") and tmpl.dim() in (2, 4)):
            continue
        key = _strip_prefix(name)
        alive = scores_by_layer.get(key, {})
        idxs = np.array(sorted(alive.keys()), dtype=int)
        scr = np.array([alive[int(i)] for i in idxs], dtype=float)
        layers.append({
            "name": name, "key": key, "depth": depth,
            "N_l": int(tmpl.shape[0]), "dim": int(tmpl.dim()),
            "idxs": idxs, "scores": scr,
        })
        depth += 1
    return layers


def build_layer_masks(scores_by_layer, template_masks, arm, budget_k):
    """Build {param_name: mask_tensor} for one arm + the occupancy record.

    Returns (masks, occupancy_dict). budget B = sum_l k_l with k_l = min(budget_k, |S_l|),
    identical across arms; per-layer occupancy c_l is the variable we measure.
    """
    pool_scope, norm = ARMS[arm]
    layers = _eligible_layers(scores_by_layer, template_masks)
    L = len(layers)

    # k_l and the total budget B come from --budget_k + the alive-set sizes (k-independent of
    # the cache). For RN18/CIFAR10 |S_l| >> k so k_l == budget_k for every layer.
    for lay in layers:
        lay["k_l"] = int(min(budget_k, len(lay["idxs"])))
    budget = int(sum(lay["k_l"] for lay in layers))

    # ----------------------------- selection -----------------------------
    chosen_by_layer = {}   # name -> set(selected original channel idx)
    if pool_scope == "layerwise":
        # main method reproduced: k_l lowest RAW NRC within each layer, every layer.
        for lay in layers:
            if len(lay["idxs"]) == 0 or lay["k_l"] == 0:
                continue
            order = np.argsort(lay["scores"], kind="stable")   # ascending
            sel = lay["idxs"][order[:lay["k_l"]]]
            chosen_by_layer[lay["name"]] = set(int(i) for i in sel)
    else:
        # global: pool every alive neuron's (possibly normalized) value, take the lowest B.
        pool = []  # (value, depth, idx, name) -- depth/idx break ties deterministically
        for lay in layers:
            vals = _normalize_within_layer(lay["scores"], norm)
            for i, idx in enumerate(lay["idxs"]):
                pool.append((float(vals[i]), lay["depth"], int(idx), lay["name"]))
        pool.sort(key=lambda t: (t[0], t[1], t[2]))            # stable, fully deterministic
        for value, depth, idx, name in pool[:budget]:
            chosen_by_layer.setdefault(name, set()).add(idx)

    # ----------------------------- build masks ---------------------------
    # Every template param gets a mask (so the finetune gate_list matches every tensor);
    # non-eligible tensors and unselected channels stay frozen (all-ones).
    masks, statistic = {}, {}
    for name, tmpl in template_masks.items():
        m = torch.ones_like(tmpl, dtype=torch.float32)
        if name in chosen_by_layer and chosen_by_layer[name]:
            sel_t = torch.as_tensor(sorted(chosen_by_layer[name]), dtype=torch.int64)
            if tmpl.dim() == 4:
                m[sel_t, :, :, :] = 0.0
            else:
                m[sel_t, :] = 0.0
        masks[name] = m
        statistic[name] = [int((m == 0).sum().item()), int(m.numel())]

    # ----------------------------- occupancy -----------------------------
    occ_layers = []
    for lay in layers:
        c_l = len(chosen_by_layer.get(lay["name"], ()))
        occ_layers.append({
            "depth_idx": lay["depth"], "name": lay["name"],
            "N_l": lay["N_l"], "n_alive": int(len(lay["idxs"])),
            "k_template": lay["k_l"], "c_l": int(c_l),
        })
    covered = sum(1 for o in occ_layers if o["c_l"] > 0)
    cmax = max((o["c_l"] for o in occ_layers), default=0)
    argmax = next((o["name"] for o in occ_layers if o["c_l"] == cmax and cmax > 0), None)
    occupancy = {
        "arm": arm, "pool_scope": pool_scope, "norm": norm,
        "budget": budget, "n_selected": int(sum(o["c_l"] for o in occ_layers)),
        "n_layers": L, "n_layers_covered": covered,
        "coverage": (covered / L) if L else float("nan"),
        "max_c_l": int(cmax), "argmax_layer": argmax,
        "max_share": (cmax / budget) if budget else float("nan"),
        "layers": occ_layers,
    }
    return masks, occupancy


def main():
    ap = argparse.ArgumentParser(
        description="Derive a layer-scope-ablation arm mask from one cached NRC attribution")
    ap.add_argument("--scores", required=True,
                    help="<sel>/neuron_mrc_list.npy from the single selection run (the cache)")
    ap.add_argument("--template", required=True,
                    help="<sel>/neuron_masks.pth from the same run (used ONLY for tensor shapes)")
    ap.add_argument("--arm", required=True, choices=list(ARMS.keys()))
    ap.add_argument("--budget_k", type=int, required=True,
                    help="per-layer k for the layerwise arm; sets total budget B = sum_l "
                         "min(k,|S_l|), enforced identically on the global arms")
    ap.add_argument("--out", required=True, help="output neuron_masks.pth for this arm")
    ap.add_argument("--occ_out", default=None,
                    help="occupancy json path (default: <out dir>/occupancy.json)")
    args = ap.parse_args()

    scores_by_layer = load_scores(args.scores)
    tmpl = torch.load(args.template, map_location="cpu")
    template_masks = tmpl["masks"]
    norm_mode_cache = tmpl.get("norm_mode", "?")   # the cache's NRC normalizer (should be l2)

    masks, occupancy = build_layer_masks(
        scores_by_layer, template_masks, args.arm, args.budget_k)
    masks = {k: v.cpu() for k, v in masks.items()}
    statistic = {k: [int((v == 0).sum().item()), int(v.numel())] for k, v in masks.items()}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "masks": masks,
        "statistic": statistic,
        "arm": args.arm,
        "pool_scope": occupancy["pool_scope"],
        "norm": occupancy["norm"],
        "budget_k": args.budget_k,
        "budget": occupancy["budget"],
        # carried through so downstream tooling can confirm every arm shares one attribution
        "norm_mode": norm_mode_cache,
        "derived_from": os.path.abspath(args.scores),
    }, args.out)

    # occupancy dumped NOW, at mask-generation time (not after training) -- this is the raw
    # data (per-layer c_l + the two headline scalars) for a figure drawn separately later.
    occ_path = args.occ_out or (os.path.join(out_dir, "occupancy.json") if out_dir
                                else "occupancy.json")
    occupancy_full = dict(occupancy)
    occupancy_full["derived_from"] = os.path.abspath(args.scores)
    occupancy_full["norm_mode_cache"] = norm_mode_cache
    with open(occ_path, "w") as f:
        json.dump(occupancy_full, f, indent=2)

    trainable_neurons = sum(t for t, _ in statistic.values())
    print(f"[derive_layer_mask] arm={args.arm}  pool={occupancy['pool_scope']}  "
          f"norm={occupancy['norm']}  budget_k={args.budget_k}  "
          f"norm_mode_cache={norm_mode_cache}")
    print(f"[derive_layer_mask] B={occupancy['budget']}  selected={occupancy['n_selected']}  "
          f"L={occupancy['n_layers']}  coverage={occupancy['coverage']:.3f} "
          f"({occupancy['n_layers_covered']}/{occupancy['n_layers']})  "
          f"max_share={occupancy['max_share']:.3f} "
          f"(max c_l={occupancy['max_c_l']} @ {occupancy['argmax_layer']})")
    print(f"[derive_layer_mask] {trainable_neurons} trainable weight elements "
          f"-> {args.out}")
    print(f"[derive_layer_mask] occupancy -> {occ_path}")
    if occupancy["n_selected"] != occupancy["budget"]:
        print(f"[derive_layer_mask] WARNING: selected {occupancy['n_selected']} != budget "
              f"{occupancy['budget']} (equal-budget control would be violated).")
    else:
        print(f"[derive_layer_mask] OK: total selection == budget "
              f"(equal-budget control holds; per-layer occupancy is the free variable).")


if __name__ == "__main__":
    main()
