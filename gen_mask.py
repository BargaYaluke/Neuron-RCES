"""
Generate a neuron mask file compatible with main.py's finetune phase, WITHOUT
running NRC selection. Used for the ablation control rows:

  * cell (c) random baseline  : --mode random   (same per-layer neuron count as NRC,
                                                  random indices instead of MRC-ranked)
  * naive full finetune       : --mode full     (everything trainable; optional floor)

Mask convention is identical to neuron_mrc_and_prune() in main.py:
    mask == 0  -> trainable      mask == 1  -> frozen
Every named parameter gets an entry (biases / BN params are frozen = all ones),
so the gradient gate in main.py reproduces NRC's structure exactly except for
*which* output channels are unfrozen.

Recommended (budget-matched) usage for cell (c): match the per-layer trainable
count of cell (a)'s mask exactly via --like, so (a) and (c) differ ONLY in which
neurons are chosen, not how many:

  py gen_mask.py --mode random --resume $CKPT --seed 0 \
     --like results/ResNet18_CIFAR10/checkpoint/<a_dir>/neuron_masks.pth \
     --out  results/ResNet18_CIFAR10/checkpoint/<c_dir>/neuron_masks.pth
"""
import argparse
import os

import torch

from model import create_model


def _strip_prefix(name):
    """Mirror main.py._strip_module_prefix: drop DataParallel / torch.compile prefixes."""
    changed = True
    while changed:
        changed = False
        for pre in ("module.", "_orig_mod."):
            if name.startswith(pre):
                name = name[len(pre):]
                changed = True
    return name


def _trainable_count(ref_mask):
    """How many output channels/units are trainable (whole-channel mask == 0)."""
    if ref_mask.dim() < 2:
        return 0
    return int((ref_mask.flatten(1).sum(dim=1) == 0).sum().item())


def build_masks(model, mode="random", neurons_per_layer=1, ref_masks=None, seed=0):
    g = torch.Generator().manual_seed(int(seed))
    masks, statistic = {}, {}

    for name, p in model.named_parameters():
        # default: frozen everywhere (matches NRC for bias / BN / non-weight tensors)
        m = torch.ones_like(p, dtype=torch.float32)

        if mode == "full":
            # naive full finetune: every parameter trainable
            m = torch.zeros_like(p, dtype=torch.float32)
        elif name.endswith(".weight") and p.dim() in (2, 4):
            # Conv2d [C_out, C_in, kH, kW] or Linear [out, in]: one "neuron" per row 0
            B = p.shape[0]
            key = _strip_prefix(name)
            if ref_masks is not None and key in ref_masks:
                k = _trainable_count(ref_masks[key])      # exact per-layer budget match
            else:
                k = min(int(neurons_per_layer), B)
            if k > 0:
                idx = torch.randperm(B, generator=g)[:k]
                if p.dim() == 4:
                    m[idx, :, :, :] = 0.0
                else:
                    m[idx, :] = 0.0

        masks[name] = m
        statistic[name] = [int((m == 0).sum().item()), int(m.numel())]

    return masks, statistic


def main():
    ap = argparse.ArgumentParser(description="Generate non-NRC neuron masks for ablation controls")
    ap.add_argument("--model", default="ResNet18")
    ap.add_argument("--dataset", default="CIFAR10")          # kept for symmetry / clarity
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--input_size", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--resume", required=True,
                    help="the SAME robust checkpoint used by every ablation cell")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["random", "full"], default="random")
    ap.add_argument("--neurons_per_layer", type=int, default=1,
                    help="per-layer trainable count when --like is not given")
    ap.add_argument("--like", default=None,
                    help="reference neuron_masks.pth (e.g. cell-a) to match the per-layer "
                         "trainable count exactly (budget-matched random control)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output neuron_masks.pth path")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = create_model(args.model, args.input_size, args.num_classes,
                         device, args.patch, args.resume)

    ref_masks = None
    if args.like:
        ref_masks = torch.load(args.like, map_location="cpu")["masks"]
        ref_masks = {_strip_prefix(k): v for k, v in ref_masks.items()}

    masks, statistic = build_masks(model, args.mode, args.neurons_per_layer, ref_masks, args.seed)
    masks = {k: v.cpu() for k, v in masks.items()}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({"masks": masks, "statistic": statistic}, args.out)

    trainable_tensors = sum(1 for v in masks.values() if bool((v == 0).any()))
    trainable_neurons = sum(t for t, _ in statistic.values())
    print(f"[gen_mask] mode={args.mode}  match={'like:' + args.like if args.like else args.neurons_per_layer}")
    print(f"[gen_mask] wrote {len(masks)} mask tensors "
          f"({trainable_tensors} with trainable entries, "
          f"{trainable_neurons} trainable weight elements) -> {args.out}")


if __name__ == "__main__":
    main()
