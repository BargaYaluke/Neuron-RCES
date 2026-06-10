"""
Jaccard sub-table for the SupCon attribution ablation: how much does adding the
SupCon term to the NRC *selection gradient* change which neurons get selected?

Compares two neuron_masks.pth files (cell a = NRC/CE, cell b = NRC/CE+SupCon),
treating each Conv/Linear .weight output channel with a whole-channel mask == 0 as
a "selected" neuron. Reports per-layer Jaccard overlap of the selected index sets,
plus aggregate micro/macro Jaccard.

  py compare_masks.py \
     --a results/ResNet18_CIFAR10/checkpoint/<a_dir>/neuron_masks.pth \
     --b results/ResNet18_CIFAR10/checkpoint/<b_dir>/neuron_masks.pth

Read it as: low Jaccard => SupCon genuinely reshapes the selected neuron set
(the paper's "effect on the selected neuron set"); high Jaccard => SupCon barely
moves the selection and any robustness delta comes from elsewhere.
"""
import argparse

import torch


def _strip_prefix(name):
    changed = True
    while changed:
        changed = False
        for pre in ("module.", "_orig_mod."):
            if name.startswith(pre):
                name = name[len(pre):]
                changed = True
    return name


def selected_sets(path):
    masks = torch.load(path, map_location="cpu")["masks"]
    sets = {}
    for name, m in masks.items():
        if not name.endswith(".weight") or m.dim() < 2:
            continue
        sel = (m.flatten(1).sum(dim=1) == 0).nonzero(as_tuple=True)[0].tolist()
        sets[_strip_prefix(name)] = set(sel)
    return sets


def main():
    ap = argparse.ArgumentParser(description="Jaccard overlap of NRC-selected neurons (a vs b)")
    ap.add_argument("--a", required=True, help="cell a neuron_masks.pth (NRC, CE selection)")
    ap.add_argument("--b", required=True, help="cell b neuron_masks.pth (NRC, CE+SupCon selection)")
    args = ap.parse_args()

    A, B = selected_sets(args.a), selected_sets(args.b)
    layers = sorted(set(A) & set(B))

    print(f"{'layer':45s} {'|a|':>5s} {'|b|':>5s} {'∩':>5s} {'∪':>5s} {'Jaccard':>8s}")
    print("-" * 80)

    tot_i = tot_u = 0
    macro = []
    for name in layers:
        sa, sb = A[name], B[name]
        inter, union = len(sa & sb), len(sa | sb)
        if union == 0:
            continue  # layer selected nothing in either -> undefined, skip
        tot_i += inter
        tot_u += union
        jac = inter / union
        macro.append(jac)
        print(f"{name:45s} {len(sa):5d} {len(sb):5d} {inter:5d} {union:5d} {jac:8.3f}")

    print("-" * 80)
    micro = (tot_i / tot_u) if tot_u else float("nan")
    macro_avg = (sum(macro) / len(macro)) if macro else float("nan")
    print(f"overall micro-Jaccard (pooled over all neurons): {micro:.4f}")
    print(f"overall macro-Jaccard (mean over {len(macro)} layers): {macro_avg:.4f}")


if __name__ == "__main__":
    main()
