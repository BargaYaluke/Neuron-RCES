"""
Directional-ablation metrics (ResNet18 / CIFAR10).

One attribution, three selection rules. This script reports, for each arm:

  (0) EQUAL-COUNT control (the headline guard) — total trainable neurons AND per-layer
      trainable counts must be byte-identical across lowest / highest / random. If they
      are not, the arms differ in parameter budget and the comparison is void; the script
      shouts about it.

  (1) selection overlap vs the lowest arm (your method) — Jaccard of the top-k selected
      sets, micro (pooled over neurons) and macro (mean over layers). Expectation:
        lowest vs lowest  = 1.000              (sanity)
        lowest vs highest ~ 0.000              (disjoint ends of the same ranking)
        lowest vs random  ~ k/N_l (chance)     (random shares only by accident)
      This confirms the arms really are different selections of the SAME size.

  (2) result — clean accuracy ("std acc") and robust accuracy (PGD-10, the main-table
      attack), read from each arm's experiment_record.json -> final_result. These are the
      gamma=1 (no-consolidation / un-interpolated) fine-tuned numbers the comparison rests
      on. The story you want: lowest >= random > highest on robust acc -> the NRC
      *direction* (train the least-critical units) is what carries the method, not merely
      "unfreeze 4 neurons per layer".

  (3) provenance — every arm should carry the same norm_mode / derived_from stamp,
      confirming all three came from ONE cached attribution.

Each arm dir holds:
    neuron_masks.pth          the arm's selected sets (+ arm / mask_seed / derived_from)
    experiment_record.json    fine-tune result -> final_result.{final_test_acc, final_robust_acc}

Usage (driver passes these for you):
    py dir_metrics.py \
        --cell lowest=results/ResNet18_CIFAR10/checkpoint/<...>_dir_lowest_s0 \
        --cell highest=results/ResNet18_CIFAR10/checkpoint/<...>_dir_highest_s0 \
        --cell random=results/ResNet18_CIFAR10/checkpoint/<...>_dir_rand_m0_s0 \
        --ref lowest [--out dir_ablation_table.json]
"""
import argparse
import json
import math
import os


def _strip_prefix(name):
    changed = True
    while changed:
        changed = False
        for pre in ("module.", "_orig_mod."):
            if name.startswith(pre):
                name = name[len(pre):]
                changed = True
    return name


def load_selected(cell_dir):
    """neuron_masks.pth -> ({layer: set(selected_idx)}, meta dict)."""
    import torch  # lazy: only this reader needs torch
    ckpt = torch.load(os.path.join(cell_dir, "neuron_masks.pth"), map_location="cpu")
    masks = ckpt["masks"]
    sets, counts = {}, {}
    for name, m in masks.items():
        if not name.endswith(".weight") or m.dim() < 2:
            continue
        sel = (m.flatten(1).sum(dim=1) == 0).nonzero(as_tuple=True)[0].tolist()
        key = _strip_prefix(name)
        sets[key] = set(int(i) for i in sel)
        counts[key] = len(sel)
    meta = {
        "arm": ckpt.get("arm", "?"),
        "mask_seed": ckpt.get("mask_seed", None),
        "norm_mode": ckpt.get("norm_mode", "?"),
        "random_pool": ckpt.get("random_pool", "?"),
        "derived_from": ckpt.get("derived_from", "?"),
    }
    return sets, counts, meta


def load_accs(cell_dir):
    """experiment_record.json -> (clean_acc, robust_acc) or (None, None). gamma=1 row."""
    path = os.path.join(cell_dir, "experiment_record.json")
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path) as f:
            rec = json.load(f)
        fr = rec.get("final_result", {}) or {}
        return fr.get("final_test_acc"), fr.get("final_robust_acc")
    except Exception:
        return None, None


def jaccard_vs_ref(sel, ref_sel):
    layers = sorted(set(sel) & set(ref_sel))
    tot_i = tot_u = 0
    macro = []
    for name in layers:
        a, b = sel[name], ref_sel[name]
        inter, union = len(a & b), len(a | b)
        if union == 0:
            continue
        tot_i += inter
        tot_u += union
        macro.append(inter / union)
    micro = (tot_i / tot_u) if tot_u else float("nan")
    macro_avg = (sum(macro) / len(macro)) if macro else float("nan")
    return micro, macro_avg


def _parse_cell(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--cell expects label=DIR, got {s!r}")
    label, path = s.split("=", 1)
    return label.strip(), path.strip()


def _fmt(v, nd=3):
    return "   -  " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="Directional-ablation metrics vs the lowest arm")
    ap.add_argument("--cell", action="append", type=_parse_cell, required=True,
                    help="label=DIR, repeatable (include the reference 'lowest')")
    ap.add_argument("--ref", default="lowest", help="reference arm label (default: lowest)")
    ap.add_argument("--out", default=None, help="optional JSON dump path")
    args = ap.parse_args()

    cells = dict(args.cell)
    if args.ref not in cells:
        ap.error(f"--ref {args.ref!r} is not among --cell labels {list(cells)}")

    ref_sel, ref_counts, ref_meta = load_selected(cells[args.ref])

    # stable, sensible row order
    order = ["lowest", "highest", "random"]
    labels = ([l for l in order if l in cells]
              + sorted(l for l in cells if l not in order))

    # ---- (0) equal-count control: per-layer counts identical across every arm? ----
    all_counts = {label: load_selected(cells[label])[1] for label in labels}
    all_layers = sorted(set().union(*[set(c) for c in all_counts.values()]))
    count_violations = []
    for name in all_layers:
        vals = {label: all_counts[label].get(name, 0) for label in labels}
        if len(set(vals.values())) > 1:
            count_violations.append((name, vals))

    rows = []
    for label in labels:
        sel, counts, meta = load_selected(cells[label])
        n_sel = sum(counts.values())
        micro, macro = jaccard_vs_ref(sel, ref_sel)
        clean, robust = load_accs(cells[label])
        rows.append({
            "label": label, "is_ref": label == args.ref, "arm": meta["arm"],
            "mask_seed": meta["mask_seed"], "norm_mode": meta["norm_mode"],
            "random_pool": meta["random_pool"], "derived_from": meta["derived_from"],
            "n_selected": n_sel,
            "jaccard_micro": micro, "jaccard_macro": macro,
            "clean_acc": clean, "robust_acc": robust,
        })

    title = f"DIRECTIONAL ABLATION  (ResNet18/CIFAR10)  —  reference = {args.ref}"
    print(title)
    print("=" * len(title))

    # provenance: all arms must share ONE attribution
    provs = set(r["derived_from"] for r in rows if r["derived_from"] != "?")
    norms = set(r["norm_mode"] for r in rows if r["norm_mode"] != "?")
    if len(provs) <= 1 and len(norms) <= 1:
        print(f"[provenance OK] one attribution for all arms "
              f"(norm_mode={next(iter(norms), '?')}).")
    else:
        print(f"[provenance WARNING] arms do NOT share one attribution: "
              f"derived_from={provs}  norm_mode={norms}")

    # equal-count control
    n_sel_set = set(r["n_selected"] for r in rows)
    if not count_violations and len(n_sel_set) == 1:
        print(f"[equal-count OK] every arm trains {next(iter(n_sel_set))} neurons, "
              f"identical per-layer counts (param budget controlled).")
    else:
        print(f"[equal-count VIOLATION] arms differ in trainable budget — comparison void:")
        print(f"    total trainable neurons per arm: "
              f"{ {r['label']: r['n_selected'] for r in rows} }")
        for name, vals in count_violations[:12]:
            print(f"    {name}: {vals}")
    print()

    print(f"{'arm':9s} {'#sel':>5s} | {'Jaccard vs ref':>15s} | {'clean%':>7s} {'robust%':>8s}")
    print(f"{'':9s} {'':>5s} | {'micro':>7s} {'macro':>7s} | {'':>7s} {'':>8s}")
    print("-" * 56)
    for r in rows:
        tag = r["label"] + (" *" if r["is_ref"] else "")
        print(f"{tag:9s} {r['n_selected']:5d} | "
              f"{_fmt(r['jaccard_micro']):>7s} {_fmt(r['jaccard_macro']):>7s} | "
              f"{_fmt(r['clean_acc'], 2):>7s} {_fmt(r['robust_acc'], 2):>8s}")
    print("-" * 56)
    print("* = reference (Jaccard trivially 1.000).  #sel = total trainable neurons (must match).")
    print("Read: lowest>=random>highest on robust% => the NRC *direction* drives the method,")
    print("      not just the per-layer unfreezing budget. lowest-vs-highest Jaccard~0 confirms")
    print("      the arms are disjoint ends of the SAME cached ranking.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"reference": args.ref, "rows": rows,
                       "equal_count_ok": (not count_violations and len(n_sel_set) == 1),
                       "count_violations": [{"layer": n, "counts": v}
                                            for n, v in count_violations]}, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
