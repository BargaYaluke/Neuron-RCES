"""
Layer-scope-ablation metrics (ResNet18 / CIFAR10).

One attribution, four pooling/scale rules. This script reports, for each arm:

  (0) EQUAL-BUDGET control (the headline guard) — TOTAL trainable neurons B must be
      byte-identical across layerwise / global_raw / global_z / global_rank. NOTE this is
      a TOTAL-only control, on purpose: unlike the directional ablation (which pinned the
      budget per layer), here the per-layer occupancy c_l is exactly the variable we measure.
      If the totals differ, the arms differ in parameter budget and the comparison is void.

  (1) OCCUPANCY — the per-layer trainable count c_l (depth-ordered), plus two scalars that
      go straight into the prose:
        coverage   = #{l : c_l>0} / L          (fraction of layers that got any budget)
        max_share  = max_l c_l / B             (budget mass in the single most occupied layer)
      Expectation:
        layerwise   : flat (c_l == k everywhere) -> coverage 1.000, max_share = k/B
        global_raw  : skewed -> low coverage, high max_share  (cross-layer NRC incomparable)
        global_z    : flattened but rippled                   (z keeps distribution shape)
        global_rank : flattest                                (rank erases scale)

  (2) RESULT — clean accuracy and robust accuracy (PGD-10, the main-table attack), read from
      each arm's experiment_record.json -> final_result. gamma=1 (un-interpolated) numbers.
      The story: global_raw << {global_z ~= global_rank ~= layerwise} on robust acc -> the
      per-layer SCALE is what makes global selection fail; selection itself is fine once
      normalized; fixed-k-per-layer still wins on simplicity + uniform coverage.

  (3) PROVENANCE — every arm should carry the same norm_mode / derived_from stamp, confirming
      all four came from ONE cached attribution.

Each arm dir holds:
    neuron_masks.pth          the arm's selected sets (+ arm / pool_scope / norm / derived_from)
    occupancy.json            per-layer c_l (depth-ordered) + coverage + max_share (dumped at
                              mask-generation time by derive_layer_mask.py)
    experiment_record.json    fine-tune result -> final_result.{final_test_acc, final_robust_acc}

Usage (driver passes these for you):
    py layer_metrics.py \
        --cell layerwise=results/ResNet18_CIFAR10/checkpoint/<...>_layer_layerwise_s0 \
        --cell global_raw=results/ResNet18_CIFAR10/checkpoint/<...>_layer_global_raw_s0 \
        --cell global_z=results/ResNet18_CIFAR10/checkpoint/<...>_layer_global_z_s0 \
        --cell global_rank=results/ResNet18_CIFAR10/checkpoint/<...>_layer_global_rank_s0 \
        --ref layerwise [--out layer_ablation_table.json]
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


def occupancy_from_masks(cell_dir):
    """neuron_masks.pth -> (occupancy_layers[depth-ordered], meta). Authoritative recompute.

    occupancy_layers: [{depth_idx, name, N_l, c_l}]   c_l = #trainable output channels.
    """
    import torch  # lazy: only this reader needs torch
    ckpt = torch.load(os.path.join(cell_dir, "neuron_masks.pth"), map_location="cpu")
    masks = ckpt["masks"]
    layers = []
    depth = 0
    for name, m in masks.items():
        if not name.endswith(".weight") or m.dim() < 2:
            continue
        c_l = int((m.flatten(1).sum(dim=1) == 0).sum().item())
        layers.append({"depth_idx": depth, "name": _strip_prefix(name),
                       "N_l": int(m.shape[0]), "c_l": c_l})
        depth += 1
    meta = {
        "arm": ckpt.get("arm", "?"), "pool_scope": ckpt.get("pool_scope", "?"),
        "norm": ckpt.get("norm", "?"), "budget": ckpt.get("budget", None),
        "norm_mode": ckpt.get("norm_mode", "?"), "derived_from": ckpt.get("derived_from", "?"),
    }
    return layers, meta


def load_occupancy_json(cell_dir):
    """occupancy.json (dumped at mask-gen time) -> dict, or None if absent."""
    path = os.path.join(cell_dir, "occupancy.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


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


def coverage_maxshare(occ_layers, budget):
    L = len(occ_layers)
    covered = sum(1 for o in occ_layers if o["c_l"] > 0)
    cmax = max((o["c_l"] for o in occ_layers), default=0)
    argmax = next((o["name"] for o in occ_layers if o["c_l"] == cmax and cmax > 0), None)
    coverage = (covered / L) if L else float("nan")
    max_share = (cmax / budget) if budget else float("nan")
    return covered, L, coverage, cmax, argmax, max_share


def _parse_cell(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--cell expects label=DIR, got {s!r}")
    label, path = s.split("=", 1)
    return label.strip(), path.strip()


def _fmt(v, nd=3):
    return "   -  " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="Layer-scope-ablation metrics vs the layerwise arm")
    ap.add_argument("--cell", action="append", type=_parse_cell, required=True,
                    help="label=DIR, repeatable (include the reference 'layerwise')")
    ap.add_argument("--ref", default="layerwise", help="reference arm label (default: layerwise)")
    ap.add_argument("--out", default=None, help="optional JSON dump path")
    args = ap.parse_args()

    cells = dict(args.cell)
    if args.ref not in cells:
        ap.error(f"--ref {args.ref!r} is not among --cell labels {list(cells)}")

    # stable, sensible row order
    order = ["layerwise", "global_raw", "global_z", "global_rank"]
    labels = ([l for l in order if l in cells]
              + sorted(l for l in cells if l not in order))

    rows, occ_dump = [], {}
    for label in labels:
        cell_dir = cells[label]
        occ_layers, meta = occupancy_from_masks(cell_dir)          # authoritative
        occ_json = load_occupancy_json(cell_dir)                   # dumped at mask-gen time
        budget = (meta.get("budget")
                  or (occ_json.get("budget") if occ_json else None)
                  or sum(o["c_l"] for o in occ_layers))
        covered, L, coverage, cmax, argmax, max_share = coverage_maxshare(occ_layers, budget)
        n_sel = sum(o["c_l"] for o in occ_layers)

        # cross-check the masks against the at-gen occupancy.json (must agree per layer)
        occ_mismatch = False
        if occ_json is not None:
            j = {_strip_prefix(o["name"]): o["c_l"] for o in occ_json.get("layers", [])}
            m = {o["name"]: o["c_l"] for o in occ_layers}   # names already stripped
            occ_mismatch = (j != m)

        clean, robust = load_accs(cell_dir)
        rows.append({
            "label": label, "is_ref": label == args.ref,
            "arm": meta["arm"], "pool_scope": meta["pool_scope"], "norm": meta["norm"],
            "norm_mode": meta["norm_mode"], "derived_from": meta["derived_from"],
            "budget": budget, "n_selected": n_sel,
            "n_layers": L, "n_layers_covered": covered, "coverage": coverage,
            "max_c_l": cmax, "argmax_layer": argmax, "max_share": max_share,
            "clean_acc": clean, "robust_acc": robust, "occ_mismatch": occ_mismatch,
        })
        occ_dump[label] = [{"depth_idx": o["depth_idx"], "name": o["name"],
                            "N_l": o["N_l"], "c_l": o["c_l"]} for o in occ_layers]

    title = f"LAYER-SCOPE ABLATION  (ResNet18/CIFAR10)  —  reference = {args.ref}"
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

    # equal-budget control (TOTAL only — per-layer is deliberately the variable)
    n_sel_set = set(r["n_selected"] for r in rows)
    budget_set = set(r["budget"] for r in rows if r["budget"] is not None)
    if len(n_sel_set) == 1 and len(budget_set) <= 1:
        print(f"[equal-budget OK] every arm trains {next(iter(n_sel_set))} neurons in total "
              f"(B={next(iter(budget_set), '?')}). Per-layer occupancy is the free variable.")
    else:
        print(f"[equal-budget VIOLATION] arms differ in TOTAL trainable budget — table void:")
        print(f"    total trainable neurons per arm: "
              f"{ {r['label']: r['n_selected'] for r in rows} }")
    for r in rows:
        if r["occ_mismatch"]:
            print(f"    [warn] {r['label']}: occupancy.json disagrees with neuron_masks.pth")
    print()

    hdr = (f"{'arm':12s} {'pool/scale':16s} {'#sel':>5s} | "
           f"{'cover':>6s} {'maxshr':>7s} {'maxlayer':>22s} | {'clean%':>7s} {'robust%':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = r["label"] + (" *" if r["is_ref"] else "")
        ps = f"{r['pool_scope']}/{r['norm']}"
        ml = (r["argmax_layer"] or "-")[-22:]
        print(f"{tag:12s} {ps:16s} {r['n_selected']:5d} | "
              f"{_fmt(r['coverage']):>6s} {_fmt(r['max_share']):>7s} {ml:>22s} | "
              f"{_fmt(r['clean_acc'], 2):>7s} {_fmt(r['robust_acc'], 2):>8s}")
    print("-" * len(hdr))
    print("#sel = total trainable neurons (must match across arms).  cover = L_cov/L.")
    print("maxshr = max_l c_l / B (budget mass in the single most occupied layer).")
    print("Read: global_raw << {global_z ~= global_rank ~= layerwise} on robust% =>")
    print("      cross-layer NRC scale is the culprit (raw fails, normalized recovers);")
    print("      fixed-k-per-layer wins on simplicity + uniform coverage (cover=1.000).")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "reference": args.ref, "rows": rows, "occupancy": occ_dump,
                "equal_budget_ok": (len(n_sel_set) == 1 and len(budget_set) <= 1),
            }, f, indent=2)
        print(f"\n[saved] {args.out}")
        print(f"[hint] per-arm occupancy (depth-ordered c_l + coverage + max_share) is in "
              f"each cell's occupancy.json and in this JSON under 'occupancy'.")


if __name__ == "__main__":
    main()
