"""
Normalizer-ablation metrics (ResNet18 / CIFAR10).

For each normalizer variant (none / l1 / l2 / linf) this reports, RELATIVE TO the
L2 reference (your method), the two layers of the ablation table:

  (1) selection stability — does the normalizer change WHICH neurons are picked?
        * Jaccard of the top-k selected sets   (micro = pooled over all neurons,
          macro = mean over layers).  High => same neurons chosen.
        * Kendall's tau of the per-neuron NRC ranking (per-layer mean + global).
          High => same ordering.  tau can drop while Jaccard stays high if only
          the *lowest k* survive unchanged — which is all your method cares about.

  (2) result stability — clean accuracy ("std accuracy") and robust accuracy
        (PGD-10, the main-table attack), read from each cell's
        experiment_record.json -> final_result.

Each variant reads two artifacts written by main.py's selection phase:
    <dir>/neuron_mrc_list.npy   (per-neuron NRC scores: list of (name, idx, score))
    <dir>/neuron_masks.pth      (the top-k selected sets, + stamped norm_mode)
and, if present, <dir>/experiment_record.json for the accuracies.

Usage (the driver passes these for you):
    py norm_metrics.py \
        --cell none=results/ResNet18_CIFAR10/checkpoint/<...>_norm_none_s0 \
        --cell l1=results/ResNet18_CIFAR10/checkpoint/<...>_norm_l1_s0 \
        --cell l2=results/ResNet18_CIFAR10/checkpoint/<...>_norm_l2_s0 \
        --cell linf=results/ResNet18_CIFAR10/checkpoint/<...>_norm_linf_s0 \
        --ref l2 [--out norm_ablation_table.json]
"""
import argparse
import json
import math
import os
from collections import Counter

import numpy as np
# torch is imported lazily inside load_selected() — the Jaccard/Kendall math and the
# .npy reader only need numpy, so the table can be inspected on a torch-less box.

# Kendall's tau: prefer scipy (O(n log n), tau-b ties), fall back to a pure-numpy
# tau-b so the script still runs on a box without scipy.
try:
    from scipy.stats import kendalltau as _scipy_kendalltau
except Exception:  # pragma: no cover - scipy is optional
    _scipy_kendalltau = None


def _strip_prefix(name):
    """Mirror main.py._strip_module_prefix (drop DataParallel / compile prefixes)."""
    changed = True
    while changed:
        changed = False
        for pre in ("module.", "_orig_mod."):
            if name.startswith(pre):
                name = name[len(pre):]
                changed = True
    return name


def _kendall_tau_b(x, y):
    """tau-b (handles ties). O(n^2); fine for per-layer (<=512) and global (~5k)."""
    n = len(x)
    if n < 2:
        return float("nan")
    nc = nd = 0
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n):
            s = (xi - x[j]) * (yi - y[j])
            if s > 0:
                nc += 1
            elif s < 0:
                nd += 1
    n0 = n * (n - 1) / 2.0
    n1 = sum(t * (t - 1) / 2.0 for t in Counter(x).values() if t > 1)
    n2 = sum(t * (t - 1) / 2.0 for t in Counter(y).values() if t > 1)
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    return (nc - nd) / denom if denom > 0 else float("nan")


def kendall_tau(x, y):
    if len(x) < 2:
        return float("nan")
    if _scipy_kendalltau is not None:
        tau, _ = _scipy_kendalltau(x, y)
        return float(tau)
    return _kendall_tau_b(list(x), list(y))


def load_scores(cell_dir):
    """neuron_mrc_list.npy -> {layer_name: {neuron_idx: nrc_score}}."""
    path = os.path.join(cell_dir, "neuron_mrc_list.npy")
    rows = np.load(path, allow_pickle=True)
    per_layer = {}
    for name, idx, score in rows:
        per_layer.setdefault(_strip_prefix(str(name)), {})[int(idx)] = float(score)
    return per_layer


def load_selected(cell_dir):
    """neuron_masks.pth -> ({layer_name: set(selected_idx)}, norm_mode)."""
    import torch  # lazy: only this reader needs torch
    ckpt = torch.load(os.path.join(cell_dir, "neuron_masks.pth"), map_location="cpu")
    masks = ckpt["masks"]
    norm_mode = ckpt.get("norm_mode", "?")
    sets = {}
    for name, m in masks.items():
        if not name.endswith(".weight") or m.dim() < 2:
            continue
        sel = (m.flatten(1).sum(dim=1) == 0).nonzero(as_tuple=True)[0].tolist()
        sets[_strip_prefix(name)] = set(int(i) for i in sel)
    return sets, str(norm_mode)


def load_accs(cell_dir):
    """experiment_record.json -> (clean_acc, robust_acc) or (None, None)."""
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
    """Per-layer Jaccard of selected sets; returns (micro, macro, per_layer)."""
    layers = sorted(set(sel) & set(ref_sel))
    tot_i = tot_u = 0
    per_layer = {}
    macro = []
    for name in layers:
        a, b = sel[name], ref_sel[name]
        inter, union = len(a & b), len(a | b)
        if union == 0:
            continue
        tot_i += inter
        tot_u += union
        j = inter / union
        per_layer[name] = j
        macro.append(j)
    micro = (tot_i / tot_u) if tot_u else float("nan")
    macro_avg = (sum(macro) / len(macro)) if macro else float("nan")
    return micro, macro_avg, per_layer


def tau_vs_ref(scores, ref_scores):
    """Per-layer-mean and global Kendall tau of NRC ranking vs reference.

    Aligned on the common (layer, neuron) set so a variant that drops/keeps a few
    zero-weight oddballs differently never biases the comparison."""
    layers = sorted(set(scores) & set(ref_scores))
    per_layer_taus = []
    gx, gy = [], []
    for name in layers:
        sv, rv = scores[name], ref_scores[name]
        common = sorted(set(sv) & set(rv))
        if len(common) < 2:
            continue
        x = [sv[i] for i in common]
        y = [rv[i] for i in common]
        t = kendall_tau(x, y)
        if not math.isnan(t):
            per_layer_taus.append(t)
        gx.extend(x)
        gy.extend(y)
    layer_mean = (sum(per_layer_taus) / len(per_layer_taus)) if per_layer_taus else float("nan")
    global_tau = kendall_tau(gx, gy) if len(gx) >= 2 else float("nan")
    return layer_mean, global_tau


def _parse_cell(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--cell expects label=DIR, got {s!r}")
    label, path = s.split("=", 1)
    return label.strip(), path.strip()


def _fmt(v, nd=3):
    return "   -  " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="Normalizer-ablation metrics vs L2 reference")
    ap.add_argument("--cell", action="append", type=_parse_cell, required=True,
                    help="label=DIR, repeatable (include the reference). "
                         "DIR holds neuron_mrc_list.npy + neuron_masks.pth [+ experiment_record.json]")
    ap.add_argument("--ref", default="l2", help="label of the reference cell (default: l2)")
    ap.add_argument("--out", default=None, help="optional path to dump the table as JSON")
    args = ap.parse_args()

    cells = dict(args.cell)
    if args.ref not in cells:
        ap.error(f"--ref {args.ref!r} is not among --cell labels {list(cells)}")

    ref_scores = load_scores(cells[args.ref])
    ref_sel, ref_norm = load_selected(cells[args.ref])

    # keep a stable, sensible row order if the usual labels are present
    order = ["none", "l1", "l2", "linf"]
    labels = [l for l in order if l in cells] + [l for l in cells if l not in order]

    rows = []
    for label in labels:
        d = cells[label]
        scores = load_scores(d)
        sel, norm_mode = load_selected(d)
        n_sel = sum(len(v) for v in sel.values())
        micro, macro, _ = jaccard_vs_ref(sel, ref_sel)
        tau_layer, tau_global = tau_vs_ref(scores, ref_scores)
        clean, robust = load_accs(d)
        rows.append({
            "label": label, "norm_mode": norm_mode, "is_ref": label == args.ref,
            "n_selected": n_sel,
            "jaccard_micro": micro, "jaccard_macro": macro,
            "kendall_tau_layer_mean": tau_layer, "kendall_tau_global": tau_global,
            "clean_acc": clean, "robust_acc": robust,
        })

    title = f"NORMALIZER ABLATION  (ResNet18/CIFAR10)  —  reference = {args.ref} (norm_mode={ref_norm})"
    print(title)
    print("=" * len(title))
    print(f"{'variant':8s} {'#sel':>5s} | {'Jaccard(sel)':>15s} | {'Kendall τ(rank)':>17s} | "
          f"{'clean%':>7s} {'robust%':>8s}")
    print(f"{'':8s} {'':>5s} | {'micro':>7s} {'macro':>7s} | {'layer':>8s} {'global':>8s} | "
          f"{'':>7s} {'':>8s}")
    print("-" * 78)
    for r in rows:
        tag = r["label"] + (" *" if r["is_ref"] else "")
        print(f"{tag:8s} {r['n_selected']:5d} | "
              f"{_fmt(r['jaccard_micro']):>7s} {_fmt(r['jaccard_macro']):>7s} | "
              f"{_fmt(r['kendall_tau_layer_mean']):>8s} {_fmt(r['kendall_tau_global']):>8s} | "
              f"{_fmt(r['clean_acc'], 2):>7s} {_fmt(r['robust_acc'], 2):>8s}")
    print("-" * 78)
    print("* = reference (Jaccard/τ trivially 1.000).  #sel = total trainable neurons.")
    print("Read: Jaccard≈1 & τ≈1  => normalizer barely moves the selection (robustness "
          "delta is noise);")
    print("      Jaccard high but τ lower => overall order shifts yet the chosen bottom-k "
          "is stable (good for you).")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"reference": args.ref, "ref_norm_mode": ref_norm, "rows": rows}, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
