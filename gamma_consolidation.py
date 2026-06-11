"""
gamma-consolidation ablation (ResNet18 / CIFAR10) — runner.

WHAT THIS ANSWERS (the only thing it answers)
---------------------------------------------
3.4 borrows RiFT's interpolation step. A reviewer will ask: "is your gain actually
RiFT's consolidation rather than the NRC selection?" This sweep makes that question
land nowhere. It is *defensive*, not a property test: we show the main-table number
is the gamma=1 (NO consolidation) point, and that no gamma<1 point beats it enough
to suggest the main method hid a private knob.

Eq. (8):   theta_final = theta_initial + gamma * (theta_tuned - theta_initial) ⊙ (1 - M̃)

with M̃ = the frozen-neuron mask actually used at training time (M̃=1 ⟺ frozen).
In this repo the stored mask already IS M̃: neuron_masks.pth["masks"] uses
1 = frozen, 0 = trainable (see main.py:369 "mask=0 -> trainable; mask=1 -> frozen").

This is a PURE-INFERENCE sweep: 11 gamma points, no training. Each point is one
build-and-evaluate. theta_initial = init_params.pth (the AT/RobustBench start saved
before fine-tuning); theta_tuned = best_params.pth["model"] (the main-table model).

THREE IMPLEMENTATION DETAILS (Eq. 8), all enforced/measured here
----------------------------------------------------------------
(a) MASK PROVENANCE. We read the landed neuron_masks.pth that sat next to the
    checkpoint during fine-tuning — never re-derive M̃ from cached NRC scores. That
    keeps any later change to the scoring code from drifting the mask out from under
    a finished run.

(b) FREE CONSISTENCY CHECK — the mask factor must be redundant. Frozen neurons are
    held at theta_initial during training (gradient gating + wd excluded), so
    (theta_tuned - theta_initial) should already be ZERO on M̃=1 positions, making
    the ⊙(1-M̃) factor numerically a no-op. We therefore interpolate the parameters
    *un-masked* (full lerp) — which guarantees gamma=0 == theta_initial and gamma=1
    == theta_tuned bit-exactly — and SEPARATELY assert
        ‖(theta_tuned - theta_initial) ⊙ M̃‖ ≈ 0.
    If that norm is not ~0 there is a training LEAK (a frozen weight moved: BN-affine
    not covered by the mask, wd touching a frozen tensor, etc.). Applying the mask
    would silently ERASE the leak and hide it; refusing to apply it and checking
    instead is what makes the leak visible. A real leak aborts the scan (--force to
    override) — it must be understood before the curve means anything.

(c) BN BUFFERS (running_mean / running_var) are NOT trainable parameters but DO
    update on every forward pass in train() mode — the gradient mask does not gate
    them — so theta_tuned's buffers ≠ theta_initial's buffers in general.
    Interpolating buffers has no theoretical basis, and using theta_initial's buffers
    would make gamma=1 miss the main-table number. Default policy: buffers follow
    theta_tuned at EVERY gamma (--bn_buffers tuned). Consequence: gamma=1 reproduces
    the main table bit-exactly, while gamma=0 is (init weights + tuned buffers), which
    equals the AT baseline only up to BN drift. We therefore also evaluate the full
    theta_initial ("AT anchor", init weights + init buffers) and report the
    gamma=0-vs-anchor gap so the drift is quantified rather than assumed away.
    --bn_buffers interp recovers bit-exact endpoints at the cost of blended interior
    buffers; --bn_buffers init is offered for completeness.

TWO FREE ENDPOINT CHECKS (printed every run)
--------------------------------------------
  * gamma=1 sweep point  vs  experiment_record.json final_result  -> the runner is
    faithful to the main-table protocol (should match to rounding).
  * gamma=0 sweep point  vs  AT anchor (full theta_initial)        -> under
    --bn_buffers tuned this gap is exactly the BN-drift effect (≈0 if BN didn't move).

METRICS (three, per gamma)
--------------------------
  clean  : clean test accuracy (normalized test set)
  adv    : PGD-10 eps=8/255 — the SAME attack as the main table (evaluate_cifar_robustness)
  ood    : CIFAR-10-C mean accuracy over the 18 corruptions (evaluate_cifar_corruption)

OOD is kept here (the first two ablations could skip it) because consolidation's whole
point is the stability-plasticity trade-off: clean/adv/ood on one axis show its shape.

SEEDS. Pass one --cell per seed checkpoint (the main table's 3 seeds). Each gamma is
evaluated in every cell; the table reports mean±std across cells.

Usage:
    py gamma_consolidation.py \
        --cell results/ResNet18_CIFAR10/checkpoint/<...>_s0 \
        --cell results/ResNet18_CIFAR10/checkpoint/<...>_s1 \
        --cell results/ResNet18_CIFAR10/checkpoint/<...>_s2 \
        --model ResNet18 --dataset CIFAR10 --num_classes 10 --input_size 32 \
        --out results/ResNet18_CIFAR10/checkpoint/gamma_consolidation.json

A cell directory must contain: init_params.pth, best_params.pth, neuron_masks.pth
(and, for the free gamma=1 check, experiment_record.json). These are exactly the
artifacts main.py writes for every fine-tune run.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from model import create_model
from utils import (
    evaluate,
    evaluate_cifar_robustness,
    evaluate_tiny_robustness,
    evaluate_cifar_corruption,
    evaluate_tiny_corruption,
)


# --------------------------------------------------------------------------- #
# key canonicalization — identical to main.py so mask keys line up with the
# (un-wrapped, single-GPU) model state_dict we interpolate over.
# --------------------------------------------------------------------------- #
def _strip_module_prefix(name):
    changed = True
    while changed:
        changed = False
        for pre in ("module.", "_orig_mod."):
            if name.startswith(pre):
                name = name[len(pre):]
                changed = True
    return name


def _canon(name):
    # main.py applies: strip module/_orig_mod, then a leading "<idx>." (norm-layer
    # Sequential prefix), then strip again. Mirror it exactly.
    return _strip_module_prefix(re.sub(r"^\d+\.", "", _strip_module_prefix(name)))


def _load_sd(path, device="cpu"):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    elif isinstance(sd, dict) and "net" in sd:
        sd = sd["net"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {_canon(k): v for k, v in sd.items()}


def _load_masks(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    masks = ckpt["masks"] if isinstance(ckpt, dict) and "masks" in ckpt else ckpt
    return {_canon(k): v for k, v in masks.items()}, ckpt


# --------------------------------------------------------------------------- #
# consistency diagnostics (Eq. 8, details b & c)
# --------------------------------------------------------------------------- #
def leak_check(init_sd, ft_sd, masks, tol):
    """Detail (b): ‖(theta_tuned - theta_initial) ⊙ M̃‖ must be ~0.

    Returns (max_frozen_leak, worst_layer, rows) where rows is a per-layer list
    of (name, frozen_leak, trainable_move) for the loudest offenders.
    """
    rows = []
    max_leak = 0.0
    worst = None
    for name, m in masks.items():
        if name not in ft_sd or name not in init_sd:
            continue
        diff = (ft_sd[name].float() - init_sd[name].float())
        if diff.shape != m.shape:
            # mask is per-parameter and same-shape by construction; skip if not.
            continue
        frozen = m.float()                      # 1 = frozen
        leak = float((diff * frozen).norm().item())
        move = float((diff * (1.0 - frozen)).norm().item())
        rows.append((name, leak, move))
        if leak > max_leak:
            max_leak, worst = leak, name
    rows.sort(key=lambda r: r[1], reverse=True)
    return max_leak, worst, rows


def bn_buffer_drift(init_sd, ft_sd, buffer_keys):
    """Detail (c): how far did BN running stats move during fine-tuning?

    This is the quantity that decides whether gamma=0 (under --bn_buffers tuned)
    can equal the AT baseline. Reported, never asserted — drift is EXPECTED.
    """
    total = 0.0
    per = []
    for k in buffer_keys:
        if not (k.endswith("running_mean") or k.endswith("running_var")):
            continue
        if k not in init_sd or k not in ft_sd:
            continue
        d = float((ft_sd[k].float() - init_sd[k].float()).norm().item())
        total += d
        per.append((k, d))
    per.sort(key=lambda r: r[1], reverse=True)
    return total, per


# --------------------------------------------------------------------------- #
# interpolation (Eq. 8) — params lerp un-masked (mask proven redundant by
# leak_check); buffers follow the chosen policy.
# --------------------------------------------------------------------------- #
def build_interpolated_sd(init_sd, ft_sd, gamma, param_keys, buffer_keys, bn_policy):
    new_sd = {}
    for k in param_keys:
        a, b = init_sd[k].float(), ft_sd[k].float()
        new_sd[k] = a + gamma * (b - a)
    for k in buffer_keys:
        if bn_policy == "tuned":
            new_sd[k] = ft_sd[k]
        elif bn_policy == "init":
            new_sd[k] = init_sd[k]
        elif bn_policy == "interp":
            a, b = init_sd[k], ft_sd[k]
            if a.is_floating_point() and b.is_floating_point():
                new_sd[k] = a.float() + gamma * (b.float() - a.float())
            else:
                # num_batches_tracked etc. — interpolation is meaningless; pin to tuned.
                new_sd[k] = b
        else:
            raise ValueError(bn_policy)
    return new_sd


# --------------------------------------------------------------------------- #
# evaluation: clean / adv / ood for one (already-loaded) bare model
# --------------------------------------------------------------------------- #
def make_clean_loader(args):
    if "CIFAR" in args.dataset:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        cls = torchvision.datasets.CIFAR10 if args.dataset == "CIFAR10" else torchvision.datasets.CIFAR100
        ds = cls(root="./data", train=False, download=True, transform=tf)
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        ds = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def eval_point(args, model, clean_loader, criterion, do_ood):
    """Returns dict {clean, adv, ood}. model is the bare backbone on args.device."""
    is_cifar = "CIFAR" in args.dataset
    _, clean = evaluate(args, model, clean_loader, criterion)
    adv = (evaluate_cifar_robustness if is_cifar else evaluate_tiny_robustness)(args, model)
    ood = None
    if do_ood:
        if is_cifar:
            ood = evaluate_cifar_corruption(args, model, data_dir=args.corruption_dir)["avg"]
        else:
            ood = evaluate_tiny_corruption(args, model, data_dir=args.corruption_dir, level=args.tiny_c_level)["avg"]
    return {"clean": clean, "adv": adv, "ood": ood}


def read_record_anchor(cell_dir):
    """experiment_record.json -> (final_test_acc, final_robust_acc) for the free
    gamma=1 cross-check. None if absent."""
    p = os.path.join(cell_dir, "experiment_record.json")
    if not os.path.isfile(p):
        return None, None
    try:
        with open(p) as f:
            fr = (json.load(f).get("final_result") or {})
        return fr.get("final_test_acc"), fr.get("final_robust_acc")
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# one cell (one seed checkpoint)
# --------------------------------------------------------------------------- #
def run_cell(args, cell_dir, gammas, criterion, clean_loader):
    init_p = os.path.join(cell_dir, "init_params.pth")
    ft_p = os.path.join(cell_dir, args.ft_ckpt)
    mask_p = os.path.join(cell_dir, "neuron_masks.pth")
    for p in (init_p, ft_p, mask_p):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} missing — is this a finished fine-tune cell?")

    init_sd = _load_sd(init_p)
    ft_sd = _load_sd(ft_p)
    masks, _ = _load_masks(mask_p)

    # fresh bare model; we overwrite its tensors at every gamma.
    model = create_model(args.model, args.input_size, args.num_classes,
                         args.device, args.patch, resume=None)
    param_keys = [n for n, _ in model.named_parameters()]
    buffer_keys = [n for n, _ in model.named_buffers()]

    # every state_dict key must be classifiable as param or buffer
    missing = [k for k in ft_sd if k not in param_keys and k not in buffer_keys]
    if missing:
        print(f"[WARN] {len(missing)} checkpoint keys are neither param nor buffer of "
              f"the model (e.g. {missing[:3]}); they will be ignored.")
    param_keys = [k for k in param_keys if k in ft_sd and k in init_sd]
    buffer_keys = [k for k in buffer_keys if k in ft_sd and k in init_sd]

    # ---- detail (b): leak check (un-masked lerp is only valid if this passes) ----
    max_leak, worst, rows = leak_check(init_sd, ft_sd, masks, args.tol)
    print(f"\n[{os.path.basename(cell_dir)}] mask provenance: neuron_masks.pth "
          f"({len(masks)} keys, M̃=1⟺frozen).")
    print(f"  leak check  ‖(θ_tuned-θ_init)⊙M̃‖_max = {max_leak:.3e}  (tol {args.tol:.1e})"
          + (f"  worst: {worst}" if worst else ""))
    if rows:
        top = rows[0]
        print(f"  loudest layer: {top[0]}  frozen_leak={top[1]:.3e}  trainable_move={top[2]:.3e}")
    if max_leak > args.tol:
        msg = (f"  LEAK > tol: a frozen weight moved during training. The ⊙(1-M̃) factor "
               f"is NOT redundant here — investigate (BN-affine outside the mask? wd on a "
               f"frozen tensor?) before trusting the curve.")
        if args.force:
            print("  [FORCED]" + msg)
        else:
            print("[ABORT]" + msg + "  (re-run with --force to scan anyway.)")
            sys.exit(2)

    # ---- detail (c): BN drift (decides gamma=0 vs AT-anchor agreement) ----
    drift_total, drift_per = bn_buffer_drift(init_sd, ft_sd, buffer_keys)
    print(f"  BN buffer drift  Σ‖Δrunning‖ = {drift_total:.3e}  "
          f"(buffers={args.bn_buffers}; gamma=0 == AT baseline only up to this).")

    # ---- AT anchor: full theta_initial (init weights + init buffers) ----
    anchor_sd = {k: init_sd[k] for k in (param_keys + buffer_keys)}
    model.load_state_dict(anchor_sd, strict=False)
    model.to(args.device)
    at_anchor = eval_point(args, model, clean_loader, criterion, args.ood)
    print(f"  AT anchor (full θ_init):  clean {at_anchor['clean']:.2f}  adv {at_anchor['adv']:.2f}"
          + (f"  ood {at_anchor['ood']:.2f}" if at_anchor['ood'] is not None else ""))

    # ---- gamma sweep ----
    rec_clean, rec_robust = read_record_anchor(cell_dir)
    points = {}
    for g in gammas:
        sd = build_interpolated_sd(init_sd, ft_sd, g, param_keys, buffer_keys, args.bn_buffers)
        model.load_state_dict(sd, strict=False)
        model.to(args.device)
        pt = eval_point(args, model, clean_loader, criterion, args.ood)
        points[round(g, 4)] = pt
        line = f"  gamma={g:4.2f}  clean {pt['clean']:6.2f}  adv {pt['adv']:6.2f}"
        if pt["ood"] is not None:
            line += f"  ood {pt['ood']:6.2f}"
        print(line)

    # ---- two free endpoint checks ----
    g0 = points[round(gammas[0], 4)]
    g1 = points[round(gammas[-1], 4)]
    if abs(gammas[0]) < 1e-9:
        dc = g0["clean"] - at_anchor["clean"]
        da = g0["adv"] - at_anchor["adv"]
        print(f"  [endpoint] gamma=0 vs AT anchor: Δclean={dc:+.2f}  Δadv={da:+.2f}  "
              f"(=BN-drift effect under --bn_buffers {args.bn_buffers})")
    if abs(gammas[-1] - 1.0) < 1e-9 and rec_clean is not None:
        dc = g1["clean"] - rec_clean
        da = (g1["adv"] - rec_robust) if rec_robust is not None else float("nan")
        print(f"  [endpoint] gamma=1 vs experiment_record final_result: "
              f"Δclean={dc:+.2f}  Δadv={da:+.2f}  (should be ~0: runner is main-table-faithful)")

    return {
        "cell": cell_dir,
        "bn_buffers": args.bn_buffers,
        "leak_max": max_leak,
        "bn_drift_total": drift_total,
        "at_anchor": at_anchor,
        "record_anchor": {"clean": rec_clean, "robust": rec_robust},
        "points": {str(k): v for k, v in points.items()},
    }


# --------------------------------------------------------------------------- #
# aggregation across cells (seeds)
# --------------------------------------------------------------------------- #
def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def aggregate(cell_results, gammas):
    agg = {}
    for g in gammas:
        key = str(round(g, 4))
        agg[key] = {}
        for metric in ("clean", "adv", "ood"):
            vals = [cr["points"][key][metric] for cr in cell_results if key in cr["points"]]
            m, s = _mean_std(vals)
            agg[key][metric] = {"mean": m, "std": s, "n": len([v for v in vals if v is not None])}
    return agg


def _fmt(m, s, nd=2):
    if m is None:
        return "   -   "
    return f"{m:.{nd}f}±{s:.{nd}f}" if s else f"{m:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="gamma-consolidation ablation runner (Eq. 8)")
    ap.add_argument("--cell", action="append", required=True,
                    help="fine-tune cell dir (repeat once per seed checkpoint)")
    ap.add_argument("--model", default="ResNet18")
    ap.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    ap.add_argument("--num_classes", default=10, type=int)
    ap.add_argument("--input_size", default=32, type=int)
    ap.add_argument("--patch", default=4, type=int)
    ap.add_argument("--batch_size", default=256, type=int)
    ap.add_argument("--num_workers", default=(2 if os.name == "nt" else 8), type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ft_ckpt", default="best_params.pth",
                    help="tuned checkpoint filename inside each cell (theta_tuned)")
    ap.add_argument("--gammas", default=None,
                    help="comma list, e.g. '0,0.5,1' (default: 0.0..1.0 step 0.1)")
    ap.add_argument("--bn_buffers", default="tuned", choices=["tuned", "init", "interp"],
                    help="BN running_mean/var policy across gamma (detail c). "
                         "tuned: follow theta_tuned everywhere (gamma=1 == main table, exact). "
                         "interp: blend (both endpoints exact, interior buffers blended). "
                         "init: follow theta_initial (gamma=0 == AT exact, gamma=1 misses main table).")
    ap.add_argument("--no-ood", dest="ood", action="store_false", default=True,
                    help="skip CIFAR-10-C / Tiny-ImageNet-C (faster smoke test)")
    ap.add_argument("--corruption_dir", default=None,
                    help="corruption set dir (default ./data/CIFAR-10-C for CIFAR10, "
                         "./data/CIFAR-100-C for CIFAR100, ./data/Tiny-ImageNet-C for Tiny)")
    ap.add_argument("--tiny_c_level", default=1, type=int)
    ap.add_argument("--tol", default=1e-5, type=float, help="leak-check tolerance (detail b)")
    ap.add_argument("--force", action="store_true", help="scan even if the leak check fails")
    ap.add_argument("--out", default=None, help="JSON dump path")
    args = ap.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = "cpu"
    if args.corruption_dir is None:
        args.corruption_dir = {"CIFAR10": "./data/CIFAR-10-C",
                               "CIFAR100": "./data/CIFAR-100-C",
                               "TinyImageNet": "./data/Tiny-ImageNet-C"}[args.dataset]

    gammas = ([float(x) for x in args.gammas.split(",")] if args.gammas
              else [round(0.1 * i, 4) for i in range(11)])
    gammas = sorted(gammas)

    criterion = nn.CrossEntropyLoss()
    clean_loader = make_clean_loader(args)

    title = f"GAMMA-CONSOLIDATION ABLATION  ({args.model}/{args.dataset})  bn_buffers={args.bn_buffers}"
    print(title)
    print("=" * len(title))
    print("Eq.(8): theta = theta_init + gamma*(theta_tuned - theta_init)  [params, mask proven redundant]")
    print(f"gammas: {gammas}\nmetrics: clean / adv(PGD-10 eps=8/255) / ood(CIFAR-C avg)\n")

    cell_results = []
    for cell in args.cell:
        cell_results.append(run_cell(args, cell, gammas, criterion, clean_loader))

    agg = aggregate(cell_results, gammas)

    # ----------------------------- summary table ----------------------------- #
    print("\n" + "-" * 60)
    print(f"SUMMARY  (mean±std over {len(cell_results)} cell(s))")
    print(f"{'gamma':>6s} | {'clean%':>11s} {'adv%':>11s} {'ood%':>11s}")
    print("-" * 60)
    for g in gammas:
        k = str(round(g, 4))
        c, a, o = agg[k]["clean"], agg[k]["adv"], agg[k]["ood"]
        tag = ""
        if abs(g) < 1e-9:
            tag = "  <- AT baseline anchor"
        elif abs(g - 1.0) < 1e-9:
            tag = "  <- MAIN TABLE (no consolidation)"
        print(f"{g:6.2f} | {_fmt(c['mean'], c['std']):>11s} {_fmt(a['mean'], a['std']):>11s} "
              f"{_fmt(o['mean'], o['std']):>11s}{tag}")
    print("-" * 60)

    # headline read: does any gamma<1 beat gamma=1 on adv by more than seed noise?
    g1k = str(round(gammas[-1], 4))
    adv1_m = agg[g1k]["adv"]["mean"]
    adv1_s = agg[g1k]["adv"]["std"] or 0.0
    best_g, best_adv = None, -1.0
    for g in gammas[:-1]:
        m = agg[str(round(g, 4))]["adv"]["mean"]
        if m is not None and m > best_adv:
            best_adv, best_g = m, g
    if adv1_m is not None and best_adv >= 0:
        margin = best_adv - adv1_m
        verdict = ("no gamma<1 beats gamma=1 beyond noise -> consolidation is an optional knob, "
                   "the gain is the NRC selection."
                   if margin <= adv1_s + 1e-9 else
                   f"gamma={best_g} beats gamma=1 by {margin:.2f} (>{adv1_s:.2f} std) on adv -> "
                   "report this; the main table is NOT the best consolidation point.")
        print(f"[read] best gamma<1 on adv = {best_g} ({best_adv:.2f}); gamma=1 = {adv1_m:.2f}"
              f"±{adv1_s:.2f}. {verdict}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": {"model": args.model, "dataset": args.dataset,
                                  "bn_buffers": args.bn_buffers, "gammas": gammas,
                                  "ood": args.ood, "tol": args.tol},
                       "cells": cell_results, "aggregate": agg}, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
