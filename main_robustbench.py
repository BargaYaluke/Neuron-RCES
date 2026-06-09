"""
Neuron-RCES Stage-1: neuron-level robust-criticality sparse adaptation on PUBLIC
robust backbones (RobustBench), extending the method from ResNet-18/34 to:

  * WRN-28-10        on CIFAR-10/100         (conv family, eps=8/255)
  * robust XCiT-S12  on Tiny-ImageNet        (Transformer family, eps=4/255,
                                              ImageNet-1k backbone, head -> 200)

This is a STANDALONE entry point (does not touch the original main.py). It loads
a robust backbone, accumulates adversarial robust-loss gradients, selects the k
least-robust-critical neurons per layer (conv channel / linear unit / attention
head), and runs cheap gradient-gated sparse adaptation, then evaluates clean +
PGD robustness.

CRITICAL: RobustBench models self-normalize. Inputs stay in [0,1] everywhere; the
model is attacked directly; PGD perturbs/clamps in [0,1]. No extra Normalize.

Typical usage
-------------
  # WRN-28-10 on CIFAR-10
  python main_robustbench.py --arch wrn-28-10 --dataset cifar10 \
      --neurons_per_layer 2 --epochs 10

  # robust XCiT-S12 transferred + adapted on Tiny-ImageNet (head -> 200)
  python main_robustbench.py --arch xcit-s12 --dataset tinyimagenet \
      --neurons_per_layer 2 --vit_neuron_granularity head \
      --head_warmup_epochs 2 --epochs 10

  # just print the module inventory + data-condition, then exit
  python main_robustbench.py --arch xcit-s12 --dataset tinyimagenet --diagnostic_only
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from rces.model_loader import load_robust_model
from rces.data_pipelines import build_loaders
from rces.diagnostics import print_module_inventory
from rces.neuron_mrc import (accumulate_adv_param_grads, compute_neuron_masks,
                             infer_num_heads)
from rces.train_loop import adapt_one_epoch, linear_probe_epochs
from rces.pgd_eval import (clean_accuracy, pgd_accuracy, autoattack_accuracy,
                           EPS_BY_DATASET)

_NUM_CLASSES = {"cifar10": 10, "cifar100": 100,
                "tinyimagenet": 200, "imagenet": 1000}


def create_logger(log_path):
    logger = logging.getLogger("neuron_rces_rb")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def parse_args():
    p = argparse.ArgumentParser(description="Neuron-RCES on RobustBench backbones")
    # backbone / data
    p.add_argument("--arch", default="wrn-28-10",
                   choices=["wrn-28-10", "xcit-s12"])
    p.add_argument("--dataset", default="cifar10",
                   choices=["cifar10", "cifar100", "tinyimagenet"])
    p.add_argument("--model_name", default=None,
                   help="RobustBench model_name (defaults chosen per arch/dataset)")
    p.add_argument("--threat_model", default="Linf", choices=["Linf", "L2"])
    p.add_argument("--model_dir", default="./rb_models")
    p.add_argument("--torch_hub_dir", default="./XCiT-S12-checkpoint",
                   help="persistent dir for the torch.hub weight cache (the XCiT "
                        "download), so it survives pod restarts. Weights land in "
                        "<dir>/checkpoints/. Pass '' to use the default ~/.cache.")
    p.add_argument("--num_classes", default=None, type=int,
                   help="override target #classes (default inferred from dataset)")
    p.add_argument("--img_size", default=224, type=int,
                   help="resize for Tiny-ImageNet -> XCiT (default 224)")
    p.add_argument("--allow_timm_fallback", action="store_true",
                   help="if robustbench missing, load CLEAN timm XCiT (non-robust)")
    p.add_argument("--xcit_state_dict", default=None,
                   help="robust XCiT state_dict for the timm-fallback path")
    # neuron-MRC
    p.add_argument("--neurons_per_layer", default=2, type=int,
                   help="k least-robust-critical neurons selected per layer")
    p.add_argument("--num_grad_batches", default=10, type=int)
    p.add_argument("--adv_steps", default=10, type=int,
                   help="PGD steps used when crafting adv samples for MRC")
    p.add_argument("--vit_neuron_granularity", default="unit",
                   choices=["unit", "head"])
    p.add_argument("--head_target_substrings", default="qkv",
                   help="comma-separated linear-name suffixes to group by head")
    p.add_argument("--per_proj_heads", action="store_true",
                   help="score Q/K/V heads separately (3*H candidates)")
    p.add_argument("--num_heads", default=None, type=int,
                   help="attention heads (auto-detected for XCiT if omitted)")
    p.add_argument("--always_train", default=None,
                   help="comma-separated substrings forced fully-trainable "
                        "(default 'head' when the head was reset)")
    # optimization
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--lr", default=0.001, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--wd", default=5e-4, type=float)
    p.add_argument("--head_warmup_epochs", default=0, type=int,
                   help="linear-probe the fresh head before sparse adaptation")
    p.add_argument("--head_warmup_lr", default=1e-3, type=float)
    p.add_argument("--amp", action="store_true", default=False,
                   help="enable torch.cuda.amp mixed precision in the adaptation loop "
                        "(fp16 compute; gradient mask stays exact). Off by default.")
    p.add_argument("--tf32", action="store_true", default=False,
                   help="allow TF32 tensor-core matmul/conv on Ampere+ (perturbs the fp32 "
                        "path / robust-acc reporting). Off by default.")
    # evaluation
    p.add_argument("--eps_override", default=None, type=float,
                   help="override Linf eps (default per dataset: cifar 8/255, "
                        "tiny/imagenet 4/255)")
    p.add_argument("--pgd_steps", default=50, type=int)
    p.add_argument("--pgd_restarts", default=1, type=int)
    p.add_argument("--eval_batches", default=None, type=int,
                   help="limit #batches in clean/PGD eval (None = full set)")
    p.add_argument("--robust_eval_interval", default=1, type=int)
    p.add_argument("--autoattack", action="store_true",
                   help="also run RobustBench AutoAttack benchmark if installed")
    p.add_argument("--autoattack_n", default=1000, type=int)
    # misc
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_workers", default=(2 if os.name == "nt" else 8), type=int,
                   help="DataLoader workers (Windows defaults lower: spawn is costly)")
    p.add_argument("--diagnostic_only", action="store_true",
                   help="print module inventory + data-condition, then exit")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        # Fixed input shapes (224 XCiT / 32 CIFAR) -> let cuDNN autotune conv algos.
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    save_dir = f"./results/{args.arch}_{args.dataset}/robustbench/"
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger(os.path.join(save_dir, "output.log"))
    logger.info(args)

    target_classes = (args.num_classes if args.num_classes is not None
                      else _NUM_CLASSES[args.dataset])

    # ------------------------------------------------------------------ #
    # 1) load robust backbone (+ optional head surgery)
    # ------------------------------------------------------------------ #
    model, meta = load_robust_model(
        arch=args.arch, dataset=args.dataset, model_name=args.model_name,
        threat_model=args.threat_model, model_dir=args.model_dir,
        num_classes=target_classes, allow_timm_fallback=args.allow_timm_fallback,
        xcit_state_dict=args.xcit_state_dict,
        torch_hub_dir=(args.torch_hub_dir or None), logger=logger)
    model = model.to(args.device)
    logger.info(f"[meta] {meta}")

    # ------------------------------------------------------------------ #
    # 2) diagnostics (always print; optionally stop here)
    # ------------------------------------------------------------------ #
    print_module_inventory(model, logger=logger)

    eps = args.eps_override if args.eps_override is not None else meta["eps"]
    logger.info(f"[eps] Linf eps for this run = {eps:.5f} "
                f"(dataset default {EPS_BY_DATASET.get(args.dataset)})")

    _print_caveats(args, meta, eps, logger)

    if args.diagnostic_only:
        logger.info("[diagnostic_only] inventory + caveats printed; exiting.")
        return

    # ------------------------------------------------------------------ #
    # 3) data
    # ------------------------------------------------------------------ #
    train_loader, test_loader = build_loaders(
        args.dataset, args.batch_size, self_normalizing=meta["self_normalizing"],
        img_size=args.img_size, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ #
    # 4) optional head warm-up (transfer case: fresh head is non-robust)
    # ------------------------------------------------------------------ #
    if meta["head_reset"] and args.head_warmup_epochs > 0:
        logger.info(f"[warmup] linear-probing fresh head for "
                    f"{args.head_warmup_epochs} epoch(s)...")
        linear_probe_epochs(model, train_loader, criterion, args.device,
                            epochs=args.head_warmup_epochs, lr=args.head_warmup_lr,
                            head_substrings=("head",), logger=logger)
    elif meta["head_reset"]:
        logger.info("[warmup][WARN] head was reset but --head_warmup_epochs=0; "
                    "the MRC estimate is crafted on a RANDOM (non-robust) head and "
                    "may be unreliable. Recommend --head_warmup_epochs >= 1.")

    # ------------------------------------------------------------------ #
    # 5) accumulate adversarial robust-loss gradients
    # ------------------------------------------------------------------ #
    accumulate_adv_param_grads(
        model, train_loader, args.device, eps=eps, steps=args.adv_steps,
        num_grad_batches=args.num_grad_batches, criterion=criterion, logger=logger)

    # ------------------------------------------------------------------ #
    # 6) compute neuron masks (granularity-aware)
    # ------------------------------------------------------------------ #
    num_heads = args.num_heads or infer_num_heads(model)
    head_subs = tuple(s for s in args.head_target_substrings.split(",") if s)
    if args.always_train is not None:
        always_train = tuple(s for s in args.always_train.split(",") if s)
    else:
        always_train = ("head",) if meta["head_reset"] else ()
    logger.info(f"[nrc] always_train_substrings={always_train}, "
                f"head_target_substrings={head_subs}, num_heads={num_heads}")

    new_masks, statistic, neuron_mrc_list, selected = compute_neuron_masks(
        model, neurons_per_layer=args.neurons_per_layer,
        vit_neuron_granularity=args.vit_neuron_granularity, num_heads=num_heads,
        head_target_substrings=head_subs, per_proj_heads=args.per_proj_heads,
        always_train_substrings=always_train, device=args.device, logger=logger,
        verbose=True)

    torch.save({"masks": new_masks, "statistic": statistic, "meta": meta},
               os.path.join(save_dir, "neuron_masks.pth"))
    np.save(os.path.join(save_dir, "neuron_mrc_list.npy"),
            np.array(neuron_mrc_list, dtype=object))

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args), "meta": meta, "eps": eps,
        "selection_phase": {
            "neurons_per_layer": args.neurons_per_layer,
            "granularity": args.vit_neuron_granularity,
            "num_heads": num_heads,
            "selected": {k: v for k, v in selected.items()},
            "trainable_pct": {k: (v[0] / max(v[1], 1)) for k, v in statistic.items()},
        },
        "results": [], "final_result": {},
    }
    record_path = os.path.join(save_dir, "experiment_record.json")
    _dump(record, record_path)

    # ------------------------------------------------------------------ #
    # 7) optimizer over all params (gradient gating enforces sparsity)
    # ------------------------------------------------------------------ #
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[max(args.epochs // 2, 1)], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.amp and str(args.device).startswith("cuda")))

    # ------------------------------------------------------------------ #
    # 8) initial eval
    # ------------------------------------------------------------------ #
    init_clean = clean_accuracy(model, test_loader, args.device, args.eval_batches)
    init_robust = pgd_accuracy(model, test_loader, args.device, eps,
                               steps=args.pgd_steps, restarts=args.pgd_restarts,
                               max_batches=args.eval_batches, logger=logger)
    logger.info(f"[init] clean {init_clean:.2f}%  robust {init_robust:.2f}%")
    record["init_result"] = {"clean_acc": init_clean, "robust_acc": init_robust}

    # ------------------------------------------------------------------ #
    # 9) sparse adaptation loop
    # ------------------------------------------------------------------ #
    best_robust = -1.0
    for epoch in range(args.epochs):
        train_loss, train_acc = adapt_one_epoch(
            model, train_loader, optimizer, criterion, new_masks,
            args.device, epoch, logger, scaler=scaler)
        clean_acc = clean_accuracy(model, test_loader, args.device, args.eval_batches)

        robust_acc = None
        if (epoch % args.robust_eval_interval == 0) or (epoch == args.epochs - 1):
            robust_acc = pgd_accuracy(model, test_loader, args.device, eps,
                                      steps=args.pgd_steps,
                                      restarts=args.pgd_restarts,
                                      max_batches=args.eval_batches, logger=logger)
        logger.info(f"[epoch {epoch}] train_acc {train_acc:.2f}% "
                    f"clean {clean_acc:.2f}% "
                    f"robust {('%.2f%%' % robust_acc) if robust_acc is not None else 'n/a'}")

        record["results"].append({
            "epoch": epoch, "train_loss": float(train_loss),
            "train_acc": float(train_acc), "clean_acc": float(clean_acc),
            "robust_acc": None if robust_acc is None else float(robust_acc)})
        _dump(record, record_path)

        if robust_acc is not None and robust_acc > best_robust:
            best_robust = robust_acc
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "clean_acc": clean_acc, "robust_acc": robust_acc},
                       os.path.join(save_dir, "best_params.pth"))
            logger.info(f"[epoch {epoch}] new best robust {best_robust:.2f}% saved.")

        scheduler.step()

    # ------------------------------------------------------------------ #
    # 10) final eval (+ optional AutoAttack)
    # ------------------------------------------------------------------ #
    final_clean = clean_accuracy(model, test_loader, args.device, args.eval_batches)
    final_robust = pgd_accuracy(model, test_loader, args.device, eps,
                                steps=args.pgd_steps, restarts=args.pgd_restarts,
                                max_batches=args.eval_batches, logger=logger)
    record["final_result"] = {
        "best_robust_acc": best_robust, "final_clean_acc": final_clean,
        "final_robust_acc": final_robust}
    logger.info(f"[final] clean {final_clean:.2f}%  robust {final_robust:.2f}%  "
                f"(best robust {best_robust:.2f}%)")

    if args.autoattack:
        aa = autoattack_accuracy(model, args.dataset, args.autoattack_n, eps,
                                 args.device, args.threat_model, logger=logger)
        if aa is not None:
            record["final_result"]["autoattack_clean"] = float(aa[0])
            record["final_result"]["autoattack_robust"] = float(aa[1])
            logger.info(f"[autoattack] clean {aa[0]*100:.2f}%  "
                        f"robust {aa[1]*100:.2f}%")

    _dump(record, record_path)
    logger.info(f"[done] artifacts in {save_dir}")


def _print_caveats(args, meta, eps, logger):
    logger.info("-" * 78)
    logger.info("[caveats] Read before reporting numbers:")
    if meta.get("head_reset"):
        logger.info("  * Transfer: the 200-class Tiny-ImageNet head is FRESH and "
                    "NOT robust -> this is 'transfer + sparse adaptation', not a "
                    "same-condition comparison. Use --head_warmup_epochs and report "
                    "it honestly.")
    if args.dataset == "tinyimagenet":
        logger.info("  * Resolution: Tiny-ImageNet 64 -> %d (bicubic) is a "
                    "distribution shift; absolute numbers are not directly "
                    "comparable to native-ImageNet/CIFAR." % args.img_size)
    logger.info(f"  * Eps differs by dataset: this run uses {eps:.5f} "
                "(CIFAR 8/255, Tiny/ImageNet 4/255). Footnote eps/steps/restarts.")
    if meta.get("footnote"):
        logger.info(f"  * CIFAR data condition: footnote={meta['footnote']!r} "
                    f"additional_data={meta.get('additional_data')}.")
    logger.info("  * No double-normalization: RobustBench models self-normalize; "
                "inputs are [0,1], the model is attacked directly.")
    logger.info("-" * 78)


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)


if __name__ == "__main__":
    main()
