"""
Robust-backbone loader for Neuron-RCES Stage-1.

Loads public robust checkpoints via RobustBench (with an opt-in non-robust timm
fallback for XCiT), optionally retargets the classifier head for a new label
space (e.g. Tiny-ImageNet 200), and reports metadata the rest of the pipeline
needs: whether the model self-normalizes, the Linf epsilon for its dataset, and
(for CIFAR) the data-condition footnote so the user can confirm "same condition".

Verified facts (see research spec):
  - load_model(model_name, model_dir='./models', dataset='cifar10'|'cifar100'|
    'imagenet', threat_model='Linf') -> nn.Module that expects [0,1] inputs.
  - ImageNet robust XCiT-S12 = 'Debenedetti2022Light_XCiT-S12' (eps=4/255).
  - No WRN-28-10 in RobustBench is free of BOTH extra-real AND synthetic data;
    'Pang2022Robustness_WRN28_10' is the lightest synthetic user (+1M).
"""

import os
import torch
import torch.nn as nn

_CIFAR_EPS = 8.0 / 255.0
_IMAGENET_EPS = 4.0 / 255.0

# Default RobustBench model_name per (arch, dataset). The Tiny-ImageNet XCiT row
# reuses the ImageNet-1k robust XCiT backbone (transfer; head reset to 200).
DEFAULT_MODEL_NAMES = {
    ("wrn-28-10", "cifar10"): "Pang2022Robustness_WRN28_10",
    ("wrn-28-10", "cifar100"): "Pang2022Robustness_WRN28_10",
    ("xcit-s12", "imagenet"): "Debenedetti2022Light_XCiT-S12",
    ("xcit-s12", "tinyimagenet"): "Debenedetti2022Light_XCiT-S12",
}

# Synthetic / extra-data disclosure tokens to scan for in the RobustBench footnote.
_EXTRA_DATA_TOKENS = ("ddpm", "edm", "synthetic", "generated", "1m", "10m",
                      "20m", "50m", "80m", "extra", "unlabeled", "additional")


def _unwrap(model):
    """Return the inner module (peels DataParallel)."""
    return model.module if hasattr(model, "module") else model


def eps_for_dataset(dataset):
    return _CIFAR_EPS if dataset in ("cifar10", "cifar100") else _IMAGENET_EPS


def _infer_num_classes(model):
    """Best-effort: out_features of the classifier ('head' or last nn.Linear)."""
    m = _unwrap(model)
    head = getattr(m, "head", None)
    if isinstance(head, nn.Linear):
        return int(head.out_features)
    last = None
    for _, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            last = mod
    return int(last.out_features) if last is not None else None


def reset_head(model, num_classes):
    """
    Replace the classifier with a fresh nn.Linear(num_features, num_classes).
    NOTE: the new head is randomly initialized and therefore NOT robust.
    Returns True on success.
    """
    m = _unwrap(model)

    # 1) timm-style models expose reset_classifier (XCiT does).
    if hasattr(m, "reset_classifier"):
        try:
            m.reset_classifier(num_classes=num_classes)
            return True
        except Exception:
            pass

    # 2) direct .head Linear.
    head = getattr(m, "head", None)
    if isinstance(head, nn.Linear):
        m.head = nn.Linear(head.in_features, num_classes)
        return True

    # 3) generic: replace the last nn.Linear found in the module tree.
    last_name, last_mod = None, None
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            last_name, last_mod = name, mod
    if last_mod is not None:
        parent = m
        *path, attr = last_name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, attr, nn.Linear(last_mod.in_features, num_classes))
        return True

    return False


def _read_model_info(rb_dataset, threat_model, model_name, model_dir):
    """
    Best-effort read of the RobustBench model_info JSON to recover the
    'footnote' (synthetic-data disclosure) and 'additional_data' flag.
    Returns (footnote_str_or_None, additional_data_bool_or_None).
    """
    candidates = []
    # Installed-package model_info (ships with the repo).
    try:
        import robustbench
        base = os.path.dirname(robustbench.__file__)
        candidates.append(os.path.join(base, "model_info", rb_dataset,
                                       threat_model, model_name + ".json"))
    except Exception:
        pass
    # Local model_dir cache (in case the user vendored model_info there).
    candidates.append(os.path.join(model_dir, "model_info", rb_dataset,
                                   threat_model, model_name + ".json"))
    import json
    for path in candidates:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                return info.get("footnote"), info.get("additional_data")
        except Exception:
            continue
    return None, None


def load_robust_model(arch, dataset, model_name=None, threat_model="Linf",
                      model_dir="./rb_models", num_classes=None,
                      allow_timm_fallback=False, xcit_state_dict=None,
                      logger=None):
    """
    Load a robust backbone and return (model, meta).

    Args:
      arch: 'wrn-28-10' | 'xcit-s12'
      dataset: 'cifar10' | 'cifar100' | 'tinyimagenet' (XCiT transfer) | 'imagenet'
      model_name: RobustBench model_name; if None a sensible default is chosen.
      num_classes: if set and != native, the head is reset (Tiny-ImageNet -> 200).
      allow_timm_fallback: if RobustBench is missing, load a CLEAN (non-robust)
                           timm XCiT (only for arch starting with 'xcit').
      xcit_state_dict: path to a robust XCiT state_dict for the fallback path.

    meta keys:
      self_normalizing, eps, native_num_classes, source, head_reset,
      model_name, footnote, additional_data
    """
    log = (logger.info if logger is not None else print)

    # Tiny-ImageNet rides on the ImageNet-1k robust backbone.
    rb_dataset = "imagenet" if dataset == "tinyimagenet" else dataset

    if model_name is None:
        model_name = DEFAULT_MODEL_NAMES.get((arch, dataset))
        if model_name is None:
            raise ValueError(
                f"No default model_name for (arch={arch}, dataset={dataset}); "
                f"pass --model_name explicitly.")

    meta = {
        "self_normalizing": True,
        "eps": eps_for_dataset(dataset),
        "native_num_classes": None,
        "source": None,
        "head_reset": False,
        "model_name": model_name,
        "footnote": None,
        "additional_data": None,
    }

    model = None
    try:
        from robustbench.utils import load_model
        log(f"[loader] robustbench.load_model(name={model_name}, "
            f"dataset={rb_dataset}, threat_model={threat_model}, dir={model_dir})")
        model = load_model(model_name=model_name, dataset=rb_dataset,
                           threat_model=threat_model, model_dir=model_dir)
        meta["source"] = "robustbench"
        meta["self_normalizing"] = True
    except ImportError:
        log("[loader][ERROR] RobustBench not installed. Install with:")
        log("    pip install git+https://github.com/RobustBench/robustbench.git")
        log("    (pulls timm>=1.0.9 + autoattack)")
        if not (allow_timm_fallback and arch.startswith("xcit")):
            raise
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for --allow_timm_fallback; "
                "install via: pip install timm")
        pretrained = xcit_state_dict is None
        log("[loader][WARN] timm fallback -> xcit_small_12_p16_224 "
            f"(pretrained={pretrained}). These are CLEAN/NON-ROBUST weights "
            "unless --xcit_state_dict supplies a robust checkpoint.")
        model = timm.create_model("xcit_small_12_p16_224", pretrained=pretrained)
        if xcit_state_dict is not None:
            sd = torch.load(xcit_state_dict, map_location="cpu")
            sd = sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd
            missing, unexpected = model.load_state_dict(sd, strict=False)
            log(f"[loader] loaded xcit_state_dict (missing={len(missing)}, "
                f"unexpected={len(unexpected)} keys)")
        # Plain timm XCiT is NOT self-normalizing -> dataloaders must add Normalize.
        meta["source"] = "timm_fallback"
        meta["self_normalizing"] = False

    meta["native_num_classes"] = _infer_num_classes(model)

    # Head surgery for a new label space (e.g. ImageNet-1k -> Tiny-ImageNet-200).
    if num_classes is not None and meta["native_num_classes"] != num_classes:
        ok = reset_head(model, num_classes)
        meta["head_reset"] = ok
        log(f"[loader] head reset {meta['native_num_classes']} -> {num_classes} "
            f"classes: success={ok}.  (fresh head is NON-robust; warm-up advised)")

    # Data-condition disclosure for the CIFAR WRN backbones.
    if rb_dataset in ("cifar10", "cifar100"):
        footnote, additional = _read_model_info(rb_dataset, threat_model,
                                                model_name, model_dir)
        meta["footnote"] = footnote
        meta["additional_data"] = additional
        fn_l = (footnote or "").lower()
        if additional or any(t in fn_l for t in _EXTRA_DATA_TOKENS):
            log("[DATA-CONDITION WARNING] "
                f"{model_name}: additional_data={additional}, footnote={footnote!r} "
                "-> uses extra/synthetic data; NOT a pure-CIFAR 'same-condition' "
                "starting point. Consider a pure-AT model (e.g. Wu2020Adversarial, "
                "Gowal2020Uncovering_70_16) or document the caveat.")
        elif footnote is None:
            log("[loader][note] could not read RobustBench model_info footnote "
                "(robustbench not importable or model_info missing); verify the "
                "data condition manually on the server.")

    return model, meta
