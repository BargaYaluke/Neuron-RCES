"""
Neuron-RCES Stage-1 (RobustBench backbones) helper package.

This package implements the "load a public robust backbone + cheap gradient-gated
sparse adaptation" pipeline used to extend Neuron-RCES from ResNet-18/34 to:
  - WRN-28-10        (CIFAR-10/100, conv family)        via RobustBench Model Zoo
  - robust XCiT-S12  (Transformer family, ImageNet)     via RobustBench / timm

CRITICAL invariant shared by every module here:
  RobustBench-loaded models normalize their inputs INTERNALLY. Therefore inputs
  must stay in [0, 1] everywhere (dataloaders use ToTensor only, no Normalize),
  the model is attacked DIRECTLY (no nn.Sequential(Normalize, model) wrapper),
  and PGD perturbs / clamps in [0, 1] pixel space. Doing otherwise double-
  normalizes and silently destroys clean + robust accuracy.

Mask convention (kept identical to the original main.py):
  mask == 0  -> trainable
  mask == 1  -> frozen
  gradient gating is `param.grad.mul_(1.0 - mask)`.
"""

from .model_loader import load_robust_model, reset_head, DEFAULT_MODEL_NAMES
from .neuron_mrc import accumulate_adv_param_grads, compute_neuron_masks
from .data_pipelines import cifar_loaders, tiny_imagenet_loaders
from .pgd_eval import clean_accuracy, pgd_accuracy, EPS_BY_DATASET
from .train_loop import adapt_one_epoch, apply_grad_mask, linear_probe_epochs
from .diagnostics import print_module_inventory

__all__ = [
    "load_robust_model", "reset_head", "DEFAULT_MODEL_NAMES",
    "accumulate_adv_param_grads", "compute_neuron_masks",
    "cifar_loaders", "tiny_imagenet_loaders",
    "clean_accuracy", "pgd_accuracy", "EPS_BY_DATASET",
    "adapt_one_epoch", "apply_grad_mask", "linear_probe_epochs",
    "print_module_inventory",
]
