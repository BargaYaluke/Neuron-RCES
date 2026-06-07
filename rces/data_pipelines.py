"""
Dataloaders for the RobustBench Neuron-RCES pipeline.

KEY RULE: when the backbone self-normalizes (every RobustBench-loaded model),
transforms emit tensors in [0,1] with NO transforms.Normalize. Only the opt-in
plain-timm fallback (self_normalizing=False) re-adds ImageNet normalization.

CIFAR:        native 32x32, torchvision auto-download.
Tiny-ImageNet: native 64x64 -> upsampled to img_size (224 for XCiT) with BICUBIC
               to match the XCiT default_cfg (input 224, bicubic, crop_pct 1.0).
               Expects ./data/tiny-imagenet-200/{train,val} in ImageFolder layout
               (same assumption as the repo's existing evaluate_tiny_robustness).
"""

import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2471, 0.2435, 0.2616)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _maybe_norm(t_list, self_normalizing, mean, std):
    if not self_normalizing:
        t_list.append(transforms.Normalize(mean, std))
    return transforms.Compose(t_list)


# --------------------------------------------------------------------------- #
# CIFAR-10 / CIFAR-100
# --------------------------------------------------------------------------- #
def cifar_loaders(dataset, batch_size, self_normalizing=True,
                  num_workers=8, root="./data"):
    cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100

    train_tf = _maybe_norm(
        [transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()],
        self_normalizing, _CIFAR_MEAN, _CIFAR_STD)
    test_tf = _maybe_norm(
        [transforms.ToTensor()],
        self_normalizing, _CIFAR_MEAN, _CIFAR_STD)

    train = cls(root=root, train=True, download=True, transform=train_tf)
    test = cls(root=root, train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# Tiny-ImageNet (transfer target for the ImageNet-robust XCiT)
# --------------------------------------------------------------------------- #
def tiny_imagenet_loaders(batch_size, img_size=224, self_normalizing=True,
                          num_workers=8, root="./data/tiny-imagenet-200"):
    train_tf = _maybe_norm(
        [transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BICUBIC),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()],
        self_normalizing, _IMAGENET_MEAN, _IMAGENET_STD)
    test_tf = _maybe_norm(
        [transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
         transforms.CenterCrop(img_size),
         transforms.ToTensor()],
        self_normalizing, _IMAGENET_MEAN, _IMAGENET_STD)

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Tiny-ImageNet train dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Tiny-ImageNet val dir not found: {val_dir} "
            "(expected ImageFolder layout: val/<wnid>/*.JPEG)")
    train = datasets.ImageFolder(train_dir, transform=train_tf)
    test = datasets.ImageFolder(val_dir, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def build_loaders(dataset, batch_size, self_normalizing=True, img_size=224,
                  num_workers=8):
    """Dispatch on dataset name. Returns (train_loader, test_loader)."""
    if dataset in ("cifar10", "cifar100"):
        return cifar_loaders(dataset, batch_size, self_normalizing, num_workers)
    if dataset == "tinyimagenet":
        return tiny_imagenet_loaders(batch_size, img_size, self_normalizing,
                                     num_workers)
    raise ValueError(f"Unsupported dataset for RobustBench pipeline: {dataset}")
