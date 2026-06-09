# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from models.densenet import DenseNet50, DenseNet121
from models.fcn import FCN
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet152
from models.vgg import VGG
from models.vit import ViT
from models.wide_resnet import WideResNet


def create_model(model_name, input_size, num_classes, device, patch_size=4, resume=None):
    if model_name == "ResNet34":
        model = ResNet34(input_size, num_classes)
    elif model_name == "ResNet18":
        model = ResNet18(input_size, num_classes)
    elif model_name == "ResNet50":
        model = ResNet50(input_size, num_classes)
    elif model_name == "ResNet152":
        model = ResNet152(input_size, num_classes)
    elif model_name == "DenseNet":
        model = DenseNet121(input_size, num_classes)
    elif model_name == "DenseNet50":
        model = DenseNet50(input_size, num_classes)
    elif model_name == "VGG19":
        model = VGG("VGG19", input_size, num_classes)
    elif model_name == "WideResNet34":
        model = WideResNet(image_size=input_size, depth=34, widen_factor=10, num_classes=num_classes)
    elif model_name == "WideResNet28":
        model = WideResNet(image_size=input_size, depth=28, widen_factor=10, num_classes=num_classes)
    elif model_name == "WideResNet22_2":
        model = WideResNet(image_size=input_size, depth=22, widen_factor=2, num_classes=num_classes)
    elif model_name == "WideResNet34_5":
        model = WideResNet(image_size=input_size, depth=34, widen_factor=5, num_classes=num_classes)
    elif model_name == "FCN":
        model = FCN()
    elif model_name == "ViT":
        model = ViT(
            image_size=input_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model.to(device)

    # Only wrap in DataParallel when there is genuinely more than one GPU.
    # On a single GPU, DataParallel is pure overhead: it replicates the model
    # and scatters/gathers every forward pass on the GIL-bound main thread,
    # which leaves the GPU idle between tiny kernel bursts ("memory not full,
    # throughput low"). The old `device == "cuda"` check also silently missed
    # "cuda:0". Numerically identical to the unwrapped model on one device.
    if str(device).startswith("cuda") and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        if "net" in checkpoint.keys():
            state = checkpoint["net"]
        elif "state_dict" in checkpoint.keys():
            state = checkpoint["state_dict"]
        elif "model" in checkpoint.keys():
            state = checkpoint["model"]
        else:
            state = checkpoint
        # Tolerate checkpoints saved with/without the DataParallel "module." prefix
        # regardless of how the current model is wrapped.
        model.load_state_dict(_align_state_dict_prefix(model, state))

    return model


def _align_state_dict_prefix(model, state):
    """Add or strip the leading 'module.' on every key so a checkpoint loads
    whether or not it (and the current model) was DataParallel-wrapped."""
    wrapped = any(k.startswith("module.") for k in model.state_dict().keys())
    has_prefix = any(k.startswith("module.") for k in state.keys())
    if wrapped and not has_prefix:
        return {f"module.{k}": v for k, v in state.items()}
    if not wrapped and has_prefix:
        return {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in state.items()}
    return state


