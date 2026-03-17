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

    if device == "cuda":
        model = torch.nn.DataParallel(model)

    if resume is not None:
        checkpoint = torch.load(resume)
        if "net" in checkpoint.keys():
            model.load_state_dict(checkpoint["net"])
        elif "state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint.keys():
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

    return model


