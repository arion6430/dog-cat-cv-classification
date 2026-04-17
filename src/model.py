"""ResNet18 with ImageNet pretraining for binary cat/dog classification.

Default mode freezes the backbone and trains only the new classification head.
Call `unfreeze_backbone()` to additionally fine-tune the last residual block.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.config import NUM_CLASSES


def build_model(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def unfreeze_backbone(model: nn.Module, last_block_only: bool = True) -> None:
    """Unfreeze parameters for fine-tuning. Head stays trainable either way."""
    if last_block_only:
        for p in model.layer4.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True


def load_model(path, device: torch.device) -> nn.Module:
    model = build_model(pretrained=False, freeze_backbone=False)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
