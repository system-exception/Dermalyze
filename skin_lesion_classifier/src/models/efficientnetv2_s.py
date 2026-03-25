"""
EfficientNetV2-S model module for skin lesion classification.

EfficientNetV2 improves training speed and parameter efficiency over the original
EfficientNet while maintaining or improving accuracy.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights

from .efficientnet import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SkinLesionClassifier,
    get_loss_function,
)


class SkinLesionClassifierV2S(SkinLesionClassifier):
    """EfficientNetV2-S based classifier for skin lesion classification."""

    def _create_backbone(self, pretrained: bool) -> tuple[nn.Module, int]:
        """Create the EfficientNetV2-S backbone."""
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_v2_s(weights=weights)
        feature_dim = 1280

        backbone.classifier = nn.Identity()
        return backbone, feature_dim


def create_model_v2s(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    use_gradient_checkpointing: bool = False,
) -> SkinLesionClassifierV2S:
    """Factory function to create an EfficientNetV2-S skin lesion classifier."""
    return SkinLesionClassifierV2S(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


__all__ = [
    "SkinLesionClassifierV2S",
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "create_model_v2s",
    "get_loss_function",
]
