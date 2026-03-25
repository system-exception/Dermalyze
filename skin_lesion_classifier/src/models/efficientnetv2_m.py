"""
EfficientNetV2-M model module for skin lesion classification.

EfficientNetV2-M is a medium-sized variant that offers a balance between
accuracy and computational efficiency.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_M_Weights

from .efficientnet import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SkinLesionClassifier,
    get_loss_function,
)


class SkinLesionClassifierV2M(SkinLesionClassifier):
    """EfficientNetV2-M based classifier for skin lesion classification."""

    def _create_backbone(self, pretrained: bool) -> tuple[nn.Module, int]:
        """Create the EfficientNetV2-M backbone."""
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_v2_m(weights=weights)
        feature_dim = 1280

        backbone.classifier = nn.Identity()
        return backbone, feature_dim


def create_model_v2m(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    use_gradient_checkpointing: bool = False,
) -> SkinLesionClassifierV2M:
    """Factory function to create an EfficientNetV2-M skin lesion classifier."""
    return SkinLesionClassifierV2M(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


__all__ = [
    "SkinLesionClassifierV2M",
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "create_model_v2m",
    "get_loss_function",
]
