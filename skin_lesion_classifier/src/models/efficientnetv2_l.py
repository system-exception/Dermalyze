"""
EfficientNetV2-L model module for skin lesion classification.

EfficientNetV2-L is the large variant offering high accuracy with improved
training speed compared to the original EfficientNet family.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_L_Weights

from .efficientnet import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SkinLesionClassifier,
    get_loss_function,
)


class SkinLesionClassifierV2L(SkinLesionClassifier):
    """EfficientNetV2-L based classifier for skin lesion classification."""

    def _create_backbone(self, pretrained: bool) -> tuple[nn.Module, int]:
        """Create the EfficientNetV2-L backbone."""
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_v2_l(weights=weights)
        feature_dim = 1280

        backbone.classifier = nn.Identity()
        return backbone, feature_dim


def create_model_v2l(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
    use_gradient_checkpointing: bool = False,
) -> SkinLesionClassifierV2L:
    """Factory function to create an EfficientNetV2-L skin lesion classifier."""
    return SkinLesionClassifierV2L(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


__all__ = [
    "SkinLesionClassifierV2L",
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "create_model_v2l",
    "get_loss_function",
]
