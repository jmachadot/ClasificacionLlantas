"""
Arquitecturas.

1. CustomCNN: una CNN implementada desde cero (requisito del enunciado: los
   componentes principales deben ser implementados por el estudiante).
2. Backbones por transfer learning (ResNet-18/50, EfficientNet-B0) con pesos
   preentrenados en ImageNet y cabeza de clasificación binaria reemplazada.

Todos los modelos producen UN solo logit (clasificación binaria con
BCEWithLogitsLoss / pérdida focal). La clase positiva es "damaged".
"""
import torch
import torch.nn as nn
from torchvision import models


# --------------------------------------------------------------------------- #
# CNN desde cero
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """CNN tipo VGG reducida, diseñada para captar textura superficial."""

    def __init__(self, num_outputs: int = 1, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),     # 224 -> 112
            ConvBlock(32, 64),    # 112 -> 56
            ConvBlock(64, 128),   # 56  -> 28
            ConvBlock(128, 256),  # 28  -> 14
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Constructor de modelos
# --------------------------------------------------------------------------- #
def build_model(name: str, pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    name = name.lower()

    if name == "custom":
        return CustomCNN(num_outputs=1, dropout=dropout)

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))
        return m

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))
        return m

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
        return m

    raise ValueError(f"Modelo desconocido: {name}")


def get_gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Devuelve la última capa convolucional, donde se calcula Grad-CAM."""
    name = name.lower()
    if name == "custom":
        # último ConvBlock -> última Conv2d dentro de su Sequential
        return model.features[-1].block[3]
    if name in ("resnet18", "resnet50"):
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Modelo desconocido para Grad-CAM: {name}")
