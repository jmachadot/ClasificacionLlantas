"""
Funciones de pérdida para clasificación binaria con un solo logit.

- BCEWithLogitsLoss (con pos_weight opcional para desbalance).
- Pérdida focal binaria (Lin et al., 2017), útil cuando la clase de daño es
  escasa y/o hay muchos ejemplos "fáciles" que dominan el gradiente.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        self.reduction = reduction

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1)

        ce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight,
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * (1 - p_t).pow(self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg, pos_weight_value: float = None, device="cpu"):
    pos_weight = None
    if cfg.imbalance_strategy == "class_weights" and pos_weight_value is not None:
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    if cfg.loss == "focal":
        return BinaryFocalLoss(
            alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, pos_weight=pos_weight
        ).to(device)

    # BCE estándar
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
