"""Cross-entropy classification against supplied unit-length class prototypes.

The caller normalizes image and anchor embeddings and supplies logit_scale.
Logits are min(exp(logit_scale), max_logit_scale) times their dot products.
The module owns no learned parameters and returns detached accuracy, cosine
and scale statistics alongside the loss. Prediction maximizes the dot product.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeCELoss(nn.Module):
    def __init__(self, max_logit_scale: float = 100.0):
        """Set the upper bound on the exponentiated logit scale."""
        super().__init__()
        self.max_logit_scale = max_logit_scale

    def forward(
        self,
        x_img: torch.Tensor,        # (B, D)  L2-normalised
        x_anc: torch.Tensor,        # (K, D)  L2-normalised
        labels: torch.Tensor,       # (B,) int in [0, K)
        logit_scale: torch.Tensor,  # scalar log inverse-temperature supplied by the model
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        scale = torch.clamp(logit_scale.exp(), max=self.max_logit_scale)
        logits = scale * (x_img @ x_anc.t())            # (B, K)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()
            cos = x_img @ x_anc.t()                      # (B, K) raw cosines
            cos_pos = cos.gather(1, labels.unsqueeze(1)).squeeze(1).mean()
            # mean cosine to the highest-scoring *wrong* anchor (margin proxy)
            cos_neg = cos.clone()
            cos_neg.scatter_(1, labels.unsqueeze(1), float("-inf"))
            cos_neg_max = cos_neg.max(dim=1).values.mean()

        stats = {
            "acc":          acc.detach(),
            "mean_cos_pos": cos_pos.detach(),
            "mean_cos_neg": cos_neg_max.detach(),
            "logit_scale":  scale.detach(),
        }
        return loss, stats


def predict_class(x_img: torch.Tensor, x_anc: torch.Tensor) -> torch.Tensor:
    """Image-only inference: pick the anchor with the highest cosine similarity."""
    return (x_img @ x_anc.t()).argmax(dim=1)
