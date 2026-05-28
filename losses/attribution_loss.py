"""
Entailment-cone loss for attribution (HySAC-style).

Given:
  - image hyperbolic embeddings  x_img: (B, D)        space components
  - anchor hyperbolic embeddings x_anc: (K, D)        one per class
  - integer labels               y:     (B,)          in [0, K)

For each (image i, anchor k) we compute:
  ξ_{ik}  = oxy_angle(x_anc[k], x_img[i])   exterior angle at the anchor
  ψ_k     = half_aperture(x_anc[k])         half-aperture of anchor k's cone

Image i with label y_i should be INSIDE the cone of anchor y_i:
  L_pos = max(0, ξ_{i,y_i} - ψ_{y_i})

And OUTSIDE the cones of all other anchors:
  L_neg = max(0, ψ_k + margin - ξ_{ik})     for k ≠ y_i

  L = mean_i L_pos_i + λ_neg * mean_{i, k≠y_i} L_neg_{ik}
"""
from __future__ import annotations

import torch
import torch.nn as nn

from geometry.lorentz import half_aperture, oxy_angle


class EntailmentConeLoss(nn.Module):
    def __init__(
        self,
        curv: float = 1.0,
        min_radius: float = 0.1,
        margin: float = 0.1,
        lambda_neg: float = 1.0,
        lambda_norm: float = 0.0,
        target_norm: float = 0.0,
    ):
        """
        lambda_norm, target_norm: optional anchor-norm regulariser.
          L_norm = mean_c max(0, target_norm - ‖t_c‖)²
        Forces anchor space-components to grow past target_norm. This breaks the
        half_aperture clamp (which triggers when ‖t‖ < 2·min_radius / √curv) and
        actually narrows the entailment cones. With lambda_norm=0 (default) the
        term is skipped.
        """
        super().__init__()
        self.curv = curv
        self.min_radius = min_radius
        self.margin = margin
        self.lambda_neg = lambda_neg
        self.lambda_norm = lambda_norm
        self.target_norm = target_norm

    def forward(
        self,
        x_img: torch.Tensor,   # (B, D)
        x_anc: torch.Tensor,   # (K, D)
        labels: torch.Tensor,  # (B,) int in [0, K)
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, _ = x_img.shape
        K, _ = x_anc.shape
        device = x_img.device

        # Half-aperture per anchor: (K,)
        psi = half_aperture(x_anc, curv=self.curv, min_radius=self.min_radius)

        # Pairwise exterior angle ξ[i, k] = oxy_angle(anchor_k, image_i).
        # oxy_angle is element-wise, so tile both sides to (B*K, D).
        x_anc_tiled = x_anc.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1)
        x_img_tiled = x_img.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        xi = oxy_angle(x_anc_tiled, x_img_tiled, curv=self.curv).reshape(B, K)

        psi_b = psi.unsqueeze(0).expand(B, K)

        pos_idx = labels.unsqueeze(1)                                  # (B, 1)
        xi_pos  = xi.gather(1, pos_idx).squeeze(1)                     # (B,)
        psi_pos = psi_b.gather(1, pos_idx).squeeze(1)                  # (B,)
        loss_pos = torch.clamp(xi_pos - psi_pos, min=0.0)

        neg_mask = torch.ones(B, K, device=device, dtype=torch.bool)
        neg_mask.scatter_(1, pos_idx, False)
        viol_neg = torch.clamp(psi_b + self.margin - xi, min=0.0)
        loss_neg = viol_neg[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=device)

        # Anchor norm regulariser: pull ‖t_c‖ above target so half_aperture is no longer clamped.
        anc_norms = x_anc.norm(dim=-1)                                  # (K,)
        if self.lambda_norm > 0 and self.target_norm > 0:
            loss_norm = torch.clamp(self.target_norm - anc_norms, min=0.0).pow(2).mean()
        else:
            loss_norm = torch.tensor(0.0, device=device)

        loss = loss_pos.mean() + self.lambda_neg * loss_neg + self.lambda_norm * loss_norm

        with torch.no_grad():
            inside = (xi_pos < psi_pos).float().mean()                 # in-cone rate for positives
            pred   = xi.argmin(dim=1)                                   # closest cone
            acc    = (pred == labels).float().mean()

        stats = {
            "loss_pos":     loss_pos.mean().detach(),
            "loss_neg":     loss_neg.detach() if isinstance(loss_neg, torch.Tensor) else torch.tensor(0.0, device=device),
            "loss_norm":    loss_norm.detach(),
            "inside_pos":   inside.detach(),
            "cone_acc":     acc.detach(),
            "mean_psi":     psi.mean().detach(),
            "mean_xi_pos":  xi_pos.mean().detach(),
            "mean_anc_norm": anc_norms.mean().detach(),
        }
        return loss, stats


def predict_class(x_img: torch.Tensor, x_anc: torch.Tensor, curv: float = 1.0) -> torch.Tensor:
    """Image-only inference: pick the anchor with smallest exterior angle."""
    B, _ = x_img.shape
    K, _ = x_anc.shape
    x_anc_tiled = x_anc.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1)
    x_img_tiled = x_img.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    xi = oxy_angle(x_anc_tiled, x_img_tiled, curv=curv).reshape(B, K)
    return xi.argmin(dim=1)
