"""Image-class entailment-cone hinges with optional anchor-norm regularization.

Positive pairs pay max(0, xi-psi); wrong-class pairs pay max(0, psi+margin-xi).
Total loss is positive + lambda_neg * negative + lambda_norm * norm_penalty.
Apertures depend on anchor depth. Prediction minimizes the exterior angle xi.
Inputs use Lorentz spatial coordinates; returned diagnostics are detached.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.lorentz import half_aperture, oxy_angle


def _pairwise_xi(apex: torch.Tensor, point: torch.Tensor, curv: float) -> torch.Tensor:
    """Pairwise oxy_angle: result[a, p] = oxy_angle(apex[a], point[p]).
    apex (A, D), point (P, D) → (A, P)."""
    A, D = apex.shape
    P, _ = point.shape
    apex_t  = apex.unsqueeze(1).expand(A, P, D).reshape(A * P, D)
    point_t = point.unsqueeze(0).expand(A, P, D).reshape(A * P, D)
    return oxy_angle(apex_t, point_t, curv=curv).reshape(A, P)


def _subsample(mask: torch.Tensor, k: int) -> torch.Tensor:
    """Keep k random True entries per row (all of them if the row has fewer)."""
    if k <= 0 or k >= mask.shape[1]:
        return mask
    noise = torch.rand(mask.shape, device=mask.device).masked_fill(~mask, -1.0)
    keep = noise.topk(k, dim=1).indices
    return torch.zeros_like(mask).scatter_(1, keep, True) & mask


class EntailmentConeLoss(nn.Module):
    def __init__(
        self,
        curv: float = 1.0,
        min_radius: float = 0.1,
        margin: float = 0.1,
        lambda_neg: float = 1.0,
        lambda_norm: float = 0.0,
        target_norm: float = 0.0,
        norm_mode: str = "floor",
        neg_samples: int = 0,
    ):
        """Configure hinges and a floor or bilateral spatial anchor-norm penalty.

        neg_samples=0 uses all wrong classes; otherwise subsample per image.
        The norm penalty is enabled when both lambda_norm and target_norm are positive.
        """
        super().__init__()
        if norm_mode not in ("floor", "bilateral"):
            raise ValueError("norm_mode must be 'floor' or 'bilateral'")
        self.curv = curv
        self.min_radius = min_radius
        self.margin = margin
        self.lambda_neg = lambda_neg
        self.lambda_norm = lambda_norm
        self.target_norm = target_norm
        self.norm_mode = norm_mode
        self.neg_samples = neg_samples

    def forward(
        self,
        x_img: torch.Tensor,  # (B, D)
        x_anc: torch.Tensor,  # (K, D)
        labels: torch.Tensor,  # (B,) class indices
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted loss and image/cone geometry diagnostics."""
        B = x_img.shape[0]
        K = x_anc.shape[0]
        device = x_img.device
        psi_anc = half_aperture(x_anc, curv=self.curv, min_radius=self.min_radius)
        xi = _pairwise_xi(x_anc, x_img, curv=self.curv).T
        xi_pos = xi.gather(1, labels.unsqueeze(1)).squeeze(1)
        psi_pos = psi_anc[labels]
        L_pos = (xi_pos - psi_pos).clamp_min(0).mean()

        neg_mask = torch.ones(B, K, device=device, dtype=torch.bool)
        neg_mask.scatter_(1, labels.unsqueeze(1), False)
        neg_mask = _subsample(neg_mask, self.neg_samples)
        L_neg = ((psi_anc.unsqueeze(0) + self.margin - xi).clamp_min(0)[neg_mask].mean()
                 if neg_mask.any() else xi.new_zeros(()))
        L_img_in_class = L_pos + self.lambda_neg * L_neg

        anc_norms = x_anc.norm(dim=-1)
        L_norm = xi.new_zeros(())
        if self.lambda_norm > 0 and self.target_norm > 0:
            deviation = self.target_norm - anc_norms
            if self.norm_mode == "floor":
                deviation = deviation.clamp_min(0)
            L_norm = deviation.square().mean()
        loss = L_img_in_class + self.lambda_norm * L_norm

        # Pairwise separation is measured, not optimized.
        with torch.no_grad():
            directions = F.normalize(x_anc, dim=-1)
            iu = torch.triu_indices(K, K, offset=1, device=device)
            angles = torch.arccos(
                (directions @ directions.T).clamp(-1 + 1e-6, 1 - 1e-6)[iu[0], iu[1]])
            stats = {
                "loss_img_in_cls": L_img_in_class.detach(),
                "loss_pos": L_pos.detach(),
                "loss_neg": L_neg.detach(),
                "loss_norm": L_norm.detach(),
                "inside_img": (xi_pos < psi_pos).float().mean(),
                "cone_acc": (xi.argmin(1) == labels).float().mean(),
                "xi_sat": (xi > math.pi - 5e-3).float().mean(),
                "mean_psi_anc": psi_anc.mean(),
                "psi_min_deg": torch.rad2deg(psi_anc.min()),
                "psi_max_deg": torch.rad2deg(psi_anc.max()),
                "sep_min_deg": torch.rad2deg(angles.min()),
                "sep_mean_deg": torch.rad2deg(angles.mean()),
                "sep_overlap": (angles < psi_anc[iu[0]] + psi_anc[iu[1]]).float().mean(),
                "mean_xi_img_anc": xi_pos.mean(),
                "mean_anc_norm": anc_norms.mean(),
            }
        return loss, stats


def predict_class(x_img: torch.Tensor, x_anc: torch.Tensor, curv: float = 1.0) -> torch.Tensor:
    """Image-only inference: pick the anchor with smallest exterior angle."""
    return _pairwise_xi(x_anc, x_img, curv=curv).T.argmin(dim=1)
