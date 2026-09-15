"""Image-class entailment-cone hinges with anchor-norm and cosine regularization.

Positive pairs pay max(0, xi-psi); wrong-class pairs pay max(0, psi+margin-xi).
The cosine penalty averages max(0, cosine(a_i, a_j)) over unique anchor pairs.
Total loss is positive + lambda_neg * negative + lambda_norm * norm_penalty
              + lambda_cosine * cosine_penalty.
Cosine penalizes directions less than 90 degrees apart, independent of anchor
norms; it does not constrain apertures or guarantee non-overlapping cones.
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
        lambda_cosine: float = 0.0,
        target_norm: float = 0.0,
        norm_mode: str = "floor",
        neg_samples: int = 0,
    ):
        """Configure hinges, spatial anchor-norm and pairwise cosine penalties.

        neg_samples=0 uses all wrong classes; otherwise subsample per image.
        The norm penalty is enabled when both lambda_norm and target_norm are positive.
        Positive lambda_cosine enables the cosine penalty when at least two anchors
        exist. This constructor defaults to zero; the training CLI defaults to 0.2.
        """
        super().__init__()
        if norm_mode not in ("floor", "bilateral"):
            raise ValueError("norm_mode must be 'floor' or 'bilateral'")
        self.curv = curv
        self.min_radius = min_radius
        self.margin = margin
        self.lambda_neg = lambda_neg
        self.lambda_norm = lambda_norm
        self.lambda_cosine = lambda_cosine
        self.target_norm = target_norm
        self.norm_mode = norm_mode
        self.neg_samples = neg_samples

    def forward(
        self,
        x_img: torch.Tensor,  # (B, D)
        x_anc: torch.Tensor,  # (K, D)
        labels: torch.Tensor,  # (B,) class indices
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted loss and detached image/cone geometry diagnostics.

        loss_cosine is unweighted. Mean/max cosine statistics use all unique
        anchor pairs, including negative similarities; both report zero for K=1.
        """
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
            
        L_cosine = x_anc.new_zeros(())
        if self.lambda_cosine > 0 and K > 1:
            anc_dir = F.normalize(x_anc, dim=-1)
            cos_sim = anc_dir @ anc_dir.T
            iu = torch.triu_indices(K, K, offset=1, device=device)
            pairwise_cos = cos_sim[iu[0], iu[1]]
            L_cosine = pairwise_cos.clamp_min(0.0).mean()

        loss = L_img_in_class + self.lambda_norm * L_norm + self.lambda_cosine * L_cosine

        # These geometry diagnostics do not contribute gradients to the loss.
        with torch.no_grad():
            directions = F.normalize(x_anc, dim=-1)
            iu = torch.triu_indices(K, K, offset=1, device=device)
            cos_eval = (directions @ directions.T)[iu[0], iu[1]] if K > 1 else x_anc.new_zeros(1)
            angles = torch.arccos(cos_eval.clamp(-1 + 1e-6, 1 - 1e-6))
            
            stats = {
                "loss_img_in_cls": L_img_in_class.detach(),
                "loss_pos": L_pos.detach(),
                "loss_neg": L_neg.detach(),
                "loss_norm": L_norm.detach(),
                "loss_cosine": L_cosine.detach(),
                "mean_anc_cos_sim": cos_eval.mean(),
                "max_anc_cos_sim": cos_eval.max(),
                "inside_img": (xi_pos < psi_pos).float().mean(),
                "cone_acc": (xi.argmin(1) == labels).float().mean(),
                "xi_sat": (xi > math.pi - 5e-3).float().mean(),
                "mean_psi_anc": psi_anc.mean(),
                "psi_min_deg": torch.rad2deg(psi_anc.min()),
                "psi_max_deg": torch.rad2deg(psi_anc.max()),
                "sep_min_deg": torch.rad2deg(angles.min()) if K > 1 else x_anc.new_zeros(()),
                "sep_mean_deg": torch.rad2deg(angles.mean()) if K > 1 else x_anc.new_zeros(()),
                "sep_overlap": ((angles < psi_anc[iu[0]] + psi_anc[iu[1]]).float().mean() 
                                if K > 1 else x_anc.new_zeros(())),
                "mean_xi_img_anc": xi_pos.mean(),
                "mean_anc_norm": anc_norms.mean(),
            }
        return loss, stats


def predict_class(x_img: torch.Tensor, x_anc: torch.Tensor, curv: float = 1.0) -> torch.Tensor:
    """Image-only inference: pick the anchor with smallest exterior angle."""
    return _pairwise_xi(x_anc, x_img, curv=curv).T.argmin(dim=1)
