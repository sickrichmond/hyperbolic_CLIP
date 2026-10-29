"""Image-class entailment-cone hinges with norm and geodesic anchor regularization.

Positive pairs pay max(0, xi-psi); wrong-class pairs pay max(0, psi+margin-xi).
The anchor penalty averages the hinges over all unique pairs, including zeros.
With family inputs, class-to-family hinges and a relative tangent-depth penalty
are added, and the anchor penalty acts exclusively on family anchors.
Geodesic separation depends on directions and depths; it does not constrain
apertures or guarantee non-overlapping cones.
Apertures depend on anchor depth. Prediction minimizes the exterior angle xi.
Inputs use Lorentz spatial coordinates; returned diagnostics are detached.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.lorentz import half_aperture, oxy_angle, pairwise_dist

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
        lambda_anchor: float = 0.0,
        target_norm: float = 0.0,
        geodesic_margin: float | None = None,
        norm_mode: str = "floor",
        neg_samples: int = 0,
        lambda_family: float = 0.0,
        lambda_family_depth: float = 0.0,
        family_depth_margin: float = 0.5,
    ):
        """Configure hinges, spatial anchor-norm and pairwise distance penalties.

        neg_samples=0 uses all wrong classes; otherwise subsample per image.
        The norm penalty is enabled when both lambda_norm and target_norm are positive.
        Positive lambda_anchor separates families when family inputs are provided,
        otherwise classes. This constructor defaults to zero; the CLI defaults to 0.2.
        The distance threshold is target_norm for classes, or target_norm/2 for
        families. An explicit geodesic_margin must match the active scope.
        Family losses use all classes each call; the relative depth loss is tangent-based.
        """
        super().__init__()
        if norm_mode not in ("floor", "bilateral"):
            raise ValueError("norm_mode must be 'floor' or 'bilateral'")
        if not math.isfinite(curv) or curv <= 0:
            raise ValueError("curv must be positive and finite")
        if not all(math.isfinite(v) and v >= 0 for v in
                   (lambda_anchor, target_norm, target_norm if geodesic_margin is None else geodesic_margin)):
            raise ValueError("Anchor weight, norm target and geodesic margin must be finite and non-negative")
        if not all(math.isfinite(v) and v >= 0 for v in
                   (lambda_family, lambda_family_depth, family_depth_margin)):
            raise ValueError("Family weights and depth margin must be finite and non-negative")
        self.curv = curv
        self.min_radius = min_radius
        self.margin = margin
        self.lambda_neg = lambda_neg
        self.lambda_norm = lambda_norm
        self.lambda_anchor = lambda_anchor
        self.target_norm = target_norm
        self.geodesic_margin = geodesic_margin
        self.norm_mode = norm_mode
        self.neg_samples = neg_samples
        self.lambda_family = lambda_family
        self.lambda_family_depth = lambda_family_depth
        self.family_depth_margin = family_depth_margin

    def forward(
        self,
        x_img: torch.Tensor,  # (B, D)
        x_anc: torch.Tensor,  # (K, D)
        labels: torch.Tensor,  # (B,) class indices
        *,
        x_fam: torch.Tensor | None = None,  # (F, D) Lorentz spatial coordinates
        family_of: torch.Tensor | None = None,  # (K,) indices in family order
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted loss and detached image/cone geometry diagnostics.

        loss_anchor is unweighted. Mean/max cosine statistics use all unique
        class anchor pairs, including negative similarities; both report zero for K=1.
        Family cosine diagnostics have separate keys; loss_anchor uses family pairs
        whenever family inputs are provided, even when family loss weights are zero.
        """
        B = x_img.shape[0]
        K = x_anc.shape[0]
        device = x_img.device
        if (x_fam is None) != (family_of is None):
            raise ValueError("Provide both x_fam and family_of")
        if x_fam is None and (self.lambda_family > 0 or self.lambda_family_depth > 0):
            raise ValueError("Family loss weights require family inputs")
        if x_fam is not None:
            if (x_fam.ndim != 2 or x_fam.shape[0] == 0 or
                    x_fam.shape[1] != x_anc.shape[1] or x_fam.device != device):
                raise ValueError("x_fam must be nonempty (F, D) on the image device")
            if (family_of.shape != (K,) or family_of.dtype != torch.long or
                    family_of.device != device or
                    bool(((family_of < 0) | (family_of >= len(x_fam))).any())):
                raise ValueError("family_of must contain one valid family index per class on the image device")
        geodesic_margin = self.target_norm / (2 if x_fam is not None else 1)
        if self.geodesic_margin is not None and self.geodesic_margin != geodesic_margin:
            raise ValueError("geodesic_margin must equal target_norm/2 for families, or target_norm for classes")
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
            
        L_family = L_family_depth = x_anc.new_zeros(())
        if x_fam is not None:
            psi_fam = half_aperture(x_fam, curv=self.curv, min_radius=self.min_radius)
            xi_family = _pairwise_xi(x_fam, x_anc, curv=self.curv).T
            xi_parent = xi_family.gather(1, family_of[:, None]).squeeze(1)
            L_family_pos = (xi_parent - psi_fam[family_of]).clamp_min(0).mean()
            wrong_family = torch.arange(len(x_fam), device=device)[None, :] != family_of[:, None]
            L_family_neg = ((psi_fam[None, :] + self.margin - xi_family)
                            .clamp_min(0)[wrong_family].mean()
                            if wrong_family.any() else xi.new_zeros(()))
            L_family = L_family_pos + self.lambda_neg * L_family_neg
            # Invert the radial exp_map0 formula; the old norm term stays spatial.
            class_depth = torch.asinh(math.sqrt(self.curv) * anc_norms) / math.sqrt(self.curv)
            family_depth = torch.asinh(math.sqrt(self.curv) * x_fam.norm(dim=-1)) / math.sqrt(self.curv)
            depth_gap = class_depth - family_depth[family_of]
            L_family_depth = (self.family_depth_margin - depth_gap).clamp_min(0).square().mean()

        anchors = x_anc if x_fam is None else x_fam
        anchor_count = len(anchors)
        L_anchor = x_anc.new_zeros(())
        if self.lambda_anchor > 0 and anchor_count > 1:
            # Keep acosh away from 1 even in float32, including self-pairs.
            distances = pairwise_dist(anchors, anchors, curv=self.curv, eps=1e-5)
            iu = torch.triu_indices(anchor_count, anchor_count, offset=1, device=device)
            distances = distances[iu[0], iu[1]]
            L_anchor = (geodesic_margin - distances).clamp_min(0).mean()

        loss = (L_img_in_class + self.lambda_norm * L_norm + self.lambda_anchor * L_anchor
                + self.lambda_family * L_family + self.lambda_family_depth * L_family_depth)

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
                "loss_anchor": L_anchor.detach(),
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
            if x_fam is not None:
                directions_fam = F.normalize(x_fam, dim=-1)
                fi = torch.triu_indices(len(x_fam), len(x_fam), offset=1, device=device)
                cos_fam = ((directions_fam @ directions_fam.T)[fi[0], fi[1]]
                           if len(x_fam) > 1 else x_fam.new_zeros(1))
                xi_img_family = _pairwise_xi(x_fam, x_img, curv=self.curv).T
                image_parent = family_of[labels]
                image_parent_xi = xi_img_family.gather(1, image_parent[:, None]).squeeze(1)
                image_depth = torch.asinh(math.sqrt(self.curv) * x_img.norm(dim=-1)) / math.sqrt(self.curv)
                stats.update({
                    "loss_family": L_family.detach(),
                    "loss_family_pos": L_family_pos.detach(),
                    "loss_family_neg": L_family_neg.detach(),
                    "loss_family_depth": L_family_depth.detach(),
                    "inside_class_family": (xi_parent <= psi_fam[family_of]).float().mean(),
                    "wrong_family_violations": ((xi_family < psi_fam[None, :] + self.margin)
                                                 [wrong_family].float().mean()
                                                 if wrong_family.any() else xi.new_zeros(())),
                    "inside_img_family": (image_parent_xi <= psi_fam[image_parent]).float().mean(),
                    "family_depth_min": family_depth.min(),
                    "family_depth_mean": family_depth.mean(),
                    "class_depth_mean": class_depth.mean(),
                    "image_depth_mean": image_depth.mean(),
                    "family_depth_gap_min": depth_gap.min(),
                    "family_depth_gap_mean": depth_gap.mean(),
                    "family_depth_violations": (depth_gap < self.family_depth_margin).float().mean(),
                    "image_class_depth_violations": (image_depth <= class_depth[labels]).float().mean(),
                    "family_psi_min_deg": torch.rad2deg(psi_fam.min()),
                    "family_psi_max_deg": torch.rad2deg(psi_fam.max()),
                    "mean_fam_cos_sim": cos_fam.mean(),
                    "max_fam_cos_sim": cos_fam.max(),
                    "family_xi_sat": (xi_family > math.pi - 5e-3).float().mean(),
                })
        return loss, stats


def predict_class(x_img: torch.Tensor, x_anc: torch.Tensor, curv: float = 1.0) -> torch.Tensor:
    """Image-only inference: pick the anchor with smallest exterior angle."""
    return _pairwise_xi(x_anc, x_img, curv=curv).T.argmin(dim=1)
