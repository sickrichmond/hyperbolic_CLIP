"""Exterior-angle entailment-cone loss and prediction for attribution.

For each positive pair, use max(0, xi-psi) or xi squared. Negative pairs pay
max(0, psi+margin-xi), optionally subsampled per row. Inference minimizes xi.

The weighted objective combines image-in-class terms with optional caption
containment, spatial anchor-norm regularization, hyperbolic axis-ray distance,
pairwise angular separation, image-class CE, and family containment/CE.
Temperatures are learned through softplus. Returned statistics are detached."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.lorentz import axis_ray_dist, half_aperture, oxy_angle


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
        lambda_cap_in_class: float = 0.0,
        lambda_img_in_cap: float = 0.0,
        lambda_norm: float = 0.0,
        target_norm: float = 0.0,
        lambda_axis: float = 0.0,
        lambda_ce: float = 0.0,
        ce_tau_init: float = 1.0,
        lambda_hinge: float = 1.0,
        norm_mode: str = "floor",
        target_norm_family: float = 0.0,
        lambda_sep: float = 0.0,
        separation_margin: float = 0.0,
        theta_max: float = 150.0,
        lambda_family: float = 0.0,
        family_of: torch.Tensor | None = None,
        pos_mode: str = "hinge",
        neg_samples: int = 0,
    ):
        """Configure loss weights and geometry.

        Norm targets apply to Lorentz spatial coordinates. Caption terms require
        x_cap; family terms require x_fam and class-to-family indices. Positive
        CE/family weights create learned temperature parameters."""
        super().__init__()
        self.curv = curv
        self.min_radius = min_radius
        self.margin = margin
        self.lambda_neg = lambda_neg
        self.lambda_cap_in_class = lambda_cap_in_class
        self.lambda_img_in_cap = lambda_img_in_cap
        self.lambda_norm = lambda_norm
        self.target_norm = target_norm
        self.lambda_axis = lambda_axis
        self.lambda_ce = lambda_ce
        self.lambda_hinge = lambda_hinge
        self.norm_mode = norm_mode
        self.target_norm_family = target_norm_family
        self.lambda_sep = lambda_sep
        self.separation_margin = math.radians(separation_margin)
        self.theta_max = math.radians(theta_max)
        self.lambda_family = lambda_family
        self.pos_mode = pos_mode
        self.neg_samples = neg_samples
        self.register_buffer("family_of", family_of)
        if lambda_ce > 0:
            raw = torch.tensor(float(ce_tau_init)).expm1().clamp(min=1e-6).log()
            self.ce_tau_raw = nn.Parameter(raw)
        if lambda_family > 0:
            raw = torch.tensor(float(ce_tau_init)).expm1().clamp(min=1e-6).log()
            self.fam_tau_raw = nn.Parameter(raw)

    def _cone_term(
        self,
        xi_pos: torch.Tensor,      # (B,)        positive exterior angles
        psi_pos: torch.Tensor,     # (B,)        cone aperture at the positive apex
        xi_neg: torch.Tensor,      # (B, M)      exterior angles to all candidate apices
        psi_neg_b: torch.Tensor,   # (B, M)      cone apertures at candidate apices
        neg_mask: torch.Tensor,    # (B, M) bool only true where the apex is a negative
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (L_pos, L_neg) — both scalars."""
        if self.pos_mode == "axis":
            L_pos = xi_pos.pow(2).mean()
        else:
            L_pos = torch.clamp(xi_pos - psi_pos, min=0.0).mean()
        neg_mask = _subsample(neg_mask, self.neg_samples)
        if neg_mask.any():
            L_neg = torch.clamp(psi_neg_b + self.margin - xi_neg, min=0.0)[neg_mask].mean()
        else:
            L_neg = torch.tensor(0.0, device=xi_pos.device)
        return L_pos, L_neg

    def _sep_term(self, x_anc: torch.Tensor, psi_anc: torch.Tensor,
                  ang=None, iu=None):
        """Penalize pair angles below aperture sums plus margin or above theta_max.

        Optional precomputed angles and pair indices avoid repeated geometry in
        forward(); standalone callers can omit both."""
        if ang is None:
            K = x_anc.shape[0]
            d = F.normalize(x_anc, dim=-1)
            cos = (d @ d.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            iu = torch.triu_indices(K, K, offset=1, device=x_anc.device)
            ang = torch.arccos(cos[iu[0], iu[1]])
        need = psi_anc[iu[0]] + psi_anc[iu[1]] + self.separation_margin
        floor = torch.clamp(need - ang, min=0.0).pow(2).mean()
        ceiling = torch.clamp(ang - self.theta_max, min=0.0).pow(2).mean()
        with torch.no_grad():
            stats = {
                "sep_min_deg": torch.rad2deg(ang.min()).detach(),
                "sep_max_deg": torch.rad2deg(ang.max()).detach(),
                "sep_overlap": (ang < need).float().mean().detach(),
            }
        return floor + ceiling, stats

    def forward(
        self,
        x_img: torch.Tensor,                     # (B, D)
        x_anc: torch.Tensor,                     # (K, D)
        labels: torch.Tensor,                    # (B,) int in [0, K)
        x_cap: torch.Tensor | None = None,       # (B, D) augmented captions, optional
        x_fam: torch.Tensor | None = None,       # (F, D) family anchors, optional
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted loss and detached per-batch diagnostics.

        Inputs use Lorentz spatial coordinates. Captions pair row-wise with
        images; family_of maps each class anchor to its family anchor.
        """
        B, _ = x_img.shape
        K, _ = x_anc.shape
        device = x_img.device

        psi_anc = half_aperture(x_anc, curv=self.curv, min_radius=self.min_radius)   # (K,)
        psi_anc_b = psi_anc.unsqueeze(0).expand(B, K)                                # (B, K)
        pos_idx   = labels.unsqueeze(1)                                              # (B, 1)
        psi_anc_pos = psi_anc_b.gather(1, pos_idx).squeeze(1)                        # (B,)

        neg_mask_anc = torch.ones(B, K, device=device, dtype=torch.bool)
        neg_mask_anc.scatter_(1, pos_idx, False)

        xi_ia = _pairwise_xi(x_anc, x_img, curv=self.curv).T                          # (B, K)
        xi_ia_pos = xi_ia.gather(1, pos_idx).squeeze(1)                               # (B,)
        L_imgcls_pos, L_imgcls_neg = self._cone_term(
            xi_ia_pos, psi_anc_pos, xi_ia, psi_anc_b, neg_mask_anc
        )
        L_img_in_class = L_imgcls_pos + self.lambda_neg * L_imgcls_neg

        with torch.no_grad():
            inside_img = (xi_ia_pos < psi_anc_pos).float().mean()
            cone_acc   = (xi_ia.argmin(dim=1) == labels).float().mean()
            xi_sat = (xi_ia > math.pi - 5e-3).float().mean()

        # Optional caption-to-class and image-to-caption objectives.
        L_cap_in_class = torch.tensor(0.0, device=device)
        L_img_in_cap   = torch.tensor(0.0, device=device)
        stats_extra = {}

        use_caps = (
            x_cap is not None
            and (self.lambda_cap_in_class > 0 or self.lambda_img_in_cap > 0)
        )
        if use_caps:
            xi_ca = _pairwise_xi(x_anc, x_cap, curv=self.curv).T                      # (B, K)
            xi_ca_pos = xi_ca.gather(1, pos_idx).squeeze(1)
            L_capcls_pos, L_capcls_neg = self._cone_term(
                xi_ca_pos, psi_anc_pos, xi_ca, psi_anc_b, neg_mask_anc
            )
            L_cap_in_class = L_capcls_pos + self.lambda_neg * L_capcls_neg

            psi_cap = half_aperture(x_cap, curv=self.curv, min_radius=self.min_radius)  # (B,)
            xi_ic = _pairwise_xi(x_cap, x_img, curv=self.curv).T                       # (B, B)
            xi_ic_pos = xi_ic.diagonal()                                               # (B,)
            psi_cap_b = psi_cap.unsqueeze(0).expand(B, B)                              # (B, B)
            neg_mask_ic = ~torch.eye(B, dtype=torch.bool, device=device)
            L_imgcap_pos, L_imgcap_neg = self._cone_term(
                xi_ic_pos, psi_cap, xi_ic, psi_cap_b, neg_mask_ic
            )
            L_img_in_cap = L_imgcap_pos + self.lambda_neg * L_imgcap_neg

            with torch.no_grad():
                inside_cap     = (xi_ca_pos < psi_anc_pos).float().mean()
                inside_img_cap = (xi_ic_pos < psi_cap).float().mean()
                stats_extra = {
                    "inside_cap":      inside_cap.detach(),
                    "inside_img_cap":  inside_img_cap.detach(),
                    "mean_psi_cap":    psi_cap.mean().detach(),
                    "mean_xi_cap_anc": xi_ca_pos.mean().detach(),
                    "mean_xi_img_cap": xi_ic_pos.mean().detach(),
                    "mean_cap_norm":   x_cap.norm(dim=-1).mean().detach(),
                }

        # Norm targets are spatial-coordinate norms, not tangent radii.
        anc_norms = x_anc.norm(dim=-1)
        if self.lambda_norm > 0 and self.target_norm > 0:
            if self.norm_mode == "bilateral":
                L_norm = (anc_norms - self.target_norm).pow(2).mean()
                if x_fam is not None and self.target_norm_family > 0:
                    L_norm = L_norm + (
                        x_fam.norm(dim=-1) - self.target_norm_family
                    ).pow(2).mean()
            else:
                L_norm = torch.clamp(self.target_norm - anc_norms, min=0.0).pow(2).mean()
        else:
            L_norm = torch.tensor(0.0, device=device)

        # Hyperbolic distance to the outward ray starting at the correct anchor.
        L_axis = torch.tensor(0.0, device=device)
        if self.lambda_axis > 0:
            a_pos = x_anc[labels]                                            # (B, D)
            L_axis = axis_ray_dist(x_img, a_pos, curv=self.curv).mean()
            with torch.no_grad():
                frac_shallow = (x_img.norm(dim=-1) <= a_pos.norm(dim=-1)).float().mean()
                stats_extra.update({
                    "loss_axis":    L_axis.detach(),
                    "frac_shallow": frac_shallow.detach(),
                })

        # Share pairwise geometry between separation loss and diagnostics.
        _d = F.normalize(x_anc, dim=-1)
        _iu = torch.triu_indices(x_anc.shape[0], x_anc.shape[0], offset=1, device=device)
        sep_ang = torch.arccos((_d @ _d.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)[_iu[0], _iu[1]])

        L_sep = torch.tensor(0.0, device=device)
        if self.lambda_sep > 0:
            L_sep, sep_stats = self._sep_term(x_anc, psi_anc, sep_ang, _iu)
            stats_extra.update(sep_stats)

        # Model-anchor containment and image-to-family cross-entropy.
        L_family = torch.tensor(0.0, device=device)
        if self.lambda_family > 0 and x_fam is not None:
            psi_fam = half_aperture(x_fam, curv=self.curv, min_radius=self.min_radius)
            fam_of = self.family_of
            fam_labels = fam_of[labels]
            xi_mf = oxy_angle(x_fam[fam_of], x_anc, curv=self.curv)              # (K,)
            L_mf = torch.clamp(xi_mf - psi_fam[fam_of], min=0.0).mean()
            xi_if = _pairwise_xi(x_fam, x_img, curv=self.curv).T                 # (B, F)
            L_if = F.cross_entropy(-xi_if / F.softplus(self.fam_tau_raw), fam_labels)
            L_family = L_mf + L_if
            with torch.no_grad():
                stats_extra.update({
                    "loss_fam_anc":  L_mf.detach(),
                    "loss_fam_img":  L_if.detach(),
                    "inside_family": (xi_mf < psi_fam[fam_of]).float().mean().detach(),
                    "family_acc":    (xi_if.argmin(1) == fam_labels).float().mean().detach(),
                    "mean_psi_fam":  psi_fam.mean().detach(),
                    "fam_tau":       F.softplus(self.fam_tau_raw).detach(),
                })

        loss = (
            self.lambda_hinge          * L_img_in_class
            + self.lambda_cap_in_class * L_cap_in_class
            + self.lambda_img_in_cap   * L_img_in_cap
            + self.lambda_norm         * L_norm
            + self.lambda_axis         * L_axis
            + self.lambda_sep          * L_sep
            + self.lambda_family       * L_family
        )

        if self.lambda_ce > 0:
            tau = F.softplus(self.ce_tau_raw)
            L_ce = F.cross_entropy(-xi_ia / tau, labels)
            loss = loss + self.lambda_ce * L_ce
            stats_extra["loss_ce"] = L_ce.detach()
            stats_extra["ce_tau"]  = tau.detach()

        stats = {
            "loss_img_in_cls": L_img_in_class.detach(),
            "xi_sat":          xi_sat.detach(),
            "loss_pos":        L_imgcls_pos.detach(),
            "loss_neg":        L_imgcls_neg.detach(),
            "loss_cap_in_cls": L_cap_in_class.detach(),
            "loss_img_in_cap": L_img_in_cap.detach(),
            "loss_norm":       L_norm.detach(),
            "loss_sep":        L_sep.detach(),
            "inside_img":      inside_img.detach(),
            "cone_acc":        cone_acc.detach(),
            "mean_psi_anc":    psi_anc.mean().detach(),
            "psi_min_deg":     torch.rad2deg(psi_anc.min()).detach(),
            "psi_max_deg":     torch.rad2deg(psi_anc.max()).detach(),
            "sep_min_deg":     torch.rad2deg(sep_ang.min()).detach(),
            "sep_mean_deg":    torch.rad2deg(sep_ang.mean()).detach(),
            "mean_xi_img_anc": xi_ia_pos.mean().detach(),
            "mean_anc_norm":   anc_norms.mean().detach(),
            "sep_overlap":     (sep_ang < psi_anc[_iu[0]] + psi_anc[_iu[1]]
                                 + self.separation_margin
                                ).float().mean().detach(),
            **stats_extra,
        }
        return loss, stats


def predict_class(x_img: torch.Tensor, x_anc: torch.Tensor, curv: float = 1.0) -> torch.Tensor:
    """Image-only inference: pick the anchor with smallest exterior angle."""
    xi = _pairwise_xi(x_anc, x_img, curv=curv).T   # (B, K)
    return xi.argmin(dim=1)
