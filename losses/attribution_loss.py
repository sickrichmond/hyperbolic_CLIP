"""
Hierarchical entailment-cone loss for attribution (HySAC-style).

Hierarchy:
    class anchor (e.g. "A real image")               — broadest cone
        ⊃ augmented caption ("Real image of ...")    — narrower cone, content-specific
            ⊃ image embedding                         — leaf

Loss terms (all use the same cone-violation primitive):

  L_img_in_class:  image i must lie inside the cone of its class anchor y_i,
                   and outside the cones of all other class anchors.
                   (this is the term used at inference)

  L_cap_in_class:  caption i must lie inside the cone of its class anchor y_i,
                   and outside the cones of all other class anchors.

  L_img_in_cap:    image i must lie inside the cone of its OWN augmented caption,
                   and outside the cones of all other captions in the batch
                   (which differ in content and/or class).

For each term:
  L_pos = max(0, ξ_pos - ψ_pos)
  L_neg = max(0, ψ_neg + margin - ξ_neg)
where ξ = oxy_angle(apex, point) and ψ = half_aperture(apex).

Total:
  L = L_img_in_class
      + λ_cap_in_class * L_cap_in_class
      + λ_img_in_cap   * L_img_in_cap
      + λ_norm         * L_norm   (anchor norm regulariser)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from geometry.lorentz import half_aperture, oxy_angle


def _pairwise_xi(apex: torch.Tensor, point: torch.Tensor, curv: float) -> torch.Tensor:
    """Pairwise oxy_angle: result[a, p] = oxy_angle(apex[a], point[p]).
    apex (A, D), point (P, D) → (A, P)."""
    A, D = apex.shape
    P, _ = point.shape
    apex_t  = apex.unsqueeze(1).expand(A, P, D).reshape(A * P, D)
    point_t = point.unsqueeze(0).expand(A, P, D).reshape(A * P, D)
    return oxy_angle(apex_t, point_t, curv=curv).reshape(A, P)


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
    ):
        """
        lambda_cap_in_class > 0 and lambda_img_in_cap > 0 enable the hierarchical
        terms. They require x_cap to be passed to forward(). With both at 0
        (default) the loss reduces to image-in-class-anchor only.

        lambda_norm, target_norm: anchor-norm regulariser.
          L_norm = mean_c max(0, target_norm - ‖t_c‖)²
        """
        super().__init__()
        self.curv = curv
        self.min_radius = min_radius
        self.margin = margin
        self.lambda_neg = lambda_neg
        self.lambda_cap_in_class = lambda_cap_in_class
        self.lambda_img_in_cap = lambda_img_in_cap
        self.lambda_norm = lambda_norm
        self.target_norm = target_norm

    def _cone_term(
        self,
        xi_pos: torch.Tensor,      # (B,)        positive exterior angles
        psi_pos: torch.Tensor,     # (B,)        cone aperture at the positive apex
        xi_neg: torch.Tensor,      # (B, M)      exterior angles to all candidate apices
        psi_neg_b: torch.Tensor,   # (B, M)      cone apertures at candidate apices
        neg_mask: torch.Tensor,    # (B, M) bool only true where the apex is a negative
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (L_pos, L_neg) — both scalars."""
        L_pos = torch.clamp(xi_pos - psi_pos, min=0.0).mean()
        if neg_mask.any():
            L_neg = torch.clamp(psi_neg_b + self.margin - xi_neg, min=0.0)[neg_mask].mean()
        else:
            L_neg = torch.tensor(0.0, device=xi_pos.device)
        return L_pos, L_neg

    def forward(
        self,
        x_img: torch.Tensor,                     # (B, D)
        x_anc: torch.Tensor,                     # (K, D)
        labels: torch.Tensor,                    # (B,) int in [0, K)
        x_cap: torch.Tensor | None = None,       # (B, D) augmented captions, optional
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, _ = x_img.shape
        K, _ = x_anc.shape
        device = x_img.device

        psi_anc = half_aperture(x_anc, curv=self.curv, min_radius=self.min_radius)   # (K,)
        psi_anc_b = psi_anc.unsqueeze(0).expand(B, K)                                # (B, K)
        pos_idx   = labels.unsqueeze(1)                                              # (B, 1)
        psi_anc_pos = psi_anc_b.gather(1, pos_idx).squeeze(1)                        # (B,)

        neg_mask_anc = torch.ones(B, K, device=device, dtype=torch.bool)
        neg_mask_anc.scatter_(1, pos_idx, False)

        # ───── 1) L_img_in_class ─────────────────────────────────────────────
        # xi_ia[i, c] = oxy_angle(anchor_c, img_i)
        xi_ia = _pairwise_xi(x_anc, x_img, curv=self.curv).T                          # (B, K)
        xi_ia_pos = xi_ia.gather(1, pos_idx).squeeze(1)                               # (B,)
        L_imgcls_pos, L_imgcls_neg = self._cone_term(
            xi_ia_pos, psi_anc_pos, xi_ia, psi_anc_b, neg_mask_anc
        )
        L_img_in_class = L_imgcls_pos + self.lambda_neg * L_imgcls_neg

        with torch.no_grad():
            inside_img = (xi_ia_pos < psi_anc_pos).float().mean()
            cone_acc   = (xi_ia.argmin(dim=1) == labels).float().mean()

        # ───── 2 & 3) hierarchical caption-based terms (optional) ────────────
        L_cap_in_class = torch.tensor(0.0, device=device)
        L_img_in_cap   = torch.tensor(0.0, device=device)
        stats_extra = {}

        use_caps = (
            x_cap is not None
            and (self.lambda_cap_in_class > 0 or self.lambda_img_in_cap > 0)
        )
        if use_caps:
            # 2) L_cap_in_class — caption inside its class anchor's cone
            xi_ca = _pairwise_xi(x_anc, x_cap, curv=self.curv).T                      # (B, K)
            xi_ca_pos = xi_ca.gather(1, pos_idx).squeeze(1)
            L_capcls_pos, L_capcls_neg = self._cone_term(
                xi_ca_pos, psi_anc_pos, xi_ca, psi_anc_b, neg_mask_anc
            )
            L_cap_in_class = L_capcls_pos + self.lambda_neg * L_capcls_neg

            # 3) L_img_in_cap — image inside its OWN caption's cone; other batch
            #    captions act as negatives.
            psi_cap = half_aperture(x_cap, curv=self.curv, min_radius=self.min_radius)  # (B,)
            # xi_ic[i, j] = oxy_angle(cap_j, img_i)
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

        # ───── 4) Anchor-norm regulariser ────────────────────────────────────
        anc_norms = x_anc.norm(dim=-1)
        if self.lambda_norm > 0 and self.target_norm > 0:
            L_norm = torch.clamp(self.target_norm - anc_norms, min=0.0).pow(2).mean()
        else:
            L_norm = torch.tensor(0.0, device=device)

        loss = (
            L_img_in_class
            + self.lambda_cap_in_class * L_cap_in_class
            + self.lambda_img_in_cap   * L_img_in_cap
            + self.lambda_norm         * L_norm
        )

        stats = {
            "loss_img_in_cls": L_img_in_class.detach(),
            "loss_cap_in_cls": L_cap_in_class.detach(),
            "loss_img_in_cap": L_img_in_cap.detach(),
            "loss_norm":       L_norm.detach(),
            "inside_img":      inside_img.detach(),
            "cone_acc":        cone_acc.detach(),
            "mean_psi_anc":    psi_anc.mean().detach(),
            "mean_xi_img_anc": xi_ia_pos.mean().detach(),
            "mean_anc_norm":   anc_norms.mean().detach(),
            **stats_extra,
        }
        return loss, stats


def predict_class(x_img: torch.Tensor, x_anc: torch.Tensor, curv: float = 1.0) -> torch.Tensor:
    """Image-only inference: pick the anchor with smallest exterior angle."""
    xi = _pairwise_xi(x_anc, x_img, curv=curv).T   # (B, K)
    return xi.argmin(dim=1)
