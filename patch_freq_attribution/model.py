"""Image + 3x3 patches in one hyperbolic space, the FFT spectrum in a second one.

Two branches over ONE shared CLIP+LoRA backbone:

  pixel branch    10 views (whole image + 3x3 grid) → self.projection      → cones
  spectral branch 1 view (centred log-magnitude FFT) → self.projection_spec → cones

Separate spaces and separate anchors on purpose: a spectrum and a photograph do
not live on the same manifold, and forcing them into shared cones would only
inflate the cones until they stop discriminating.

The two logit vectors are summed with a learned per-branch temperature. Summing
logits is a product of experts, so a branch that is peaked on one class already
dominates a flat one — what has to be learned is the ratio of SCALES between the
branches, which is exactly what the two temperatures are.
"""
import torch
import torch.nn as nn

from data.spectral import spectrum
from geometry.lorentz import exp_map0
from patch_attribution.model import PatchAttributionCLIP, view_logits


class PatchFreqAttributionCLIP(PatchAttributionCLIP):
    def __init__(self, *args, init_scale: float = 0.1, **kwargs):
        super().__init__(*args, init_scale=init_scale, **kwargs)
        clip_dim = self.clip.base_model.model.config.projection_dim
        self.projection_spec = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.GELU(),
            nn.Linear(clip_dim, self.hyperbolic_dim),
        )
        # Same small init as the pixel head: without it sinh(||v||) in exp_map0
        # blows up on the first step.
        with torch.no_grad():
            self.projection_spec[-1].weight.mul_(init_scale)
            self.projection_spec[-1].bias.zero_()

    def _clip_spectrum(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """CLIP-space embedding of the image's spectrum (B, D_clip).

        With `--patch_source native` the input already carries the view axis; the
        spectrum is always taken on the whole-image view (index 0), never on a crop.
        """
        if pixel_values.dim() == 5:
            pixel_values = pixel_values[:, 0]
        return self._clip_image(spectrum(pixel_values))

    def encode_spectrum(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, D_hyp) in the SPECTRAL hyperbolic space."""
        feats = self._clip_spectrum(pixel_values)
        with torch.amp.autocast("cuda", enabled=False):
            return exp_map0(self.projection_spec(feats.float()), curv=self.curv)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Both outputs are sliced/gathered on dim 0, so DataParallel is happy.
        return self.encode_views(pixel_values), self.encode_spectrum(pixel_values)


def fused_logits(x_views, x_spec, x_anc_pix, x_anc_spec, tau, curv=1.0):
    """tau[0] * (pixel logits) + tau[1] * (spectral logits) → (B, K).

    tau holds the two RAW temperatures; they are exponentiated so they stay
    positive (raw 0 → temperature 1 → the plain sum).
    """
    t = tau.exp()
    return (t[0] * view_logits(x_views, x_anc_pix, curv=curv)
            + t[1] * view_logits(x_spec.unsqueeze(1), x_anc_spec, curv=curv))
