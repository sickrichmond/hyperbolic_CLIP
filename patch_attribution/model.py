"""Whole image + 3x3 patch grid, all pushed into the SAME class cone.

One hyperbolic space, one projection head, one anchor set: a patch of a FLUX
image is still a FLUX image, so it belongs in the same cone as the whole frame.
Ten views per sample instead of one exploit the volume of hyperbolic space and
force the representation to be consistent across sub-views, which a feature
built on local high-frequency detail cannot be.

Views are cut from the 224x224 tensor the dataloaders already produce, so no
dataset or eval adapter changes and no extra I/O — and the patches add no new
high-frequency content, they are purely spatial sub-views.
"""
import torch
import torch.nn.functional as F

from losses.attribution_loss import _pairwise_xi
from models.attribution_clip import AttributionCLIP


def patch_views(pixel_values: torch.Tensor, patch_size: int = 112) -> torch.Tensor:
    """(B, C, H, W) → (B, 10, C, H, W): the image itself plus a 3x3 grid of crops.

    The three grid offsets are 0, s, 2s with s = (H - patch_size) // 2, so
    patch_size=112 gives 50%-overlapping quarter-image windows and patch_size=75
    the (almost) disjoint ninths. Crops are resized back to the full input size,
    which is what CLIP's position embeddings expect.
    """
    B, C, H, W = pixel_values.shape
    if patch_size > min(H, W):
        raise ValueError(f"patch_size {patch_size} exceeds input {H}x{W}")
    sy, sx = (H - patch_size) // 2, (W - patch_size) // 2
    crops = [
        pixel_values[:, :, i:i + patch_size, j:j + patch_size]
        for i in (0, sy, 2 * sy)
        for j in (0, sx, 2 * sx)
    ]
    patches = torch.stack(crops, dim=1).flatten(0, 1)              # (B*9, C, p, p)
    patches = F.interpolate(patches, size=(H, W), mode="bicubic", align_corners=False)
    return torch.cat([pixel_values.unsqueeze(1), patches.view(B, 9, C, H, W)], dim=1)


def view_logits(x_views: torch.Tensor, x_anc: torch.Tensor, curv: float = 1.0) -> torch.Tensor:
    """(B, V, D) views + (K, D) anchors → (B, K) logits, the per-view mean of -xi.

    argmax over these reproduces the single-view decision rule of
    `losses.attribution_loss.predict_class` when V == 1.
    """
    B, V, D = x_views.shape
    xi = _pairwise_xi(x_anc, x_views.reshape(B * V, D), curv=curv).T    # (B*V, K)
    return -xi.view(B, V, -1).mean(dim=1)


class PatchAttributionCLIP(AttributionCLIP):
    """AttributionCLIP whose forward returns one hyperbolic point per view."""

    def __init__(self, *args, patch_size: int = 112, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size

    def encode_views(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, V, D_hyp). All views go through the shared backbone
        in a single batch."""
        views = patch_views(pixel_values, self.patch_size)
        B, V = views.shape[:2]
        x, _ = self.encode_image(views.flatten(0, 1))
        return x.view(B, V, -1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # DataParallel-friendly: input and output are both sliced/gathered on dim 0.
        return self.encode_views(pixel_values)
