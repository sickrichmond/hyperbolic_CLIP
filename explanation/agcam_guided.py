"""AGCAM and Guided heatmaps for exterior-angle AttributionCLIP scores.

Backpropagate -xi(target) or min_other(xi)-xi(target) through vision attention.
AGCAM aggregates layers/heads; Guided uses the final layer. Return detached
min-max-normalized spatial heatmaps.

The forward path normalizes CLIP features, applies the projection and exp_map0,
and requests attention tensors. It does not apply fixed image-radius
normalization. Call outside no_grad/inference_mode with
eager attention and gradient-bearing model parameters.
encode_anchors accepts text prompts; checkpoint free anchors must be supplied
separately by a caller that supports them.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer

from geometry.lorentz import exp_map0, oxy_angle


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pairwise_angles(
    apex: torch.Tensor,   # (K, D)
    point: torch.Tensor,  # (B, D)
    curv: float,
) -> torch.Tensor:
    """Return xi[k, b] = oxy_angle(apex[k], point[b]).  Shape (K, B)."""
    K, D = apex.shape
    B, _ = point.shape
    apex_e  = apex.unsqueeze(1).expand(K, B, D).reshape(K * B, D)
    point_e = point.unsqueeze(0).expand(K, B, D).reshape(K * B, D)
    return oxy_angle(apex_e, point_e, curv=curv).reshape(K, B)


def _reduce(tensor: torch.Tensor, dim: int, mode: str) -> torch.Tensor:
    if mode == "sum":
        return tensor.sum(dim=dim)
    if mode == "mean":
        return tensor.mean(dim=dim)
    if mode == "max":
        return tensor.max(dim=dim).values
    raise ValueError(f"Unknown reduction mode: {mode!r}")


def _normalize_heatmap(h: torch.Tensor, pct: float = 99.0) -> torch.Tensor:
    """Robustly normalise to [0, 1], detach, move to CPU.

    Uses the [1, pct] percentile range instead of plain min-max: ViT attention
    maps (especially last-layer Guided and rolled-out Chefer) are often
    dominated by a single "attention sink" patch whose value dwarfs everything
    else.  Plain min-max would map that one patch to 1 and crush the entire
    rest of the map to ~0, hiding all real structure.  Clipping the top
    percentile first keeps the map readable while barely touching well-behaved
    maps like AGCAM (only the top ~1% of patches are clipped).
    """
    h = h.detach().cpu().float()
    flat = h.flatten()
    lo = torch.quantile(flat, 0.01)
    hi = torch.quantile(flat, pct / 100.0)
    return ((h - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)


def _patches_to_grid(mask: torch.Tensor) -> torch.Tensor:
    """Reshape flat (n_patches,) or (1, n_patches) tensor to (side, side)."""
    mask = mask.flatten()
    n    = mask.shape[0]
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise RuntimeError(
            f"Patch count {n} is not a perfect square.  "
            "Only square-grid ViT variants (e.g. ViT-B/32, ViT-L/14) are supported."
        )
    return mask.view(side, side)


# ---------------------------------------------------------------------------
# Anchor encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_anchors(
    model,                        # AttributionCLIP
    anchor_texts: list[str],
    tokenizer: CLIPTokenizer,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Encode class-anchor texts into hyperbolic space and return a detached
    (K, D_hyp) tensor.

    Anchors are fixed at inference time; detaching ensures that gradients
    during AGCAM/Guided do not propagate into the text encoder.

    Args:
        model:        AttributionCLIP (any device, any mode).
        anchor_texts: List of K strings — one per generator class.
        tokenizer:    CLIPTokenizer matching model.clip_name.
        device:       Target device for the returned tensor.

    Returns:
        x_anchors: (K, D_hyp) hyperbolic anchor embeddings, detached.
    """
    tok = tokenizer(
        anchor_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    input_ids      = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)
    x_anc, _ = model.encode_text(input_ids, attention_mask)
    return x_anc.detach()


# ---------------------------------------------------------------------------
# Forward pass that keeps the computation graph for attention tensors
# ---------------------------------------------------------------------------

def forward_with_attentions(
    model,                       # AttributionCLIP, must be in eval()
    pixel_values: torch.Tensor,  # (1, C, H, W)
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return hyperbolic embeddings and graph-connected layer attentions.

    Requests attention tensors from the vision encoder, normalizes projected
    CLIP features, applies the projection head and exp_map0. Unlike the model's
    image forward path, this helper does not apply fixed image-radius scaling.
    Call with gradients enabled and outside autocast; eval mode alone does not
    enable gradients on frozen parameters.

    Returns (1, D_hyp) embeddings and L attention tensors (1, heads, S, S),
    where S is the number of patches plus the CLS token.
    """
    vision_out = model.clip.vision_model(
        pixel_values=pixel_values,
        output_attentions=True,
        return_dict=True,
    )
    attentions = [a for a in vision_out.attentions if a is not None]
    if not attentions:
        raise RuntimeError(
            "The vision model returned no attention tensors.  "
            "Check that output_attentions=True is supported by the "
            "underlying CLIPVisionModel variant."
        )

    # --- replicate _clip_image -----------------------------------------------
    feats = model.clip.visual_projection(vision_out.pooler_output)
    feats = F.normalize(feats, dim=-1)

    # Cast features to float before the head; the caller must disable autocast.
    feats   = feats.float()
    tangent = model.projection(feats)
    x_hyp   = exp_map0(tangent, curv=model.curv)

    return x_hyp, attentions


# ---------------------------------------------------------------------------
# Exterior-angle score computation
# ---------------------------------------------------------------------------

def compute_score(
    x_hyp: torch.Tensor,               # (1, D_hyp) — in computation graph
    x_anchors: torch.Tensor,            # (K, D_hyp) — detached
    target_class: int,
    score_mode: Literal["angle", "margin"],
    curv: float,
) -> torch.Tensor:
    """Return a scalar exterior-angle score for backpropagation.

    "angle" returns negative target xi; "margin" returns the smallest other
    class xi minus target xi. A positive margin means the target ranks first,
    not that it contains the image or that the prediction is calibrated.
    x_hyp has shape (1, D); detached x_anchors has shape (K, D).
    """
    # xi shape: (K,) — one angle per class for the single image
    xi = _pairwise_angles(x_anchors, x_hyp, curv=curv).squeeze(-1)  # (K,)

    xi_target = xi[target_class]

    if score_mode == "angle":
        return -xi_target

    if score_mode == "margin":
        K   = xi.shape[0]
        idx = torch.arange(K, device=xi.device)
        xi_second = xi[idx != target_class].min()
        return xi_second - xi_target  # positive = target has the smallest exterior angle

    raise ValueError(f"Unknown score_mode: {score_mode!r}")


# ---------------------------------------------------------------------------
# AGCAM
# ---------------------------------------------------------------------------

def compute_agcam_heatmap(
    model,
    pixel_values: torch.Tensor,
    x_anchors: torch.Tensor,
    target_class: int,
    score_mode: Literal["angle", "margin"] = "margin",
    head_fusion: Literal["sum", "mean", "max"] = "sum",
    layer_fusion: Literal["sum", "mean", "max"] = "sum",
    apply_sigmoid: bool = True,
    curv: float | None = None,
) -> torch.Tensor:
    """
    Compute an AGCAM heatmap for a single image.

    Uses all transformer layers (vs Guided which uses only the last).
    Produces richer heatmaps but requires backpropagating through the full
    attention stack.

    Args:
        model:         AttributionCLIP in eval() mode.
        pixel_values:  (1, C, H, W) on the model device, fp32 recommended.
        x_anchors:     (K, D_hyp) detached class prototypes from encode_anchors().
        target_class:  Index of the class to explain.
        score_mode:    "angle" or "margin" — see compute_score().
        head_fusion:   Aggregation across attention heads ("sum"/"mean"/"max").
        layer_fusion:  Aggregation across transformer layers.
        apply_sigmoid: Sigmoid-normalise attention maps before weighting.
                       Matches the original AGCAM paper; disable to use raw
                       post-softmax attention weights directly.
        curv:          Curvature; defaults to model.curv.

    Returns:
        heatmap: (side, side) float tensor in [0, 1] on CPU.
                 For ViT-L/14 at 224 px this is (14, 14).
    """
    if curv is None:
        curv = model.curv

    # --- forward (OUTSIDE no_grad) -------------------------------------------
    x_hyp, attentions = forward_with_attentions(model, pixel_values)

    # --- score + backprop -----------------------------------------------------
    score = compute_score(x_hyp, x_anchors, target_class, score_mode, curv)
    gradients = torch.autograd.grad(
        score,
        attentions,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,   # safer than False when some layers are frozen
    )

    valid = [
        (attn, grad)
        for attn, grad in zip(attentions, gradients)
        if grad is not None
    ]
    if not valid:
        raise RuntimeError(
            "No attention gradient is available.  Check that:\n"
            "  1. forward_with_attentions() is called outside torch.no_grad()\n"
            "  2. LoRA parameters have requires_grad=True (default in eval mode)\n"
            "  3. x_anchors is detached so the graph terminates at x_hyp"
        )

    # --- AGCAM formulation ---------------------------------------------------
    # cls_attn[l, h, 0, :] = how much CLS token at layer l, head h
    #                         attends to each patch
    # cls_grad[l, h, 0, :] = gradient of score w.r.t. those attention weights
    cls_attn = torch.stack([a[0, :, 0:1, :] for a, _ in valid], dim=0)  # [L, H, 1, S]
    cls_grad = torch.stack([g[0, :, 0:1, :] for _, g in valid], dim=0)

    cls_grad = F.relu(cls_grad)               # keep only positive influence
    if apply_sigmoid:
        cls_attn = torch.sigmoid(cls_attn)    # normalise to [0, 1]

    mask = cls_grad * cls_attn                # [L, H, 1, S]
    mask = mask[:, :, :, 1:]                  # drop CLS column → [L, H, 1, n_patches]
    mask = _reduce(mask, dim=1, mode=head_fusion)    # [L, 1, n_patches]
    mask = _reduce(mask, dim=0, mode=layer_fusion)   # [1, n_patches]

    heatmap = _patches_to_grid(mask)
    return _normalize_heatmap(heatmap)


# ---------------------------------------------------------------------------
# Guided
# ---------------------------------------------------------------------------

def compute_guided_heatmap(
    model,
    pixel_values: torch.Tensor,
    x_anchors: torch.Tensor,
    target_class: int,
    score_mode: Literal["angle", "margin"] = "margin",
    head_fusion: Literal["sum", "mean", "max"] = "sum",
    apply_sigmoid: bool = True,
    curv: float | None = None,
) -> torch.Tensor:
    """
    Compute a Guided attribution heatmap for a single image.

    Uses only the last transformer layer — faster than AGCAM and often
    sufficient for localising the most salient attribution region.
    Lacks the multi-layer integration that gives AGCAM its global context.

    Args: same as compute_agcam_heatmap (no layer_fusion).

    apply_sigmoid: Sigmoid-squash the attention map before weighting (as in
                   AGCAM).  Recommended for Guided: the last layer's CLS
                   attention is heavily dominated by an "attention sink" patch,
                   and the sigmoid tames that spike so the rest of the map
                   stays informative.  Disable to use raw post-softmax weights.

    Returns:
        heatmap: (side, side) float tensor in [0, 1] on CPU.
    """
    if curv is None:
        curv = model.curv

    x_hyp, attentions = forward_with_attentions(model, pixel_values)
    last_attn = attentions[-1]  # (1, H, S, S)

    score = compute_score(x_hyp, x_anchors, target_class, score_mode, curv)
    (gradient,) = torch.autograd.grad(
        score,
        last_attn,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    if gradient is None:
        raise RuntimeError(
            "Last attention layer has no gradient.  "
            "Verify the computation graph includes this attention tensor."
        )

    # CLS-row, patch columns: [H, n_patches]
    cls_attn = last_attn[0, :, 0, 1:]
    cls_grad = gradient[0, :, 0, 1:]
    cls_grad = F.relu(cls_grad)
    if apply_sigmoid:
        cls_attn = torch.sigmoid(cls_attn)

    mask   = cls_grad * cls_attn                          # [H, n_patches]
    mask   = _reduce(mask, dim=0, mode=head_fusion)       # [n_patches]

    heatmap = _patches_to_grid(mask)
    return _normalize_heatmap(heatmap)


# ---------------------------------------------------------------------------
# Chefer et al. (transformer attribution / relevance rollout)
# ---------------------------------------------------------------------------

def compute_chefer_heatmap(
    model,
    pixel_values: torch.Tensor,
    x_anchors: torch.Tensor,
    target_class: int,
    score_mode: Literal["angle", "margin"] = "margin",
    start_layer: int = 0,
    curv: float | None = None,
) -> torch.Tensor:
    """
    Compute a Chefer-et-al. relevance heatmap for a single image.

    Implements the gradient-weighted attention rollout of
    Chefer, Gur & Wolf, "Transformer Interpretability Beyond Attention
    Visualization" (CVPR 2021) / "Generic Attention-model Explainability"
    (ICCV 2021).  Unlike AGCAM — which weights and fuses each layer's CLS
    attention independently — this method propagates a relevance matrix
    through the whole attention stack, modelling how information mixes across
    tokens layer by layer:

        R^(0)   = I                                   (S x S identity)
        A_bar   = mean_heads( relu(grad_l * attn_l) )  (S x S, per layer)
        R^(l)   = R^(l-1) + A_bar @ R^(l-1)

    The identity initialisation and the additive update account for the
    residual (skip) connections around each attention block.  The CLS row of
    the final R, restricted to the patch columns, is the per-patch relevance.

    This is generally a stronger, more faithful ViT explanation than raw
    attention or AGCAM, at the cost of the S x S matrix products.

    Args:
        model:         AttributionCLIP in eval() mode.
        pixel_values:  (1, C, H, W) on the model device, fp32 recommended.
        x_anchors:     (K, D_hyp) detached class prototypes from encode_anchors().
        target_class:  Index of the class to explain.
        score_mode:    "angle" or "margin" — see compute_score().
        start_layer:   Begin the rollout at this transformer layer (0 = all
                       layers).  Skipping very early layers sometimes sharpens
                       the map; the CVPR-2021 default is 0.
        curv:          Curvature; defaults to model.curv.

    Returns:
        heatmap: (side, side) float tensor in [0, 1] on CPU.
    """
    if curv is None:
        curv = model.curv

    # --- forward (OUTSIDE no_grad) -------------------------------------------
    x_hyp, attentions = forward_with_attentions(model, pixel_values)

    # --- score + backprop through every attention layer ----------------------
    score = compute_score(x_hyp, x_anchors, target_class, score_mode, curv)
    gradients = torch.autograd.grad(
        score,
        attentions,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    valid = [
        (attn, grad)
        for attn, grad in zip(attentions, gradients)
        if grad is not None
    ]
    if not valid:
        raise RuntimeError(
            "No attention gradient is available.  Check that:\n"
            "  1. forward_with_attentions() is called outside torch.no_grad()\n"
            "  2. LoRA parameters have requires_grad=True (default in eval mode)\n"
            "  3. x_anchors is detached so the graph terminates at x_hyp"
        )

    # --- relevance rollout ----------------------------------------------------
    S      = valid[0][0].shape[-1]              # tokens = n_patches + 1 (CLS)
    device = valid[0][0].device
    R = torch.eye(S, device=device, dtype=torch.float32)

    for layer_idx, (attn, grad) in enumerate(valid):
        if layer_idx < start_layer:
            continue
        # attn / grad: (1, H, S, S).  Gradient-weight the attention, keep only
        # positive contributions, then average over heads (Chefer's avg_heads).
        cam = (grad * attn)[0].float()          # (H, S, S)
        cam = F.relu(cam).mean(dim=0)           # (S, S)
        R = R + cam @ R                         # additive residual-aware update

    # CLS-row relevance over patch columns (drop the CLS self-relevance).
    relevance = R[0, 1:]                        # (n_patches,)

    heatmap = _patches_to_grid(relevance)
    return _normalize_heatmap(heatmap)


# ---------------------------------------------------------------------------
# Multi-class analysis
# ---------------------------------------------------------------------------

# Registry so callers (and the CLI) can dispatch by method name.
HEATMAP_METHODS = {
    "agcam":  compute_agcam_heatmap,
    "guided": compute_guided_heatmap,
    "chefer": compute_chefer_heatmap,
}


def explain_all_classes(
    model,
    pixel_values: torch.Tensor,
    x_anchors: torch.Tensor,
    class_names: list[str],
    method: Literal["agcam", "guided", "chefer"] = "agcam",
    score_mode: Literal["angle", "margin"] = "margin",
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Compute one heatmap per class and return {class_name: heatmap}.

    Running this for all classes lets you compare which image regions the
    model associates with each generator — e.g. faces for one GAN,
    background smoothness for a diffusion model.

    Note: each call to agcam/guided does a separate forward+backward pass,
    so this is K times more expensive than a single call.

    Args:
        model:       AttributionCLIP in eval() mode.
        pixel_values: (1, C, H, W).
        x_anchors:   (K, D_hyp) from encode_anchors().
        class_names: List of K generator names in the same order as x_anchors.
        method:      "agcam", "guided" or "chefer".
        score_mode:  Passed to the chosen method.
        **kwargs:    Additional keyword args forwarded to the method
                     (e.g. head_fusion, layer_fusion, apply_sigmoid for
                     agcam/guided; start_layer for chefer).

    Returns:
        Dict mapping each class name to its (side, side) heatmap.
    """
    fn = HEATMAP_METHODS[method]
    results: dict[str, torch.Tensor] = {}
    for c, name in enumerate(class_names):
        results[name] = fn(
            model=model,
            pixel_values=pixel_values,
            x_anchors=x_anchors,
            target_class=c,
            score_mode=score_mode,
            curv=model.curv,
            **kwargs,
        )
    return results
