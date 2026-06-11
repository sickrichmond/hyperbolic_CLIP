"""
AGCAM and Guided attribution for AttributionCLIP.

Both methods backpropagate an attribution score through the CLIP vision
transformer to the per-layer attention maps (AGCAM) or only the last layer
(Guided), producing a spatial heatmap that highlights which image regions
drove the model's attribution decision.

Adaptation notes vs. the HySAC explanation pipeline
-----------------------------------------------------
* Score function: replaces the SVDD distance with an entailment-cone score
  derived from oxy_angle.  Two modes are provided:
    - "angle"  : directly measures image-in-cone membership for one class.
    - "margin" : oxy_angle(second-best anchor) - oxy_angle(target anchor).
                 Positive when the model is confident; recommended.
* Model accessor: HySAC exposed model.visual(...); here the chain is
  model.clip.vision_model → visual_projection → normalize → projection → exp_map0.
* fp32: to_hyperbolic already disables autocast; we replicate that here so
  the full backprop path stays numerically stable.
* LoRA on q_proj / v_proj: attention weights depend on LoRA-adapted queries,
  so gradients flow correctly even in eval() mode (LoRA params keep
  requires_grad=True at all times).

Usage
-----
    from explanation.agcam_guided import (
        encode_anchors,
        compute_agcam_heatmap,
        compute_guided_heatmap,
        explain_all_classes,
    )

    x_anchors = encode_anchors(model, anchor_texts, tokenizer, device)

    heatmap = compute_agcam_heatmap(
        model, pixel_values, x_anchors,
        target_class=0,          # e.g. "real"
        score_mode="margin",
    )
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


def _normalize_heatmap(h: torch.Tensor) -> torch.Tensor:
    """Min-max normalise to [0, 1], detach, move to CPU."""
    h = h.detach().cpu().float()
    h = h - h.min()
    return h / h.max().clamp_min(1e-8)


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
    """
    Run the AttributionCLIP visual pipeline and return both the hyperbolic
    embedding and all per-layer attention tensors, connected in the same
    computation graph so that AGCAM/Guided can backpropagate through them.

    IMPORTANT: call this function OUTSIDE any torch.no_grad() /
    torch.inference_mode() context.  LoRA adapter weights keep
    requires_grad=True in eval() mode, which is sufficient for autograd.

    Implementation mirrors AttributionCLIP._clip_image + to_hyperbolic but
    explicitly requests output_attentions=True.

    Returns:
        x_hyp:      (1, D_hyp) hyperbolic embedding, in the computation graph.
        attentions: list of L tensors each shaped (1, n_heads, S, S) where
                    S = n_patches + 1 (CLS token included).
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

    # --- replicate to_hyperbolic (fp32, no autocast) -------------------------
    # to_hyperbolic disables autocast to prevent fp16 NaN in sinh/acosh/asin.
    # We replicate that behaviour here so the full backprop path is in fp32.
    feats   = feats.float()
    tangent = model.projection(feats)
    x_hyp   = exp_map0(tangent, curv=model.curv)

    return x_hyp, attentions


# ---------------------------------------------------------------------------
# Score computation (replaces SVDD distance)
# ---------------------------------------------------------------------------

def compute_score(
    x_hyp: torch.Tensor,               # (1, D_hyp) — in computation graph
    x_anchors: torch.Tensor,            # (K, D_hyp) — detached
    target_class: int,
    score_mode: Literal["angle", "margin"],
    curv: float,
) -> torch.Tensor:
    """
    Compute a scalar score to backpropagate for attribution.

    "angle"
        Returns  -oxy_angle(anchor_target, x_hyp).
        Gradient points toward image regions that push the embedding inside
        the target entailment cone.  Simple and fast.

    "margin"
        Returns  oxy_angle(anchor_second_best, x_hyp) - oxy_angle(anchor_target, x_hyp).
        Positive when the model is confident; highlights what distinguishes
        the predicted generator from the closest alternative.
        Recommended for multi-class attribution.

    In both cases a larger score corresponds to a stronger attribution signal
    for target_class, matching the convention of the heatmap methods below.

    Args:
        x_hyp:        Hyperbolic image embedding (1, D_hyp), part of the graph.
        x_anchors:    Detached class prototypes (K, D_hyp).
        target_class: Class index to explain.
        score_mode:   "angle" or "margin".
        curv:         Curvature of the hyperbolic space.

    Returns:
        Scalar tensor (differentiable w.r.t. x_hyp and its upstream graph).
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
        return xi_second - xi_target  # positive = model is confident

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
    curv: float | None = None,
) -> torch.Tensor:
    """
    Compute a Guided attribution heatmap for a single image.

    Uses only the last transformer layer — faster than AGCAM and often
    sufficient for localising the most salient attribution region.
    Lacks the multi-layer integration that gives AGCAM its global context.

    Args: same as compute_agcam_heatmap (no layer_fusion, no apply_sigmoid).

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

    mask   = cls_grad * cls_attn                          # [H, n_patches]
    mask   = _reduce(mask, dim=0, mode=head_fusion)       # [n_patches]

    heatmap = _patches_to_grid(mask)
    return _normalize_heatmap(heatmap)


# ---------------------------------------------------------------------------
# Multi-class analysis
# ---------------------------------------------------------------------------

def explain_all_classes(
    model,
    pixel_values: torch.Tensor,
    x_anchors: torch.Tensor,
    class_names: list[str],
    method: Literal["agcam", "guided"] = "agcam",
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
        method:      "agcam" or "guided".
        score_mode:  Passed to the chosen method.
        **kwargs:    Additional keyword args forwarded to the method
                     (e.g. head_fusion, layer_fusion, apply_sigmoid).

    Returns:
        Dict mapping each class name to its (side, side) heatmap.
    """
    fn = compute_agcam_heatmap if method == "agcam" else compute_guided_heatmap
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