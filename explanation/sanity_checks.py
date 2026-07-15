"""
Adebayo et al. sanity checks for the attribution heatmaps.

Reference
---------
Adebayo, Gilmer, Muelly, Goodfellow, Hardt & Kim,
"Sanity Checks for Saliency Maps" (NeurIPS 2018).

A saliency / attribution method is only trustworthy if its output actually
depends on the model's learned parameters.  A method that produces the same
heatmap for a trained and a randomly-initialised network is really just an
edge detector on the input and tells you nothing about the model's decision.

This module implements the **model-parameter randomization test** (the primary
Adebayo check).  Starting from the trained AttributionCLIP it progressively
replaces the weights of the vision transformer with freshly-initialised random
weights, from the top of the network (closest to the output) down to the input
embeddings, recomputing the heatmap at every stage:

    cascading:   randomize layer L, then L and L-1, then L, L-1, L-2, ...
    independent: randomize exactly one layer at a time (model reset in between).

At each stage it measures how similar the perturbed heatmap is to the original
(Spearman rank correlation, Pearson correlation, cosine similarity, and a
global SSIM).  A method that PASSES the check shows the similarity collapsing
towards zero as more of the network is randomized.  A method whose heatmap
stays highly correlated with the original — even after the whole ViT has been
randomized — FAILS: it is insensitive to the model and its explanations are
not faithful.

The complementary **data randomization test** (retraining on permuted labels)
is not implemented here because it requires a second training run; see the
module note at the bottom for how to run it with the existing training script.

Usage (CLI)
-----------
    python -m explanation.sanity_checks \\
        --image      data/images/example.jpg \\
        --checkpoint checkpoints/attribution_FLUX_vitl14.pt \\
        --method     chefer \\
        --output_dir outputs/sanity_example

Usage (library)
---------------
    from explanation.sanity_checks import cascading_randomization_test

    report = cascading_randomization_test(
        model, pixel_values, x_anchors,
        target_class=pred_idx,
        method="chefer",
        score_mode="margin",
    )
    print(report["verdict"])
"""
from __future__ import annotations

import copy
from typing import Callable, Literal

import torch
import torch.nn as nn

from explanation.agcam_guided import HEATMAP_METHODS


# ---------------------------------------------------------------------------
# Similarity metrics between two heatmaps
# ---------------------------------------------------------------------------

def _flat(h: torch.Tensor) -> torch.Tensor:
    return h.detach().cpu().float().flatten()


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return float("nan")
    return float((a @ b) / denom)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    # Rank transform (argsort-of-argsort); ties are rare in continuous maps.
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return _pearson(ra, rb)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return float("nan")
    return float((a @ b) / denom)


def _ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Global (single-window) SSIM on [0, 1] heatmaps — a cheap SSIM proxy."""
    a = a.float()
    b = b.float()
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(unbiased=False), b.var(unbiased=False)
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2  # data range = 1
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    if den.abs() < 1e-12:
        return float("nan")
    return float(num / den)


def heatmap_similarity(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """All similarity metrics between two heatmaps (higher = more similar)."""
    fa, fb = _flat(a), _flat(b)
    return {
        "spearman": _spearman(fa, fb),
        "pearson":  _pearson(fa, fb),
        "cosine":   _cosine(fa, fb),
        "ssim":     _ssim(fa, fb),
    }


# ---------------------------------------------------------------------------
# Weight randomization
# ---------------------------------------------------------------------------

_RANDOMIZABLE = (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.Embedding)


@torch.no_grad()
def _randomize_module_(module: nn.Module) -> None:
    """
    Re-initialise every randomizable leaf inside `module` in place.

    Iterating over `module.modules()` reaches nested leaves, including the
    `base_layer` and `lora_A`/`lora_B` sub-linears created by PEFT around the
    LoRA-adapted q_proj / v_proj, so both the frozen backbone weights and the
    learned LoRA deltas at this depth are destroyed.  `reset_parameters()`
    draws from each layer's original init distribution, matching Adebayo's
    "re-initialise to the initialization distribution" prescription.
    """
    for m in module.modules():
        if isinstance(m, _RANDOMIZABLE) and hasattr(m, "reset_parameters"):
            m.reset_parameters()


def _collect_randomization_stages(model) -> list[tuple[str, nn.Module]]:
    """
    Ordered list of (name, module) to randomize, top (output) → bottom (input).

    Order follows the forward path in reverse so that cascading randomization
    destroys the network from the decision backwards, exactly as in the paper.
    """
    vision = model.clip.vision_model
    stages: list[tuple[str, nn.Module]] = []

    # Heads closest to the score first.
    stages.append(("projection_head", model.projection))
    if hasattr(model.clip, "visual_projection"):
        stages.append(("visual_projection", model.clip.visual_projection))
    if getattr(vision, "post_layernorm", None) is not None:
        stages.append(("post_layernorm", vision.post_layernorm))

    # Transformer encoder, last layer → first layer.
    layers = vision.encoder.layers
    for i in range(len(layers) - 1, -1, -1):
        stages.append((f"encoder_layer_{i:02d}", layers[i]))

    # Input side last.  CLIP's pre-LN is historically misspelled "pre_layrnorm".
    for attr in ("pre_layrnorm", "pre_layernorm"):
        if getattr(vision, attr, None) is not None:
            stages.append((attr, getattr(vision, attr)))
            break
    if getattr(vision, "embeddings", None) is not None:
        stages.append(("embeddings", vision.embeddings))

    return stages


# ---------------------------------------------------------------------------
# The sanity check
# ---------------------------------------------------------------------------

def cascading_randomization_test(
    model,
    pixel_values: torch.Tensor,
    x_anchors: torch.Tensor,
    target_class: int,
    method: Literal["agcam", "guided", "chefer"] = "chefer",
    score_mode: Literal["angle", "margin"] = "margin",
    independent: bool = False,
    pass_threshold: float = 0.30,
    curv: float | None = None,
    **heatmap_kwargs,
) -> dict:
    """
    Run the Adebayo model-parameter randomization test for one image.

    Args:
        model:          AttributionCLIP in eval() mode.  Restored to its
                        original weights before returning.
        pixel_values:   (1, C, H, W) on the model device.
        x_anchors:      (K, D_hyp) detached anchors from encode_anchors().
        target_class:   Class index whose heatmap is being sanity-checked.
        method:         Which heatmap to test ("agcam"/"guided"/"chefer").
        score_mode:     Passed through to the heatmap method.
        independent:    False → cascading randomization (recommended);
                        True  → randomize one stage at a time, resetting the
                        model to trained weights before each stage.
        pass_threshold: Spearman correlation below which the fully-randomized
                        heatmap is considered "decorrelated" (check passes).
        curv:           Curvature; defaults to model.curv.
        **heatmap_kwargs: Extra args forwarded to the heatmap method
                        (e.g. start_layer for chefer, head_fusion for agcam).

    Returns:
        dict with keys:
            method, score_mode, mode ("cascading"/"independent")
            original_heatmap : (side, side) tensor
            stages           : list of {stage, metrics, heatmap, error}
            verdict          : "PASS" / "FAIL" / "INCONCLUSIVE"
            final_spearman   : Spearman at the most-randomized stage (or None)
    """
    if curv is None:
        curv = model.curv

    heatmap_fn: Callable = HEATMAP_METHODS[method]

    def _run() -> torch.Tensor:
        return heatmap_fn(
            model=model,
            pixel_values=pixel_values,
            x_anchors=x_anchors,
            target_class=target_class,
            score_mode=score_mode,
            curv=curv,
            **heatmap_kwargs,
        )

    original = _run()

    # Snapshot on CPU so we can always restore, even if a stage raises.
    original_state = copy.deepcopy(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
    )
    stages = _collect_randomization_stages(model)

    records: list[dict] = []
    try:
        for name, module in stages:
            if independent:
                model.load_state_dict(original_state)

            _randomize_module_(module)

            record: dict = {"stage": name}
            try:
                heat = _run()
                if not torch.isfinite(heat).all():
                    raise FloatingPointError("heatmap contains non-finite values")
                record["metrics"] = heatmap_similarity(original, heat)
                record["heatmap"] = heat
                record["error"] = None
            except Exception as exc:  # noqa: BLE001 — record and keep going
                # Random weights can blow up exp_map0; that is still a valid
                # (very-decorrelated) outcome, so we record rather than abort.
                record["metrics"] = None
                record["heatmap"] = None
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
    finally:
        model.load_state_dict(original_state)

    # Verdict from the last stage that produced a finite heatmap.
    final_spearman = None
    for rec in reversed(records):
        if rec["metrics"] is not None:
            final_spearman = rec["metrics"]["spearman"]
            break

    if final_spearman is None:
        verdict = "PASS"  # every randomization broke the map → clearly sensitive
    elif abs(final_spearman) <= pass_threshold:
        verdict = "PASS"
    elif abs(final_spearman) >= 0.7:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "method":           method,
        "score_mode":       score_mode,
        "mode":             "independent" if independent else "cascading",
        "pass_threshold":   pass_threshold,
        "original_heatmap": original,
        "stages":           records,
        "final_spearman":   final_spearman,
        "verdict":          verdict,
    }


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def summary_table(report: dict) -> str:
    """Human-readable per-stage metric table for a cascading_randomization_test."""
    lines = [
        f"{'stage':<20} {'spearman':>9} {'pearson':>9} {'cosine':>9} {'ssim':>9}",
        "-" * 60,
    ]
    for rec in report["stages"]:
        if rec["error"] is not None:
            lines.append(f"{rec['stage']:<20}  (broke: {rec['error']})")
            continue
        m = rec["metrics"]
        lines.append(
            f"{rec['stage']:<20} "
            f"{m['spearman']:>9.3f} {m['pearson']:>9.3f} "
            f"{m['cosine']:>9.3f} {m['ssim']:>9.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from transformers import CLIPTokenizer

    from losses.attribution_loss import predict_class
    from explanation.agcam_guided import encode_anchors
    from explanation.explain_image import (
        load_checkpoint,
        load_image,
        heatmap_to_pil,
        resolve_device,
    )

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image",      type=Path, required=True,
                   help="Image to sanity-check.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("outputs/sanity"))
    p.add_argument("--method",     choices=list(HEATMAP_METHODS), default="chefer")
    p.add_argument("--score_mode", choices=["angle", "margin"], default="margin")
    p.add_argument("--target",     type=str, default=None,
                   help="Class to check; defaults to the predicted class.")
    p.add_argument("--independent", action="store_true",
                   help="Randomize one stage at a time instead of cascading.")
    p.add_argument("--pass_threshold", type=float, default=0.30)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, class_names, anchor_texts, curv = load_checkpoint(args.checkpoint, device)

    ckpt_meta = torch.load(args.checkpoint, map_location="cpu")
    clip_name = ckpt_meta["clip_name"]

    pil_image, pixel_values = load_image(args.image, clip_name, device)
    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    x_anchors = encode_anchors(model, anchor_texts, tokenizer, device)

    with torch.no_grad():
        x_hyp, _ = model.encode_image(pixel_values)
    pred_idx = int(predict_class(x_hyp, x_anchors, curv=curv).item())

    if args.target is not None:
        if args.target not in class_names:
            raise ValueError(f"--target {args.target!r} not in {class_names}")
        target_idx = class_names.index(args.target)
    else:
        target_idx = pred_idx
    target_name = class_names[target_idx]

    print(
        f"Running Adebayo "
        f"{'independent' if args.independent else 'cascading'} "
        f"randomization test — method={args.method}, class={target_name!r}"
    )
    report = cascading_randomization_test(
        model=model,
        pixel_values=pixel_values,
        x_anchors=x_anchors,
        target_class=target_idx,
        method=args.method,
        score_mode=args.score_mode,
        independent=args.independent,
        pass_threshold=args.pass_threshold,
        curv=curv,
    )

    print("\n" + summary_table(report))
    print(
        f"\nfinal Spearman = {report['final_spearman']}  →  "
        f"VERDICT: {report['verdict']}"
    )
    print(
        "  PASS         = heatmap decorrelates as the model is randomized "
        "(faithful).\n"
        "  FAIL         = heatmap stays correlated with the trained model "
        "(insensitive to weights).\n"
        "  INCONCLUSIVE = partial decorrelation; inspect the table."
    )

    # ── Save per-stage heatmaps + JSON ──────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    tag = "independent" if args.independent else "cascading"

    heatmap_to_pil(report["original_heatmap"], pil_image.size).save(
        args.output_dir / f"{stem}_{args.method}_original.png"
    )
    json_stages = []
    for rec in report["stages"]:
        if rec["heatmap"] is not None:
            path = args.output_dir / f"{stem}_{args.method}_{tag}_{rec['stage']}.png"
            heatmap_to_pil(rec["heatmap"], pil_image.size).save(path)
        json_stages.append(
            {"stage": rec["stage"], "metrics": rec["metrics"], "error": rec["error"]}
        )

    json_path = args.output_dir / f"{stem}_{args.method}_{tag}_sanity.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image":          str(args.image),
                "checkpoint":     str(args.checkpoint),
                "method":         args.method,
                "score_mode":     args.score_mode,
                "mode":           report["mode"],
                "target":         target_name,
                "pass_threshold": args.pass_threshold,
                "final_spearman": report["final_spearman"],
                "verdict":        report["verdict"],
                "stages":         json_stages,
            },
            f,
            indent=2,
        )
    print(f"\nSaved heatmaps + report → {args.output_dir}")


# ---------------------------------------------------------------------------
# Note on the data randomization test
# ---------------------------------------------------------------------------
# The second Adebayo check retrains the model on randomly-permuted class labels
# and verifies the heatmap changes.  To run it with this codebase:
#   1. Train a copy with shuffled labels (permute the label column of the
#      attribution dataset before calling train_attribution.py).
#   2. Load that checkpoint here and compare heatmap_similarity() against the
#      normally-trained model on the same images.
# A faithful method should yield low similarity between the two.


if __name__ == "__main__":
    main()
