"""
Explain a single image with AGCAM or Guided attribution.

Loads an AttributionCLIP checkpoint, encodes class anchors, runs the chosen
explanation method for one or all classes, and saves heatmap PNGs and
colour-overlay PNGs alongside the output JSON.

Usage examples
--------------
# Explain predicted class with AGCAM (margin score, all outputs)
python explanation/explain_image.py \\
    --image       data/images/example.jpg \\
    --checkpoint  checkpoints/attribution_FLUX_vitl14.pt \\
    --method      agcam \\
    --score_mode  margin \\
    --output_dir  outputs/example \\
    --all_classes

# Explain a specific class with Guided
python explanation/explain_image.py \\
    --image       data/images/example.jpg \\
    --checkpoint  checkpoints/attribution_FLUX_vitl14.pt \\
    --method      guided \\
    --target      FLUX \\
    --output_dir  outputs/example_guided
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPTokenizer

from models.attribution_clip import AttributionCLIP
from losses.attribution_loss import predict_class
from explanation.agcam_guided import (
    encode_anchors,
    explain_all_classes,
    HEATMAP_METHODS,
)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """
    Load an AttributionCLIP checkpoint saved by train_attribution.py.

    Returns (model, class_names, anchor_texts, curv).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = AttributionCLIP(
        clip_name=ckpt["clip_name"],
        lora_r=ckpt["lora_r"],
        lora_alpha=ckpt["lora_alpha"],
        hyperbolic_dim=ckpt["hyperbolic_dim"],
        curv=ckpt["curv"],
        # Eager attention so vision_model can return attention maps for AGCAM/Guided.
        attn_implementation="eager",
    ).to(device)

    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection"])
    model.eval()

    return model, ckpt["class_names"], ckpt["anchor_texts"], ckpt["curv"]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def load_image(image_path: Path, clip_name: str, device: torch.device):
    """
    Load and preprocess an image with the CLIP processor.

    Returns (pil_image, pixel_values) where pixel_values is (1, C, H, W) fp32.
    """
    processor = CLIPProcessor.from_pretrained(clip_name)
    pil_image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=pil_image, return_tensors="pt")["pixel_values"]
    return pil_image, pixel_values.to(device)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def heatmap_to_pil(
    heatmap: torch.Tensor,
    size: tuple[int, int],
    cmap_name: str = "inferno",
) -> Image.Image:
    """
    Convert a (side, side) float [0,1] tensor to a resized RGB heatmap image
    using a perceptually-uniform sequential colormap.

    Default is matplotlib's ``inferno``: dark (low saliency) → bright yellow
    (high saliency), so low values read as "cold / nothing" instead of the
    counter-intuitive full red of a naive red-yellow-white ramp.  Perceptually
    uniform and colourblind-safe by construction.
    """
    import numpy as np
    from matplotlib import colormaps

    h = np.clip(heatmap.numpy(), 0.0, 1.0)
    rgb = (colormaps[cmap_name](h)[..., :3] * 255).astype("uint8")

    pil = Image.fromarray(rgb, mode="RGB")
    return pil.resize(size, resample=Image.BILINEAR)


def overlay_heatmap(
    pil_image: Image.Image,
    heatmap: torch.Tensor,
    alpha: float = 0.60,
    cmap_name: str = "inferno",
) -> Image.Image:
    """
    Overlay the heatmap on the image with saliency-modulated opacity.

    Rather than tinting the whole frame uniformly, each pixel's opacity scales
    with its heatmap value: low-saliency regions stay transparent (the original
    image shows through) and only high-saliency regions are coloured.  This
    keeps the underlying image readable and puts the emphasis exactly where the
    model attributes signal.

    Args:
        pil_image: Original RGB image.
        heatmap:   (side, side) float [0, 1] tensor.
        alpha:     Opacity cap for the most-salient pixels (0 = invisible).
        cmap_name: Colormap name (see heatmap_to_pil).

    Returns:
        Composite RGB image, same size as pil_image.
    """
    import numpy as np
    from matplotlib import colormaps

    h = np.clip(heatmap.numpy(), 0.0, 1.0)
    color = (colormaps[cmap_name](h)[..., :3] * 255).astype("uint8")
    color_pil = Image.fromarray(color, mode="RGB").resize(
        pil_image.size, resample=Image.BILINEAR
    )
    # Per-pixel alpha from the heatmap: low saliency → transparent.
    a_pil = Image.fromarray((h * alpha * 255).astype("uint8"), mode="L").resize(
        pil_image.size, resample=Image.BILINEAR
    )
    return Image.composite(color_pil, pil_image.convert("RGB"), a_pil)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image",       type=Path, required=True,
                   help="Path to the image to explain.")
    p.add_argument("--checkpoint",  type=Path, required=True,
                   help="AttributionCLIP checkpoint (.pt).")
    p.add_argument("--output_dir",  type=Path, default=Path("outputs/explanation"),
                   help="Directory where outputs are written.")
    p.add_argument("--method",
                   choices=["agcam", "guided", "chefer"], default="agcam",
                   help="Explanation method. 'chefer': gradient-weighted "
                        "relevance rollout through the full attention stack "
                        "(Chefer et al. 2021), usually the most faithful.")
    p.add_argument("--score_mode",
                   choices=["angle", "margin"], default="margin",
                   help="Score used for backpropagation. "
                        "'margin' (recommended): angle difference between "
                        "predicted class and runner-up. "
                        "'angle': raw cone membership angle for target class.")
    p.add_argument("--target",      type=str, default=None,
                   help="Generator name to explain (e.g. 'FLUX'). "
                        "Defaults to the model's predicted class.")
    p.add_argument("--all_classes", action="store_true",
                   help="Produce one heatmap per class instead of just target.")
    p.add_argument("--head_fusion",
                   choices=["sum", "mean", "max"], default="sum",
                   help="How to aggregate attention heads.")
    p.add_argument("--layer_fusion",
                   choices=["sum", "mean", "max"], default="sum",
                   help="(AGCAM only) How to aggregate transformer layers.")
    p.add_argument("--no_sigmoid",  action="store_true",
                   help="(AGCAM/Guided) Disable sigmoid on attention maps.")
    p.add_argument("--start_layer", type=int, default=0,
                   help="(Chefer only) Begin the relevance rollout at this "
                        "transformer layer (0 = all layers).")
    p.add_argument("--overlay_alpha", type=float, default=0.60,
                   help="Opacity cap for the most-salient pixels in overlays.")
    p.add_argument("--cmap", type=str, default="inferno",
                   help="Matplotlib colormap for heatmaps/overlays "
                        "(sequential, e.g. inferno/magma/viridis).")
    p.add_argument("--device",
                   choices=["auto", "cpu", "cuda"], default="auto",
                   help="Compute device.")
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model, class_names, anchor_texts, curv = load_checkpoint(args.checkpoint, device)
    print(f"Classes ({len(class_names)}): {class_names}")

    # ── Load and preprocess image ──────────────────────────────────────────────
    ckpt_meta = torch.load(args.checkpoint, map_location="cpu")
    clip_name  = ckpt_meta["clip_name"]

    pil_image, pixel_values = load_image(args.image, clip_name, device)
    print(f"Image loaded: {args.image}  ({pil_image.width}×{pil_image.height})")

    # ── Encode class anchors ───────────────────────────────────────────────────
    tokenizer  = CLIPTokenizer.from_pretrained(clip_name)
    x_anchors  = encode_anchors(model, anchor_texts, tokenizer, device)

    # ── Predict class ──────────────────────────────────────────────────────────
    with torch.no_grad():
        x_hyp, _ = model.encode_image(pixel_values)
    pred_idx   = int(predict_class(x_hyp, x_anchors, curv=curv).item())
    pred_class = class_names[pred_idx]
    print(f"Predicted class: {pred_class!r}  (index {pred_idx})")

    # ── Resolve target class ───────────────────────────────────────────────────
    if args.target is not None:
        if args.target not in class_names:
            raise ValueError(
                f"--target {args.target!r} not in class_names {class_names}"
            )
        target_idx   = class_names.index(args.target)
        target_class = args.target
    else:
        target_idx   = pred_idx
        target_class = pred_class

    # ── Run explanation ────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Only forward the keyword args that the chosen method understands.
    if args.method == "agcam":
        extra = dict(
            head_fusion=args.head_fusion,
            apply_sigmoid=not args.no_sigmoid,
            layer_fusion=args.layer_fusion,
        )
    elif args.method == "guided":
        extra = dict(head_fusion=args.head_fusion, apply_sigmoid=not args.no_sigmoid)
    else:  # chefer
        extra = dict(start_layer=args.start_layer)

    if args.all_classes:
        print(f"Running {args.method.upper()} for all {len(class_names)} classes …")
        heatmaps = explain_all_classes(
            model=model,
            pixel_values=pixel_values,
            x_anchors=x_anchors,
            class_names=class_names,
            method=args.method,
            score_mode=args.score_mode,
            **extra,
        )
    else:
        print(f"Running {args.method.upper()} for class {target_class!r} …")
        fn     = HEATMAP_METHODS[args.method]
        result = fn(
            model=model,
            pixel_values=pixel_values,
            x_anchors=x_anchors,
            target_class=target_idx,
            score_mode=args.score_mode,
            curv=curv,
            **extra,
        )
        heatmaps = {target_class: result}

    # ── Save outputs ───────────────────────────────────────────────────────────
    stem = args.image.stem
    saved_files: dict[str, list[str]] = {}

    for cls_name, heatmap in heatmaps.items():
        safe_name = cls_name.replace("/", "_").replace(" ", "_")

        heatmap_path = args.output_dir / f"{stem}_{args.method}_{safe_name}_heatmap.png"
        overlay_path = args.output_dir / f"{stem}_{args.method}_{safe_name}_overlay.png"

        heat_pil = heatmap_to_pil(heatmap, pil_image.size, cmap_name=args.cmap)
        over_pil = overlay_heatmap(
            pil_image, heatmap, alpha=args.overlay_alpha, cmap_name=args.cmap
        )

        heat_pil.save(heatmap_path)
        over_pil.save(overlay_path)
        print(f"  [{cls_name}]  heatmap → {heatmap_path}")
        print(f"  [{cls_name}]  overlay → {overlay_path}")

        saved_files[cls_name] = {
            "heatmap": str(heatmap_path),
            "overlay": str(overlay_path),
        }

    # ── Save decision JSON ─────────────────────────────────────────────────────
    result_json = {
        "image":         str(args.image),
        "predicted":     pred_class,
        "pred_idx":      pred_idx,
        "target":        target_class,
        "method":        args.method,
        "score_mode":    args.score_mode,
        "class_names":   class_names,
        "all_classes":   args.all_classes,
        "checkpoint":    str(args.checkpoint),
        "outputs":       saved_files,
    }
    json_path = args.output_dir / f"{stem}_{args.method}_explanation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)
    print(f"\nDecision JSON → {json_path}")


if __name__ == "__main__":
    main()