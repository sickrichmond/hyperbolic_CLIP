"""
Plot the distribution of hyperbolic distance-from-root, per class.

This is the AttributionCLIP analogue of Figure 4 in HySAC: for every image we
measure its geodesic distance from the origin (root) of the Lorentz model and
plot the per-class distributions.

Interpretation caveat
----------------------
In HySAC the radial coordinate (distance from root) encodes the safe/unsafe
hierarchy, so the distributions separate cleanly. In this attribution model the
radius mainly separates **real vs fake**; individual generators are separated
*angularly* (entailment cones / oxy_angle), NOT radially. So expect `real` at
one radius and the fake generators largely overlapping at another — that is the
honest, expected result, not a bug. The cleaner HySAC-style read is the
real-vs-fake panel.

Two figures are written:
  - <output>_per_class.png : one KDE per generator (what you asked for).
  - <output>_real_vs_fake.png : two distributions (real vs all-fake), the
    direct HySAC Fig.4 analogue.

Usage
-----
    python -m tests.plot_distance_from_root \\
        --checkpoint    $WORK/checkpoints/attribution_all_no_dalle_d16.pt \\
        --dataset_path  $WORK/iab_dataset \\
        --semantics     COCO \\
        --max_per_class 300 \\
        --output        $WORK/outputs/dist_from_root/d16
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

import matplotlib
matplotlib.use("Agg")  # headless: compute nodes have no display
import matplotlib.pyplot as plt

from models.attribution_clip import AttributionCLIP
from data.iab_dataset import IABDataset
from geometry.lorentz import elementwise_dist


# 22-class default (real + 21 generators), matching slurm_cineca_all.sh.
DEFAULT_GENERATORS = [
    "real", "4o", "gemini", "grok3", "FLUX",
    "SD1_5", "SD2_1", "SD3", "SD3_5", "SDXL",
    "PIXART", "PLAYGROUND_2_5", "KANDINSKY", "CogView3_PLUS",
    "hidream", "hunyuan", "ideogram", "infinity", "janus-pro", "kling",
    "mid-5.2", "mid-6.0",
]
DEFAULT_SEMANTICS = ["COCO"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset_path",  required=True)
    p.add_argument("--generators",    nargs="+", default=DEFAULT_GENERATORS)
    p.add_argument("--semantics",     nargs="+", default=DEFAULT_SEMANTICS)
    p.add_argument("--max_per_class", type=int, default=300,
                   help="Cap images per (generator, semantic) pair for speed.")
    p.add_argument("--batch_size",    type=int, default=128)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--output",        required=True,
                   help="Output path prefix (two PNGs are written).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    """Gaussian KDE on `grid`. Returns None if it cannot be estimated."""
    if values.size < 2 or np.allclose(values, values[0]):
        return None
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(values)(grid)
    except Exception:
        # Fallback: histogram-as-density, linearly interpolated onto the grid.
        hist, edges = np.histogram(values, bins=40, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(grid, centers, hist, left=0.0, right=0.0)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    clip_name = ckpt["clip_name"]
    curv = float(ckpt.get("curv", 1.0))
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  curv={curv}  hyperbolic_dim={ckpt.get('hyperbolic_dim')}")

    model = AttributionCLIP(
        clip_name=clip_name,
        lora_r=ckpt.get("lora_r", 8),
        lora_alpha=ckpt.get("lora_alpha", 16),
        hyperbolic_dim=ckpt.get("hyperbolic_dim", 128),
        curv=curv,
    ).to(device)
    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection"])
    model.eval()

    dataset = IABDataset(
        root=args.dataset_path,
        generators=args.generators,
        semantics=args.semantics,
        processor_name=clip_name,
        max_per_class=args.max_per_class,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ── Compute distance-from-root per image ─────────────────────────────────
    dist_by_gen: dict[str, list[float]] = {g: [] for g in args.generators}
    with torch.no_grad():
        for batch in tqdm(loader, desc="distance-from-root"):
            pixel = batch["pixel_values"].to(device)
            x_hyp, _ = model.encode_image(pixel)              # (B, D) Lorentz space
            origin = torch.zeros_like(x_hyp)
            d = elementwise_dist(x_hyp, origin, curv=curv)    # (B,)
            for gen, dv in zip(batch["generator"], d.cpu().tolist()):
                dist_by_gen[gen].append(dv)

    dist_by_gen = {g: np.asarray(v) for g, v in dist_by_gen.items() if len(v) > 0}
    if not dist_by_gen:
        raise RuntimeError("No embeddings computed — check dataset paths/semantics.")

    all_d = np.concatenate(list(dist_by_gen.values()))
    lo, hi = float(all_d.min()), float(all_d.max())
    pad = 0.05 * (hi - lo + 1e-6)
    grid = np.linspace(lo - pad, hi + pad, 400)

    # Print per-class summary (mean ± std distance from root).
    print("\nDistance-from-root (mean ± std):")
    for g in args.generators:
        if g in dist_by_gen:
            v = dist_by_gen[g]
            print(f"  {g:>14}: {v.mean():.4f} ± {v.std():.4f}  (n={v.size})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    title_suffix = f"{Path(args.checkpoint).stem} · curv={curv} · {','.join(args.semantics)}"

    # ── Figure 1: per-class ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")
    for i, g in enumerate(args.generators):
        if g not in dist_by_gen:
            continue
        y = kde_curve(dist_by_gen[g], grid)
        color = "black" if g == "real" else cmap(i % 20)
        lw = 2.4 if g == "real" else 1.3
        if y is not None:
            ax.plot(grid, y, label=g, color=color, linewidth=lw)
        else:
            ax.axvline(float(dist_by_gen[g].mean()), color=color, linewidth=lw, label=g)
    ax.set_xlabel("Hyperbolic distance from root")
    ax.set_ylabel("Density")
    ax.set_title(f"Distance from root — per class\n{title_suffix}", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    p1 = out.with_name(out.name + "_per_class.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPer-class figure → {p1}")

    # ── Figure 2: real vs fake (HySAC Fig.4 analogue) ────────────────────────
    real_d = dist_by_gen.get("real", np.array([]))
    fake_d = np.concatenate([v for g, v in dist_by_gen.items() if g != "real"]) \
        if any(g != "real" for g in dist_by_gen) else np.array([])

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, vals, color in [
        ("real", real_d, "#1f77b4"),
        ("fake (all generators)", fake_d, "#d62728"),
    ]:
        if vals.size == 0:
            continue
        y = kde_curve(vals, grid)
        if y is not None:
            ax.plot(grid, y, color=color, linewidth=2.2, label=label)
            ax.fill_between(grid, y, color=color, alpha=0.25)
        ax.axvline(float(vals.mean()), color=color, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Hyperbolic distance from root")
    ax.set_ylabel("Density")
    ax.set_title(f"Distance from root — real vs fake\n{title_suffix}", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    p2 = out.with_name(out.name + "_real_vs_fake.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Real-vs-fake figure → {p2}")


if __name__ == "__main__":
    main()
