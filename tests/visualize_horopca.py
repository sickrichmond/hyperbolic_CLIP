"""
Visualise hyperbolic embeddings from an AttributionCLIP checkpoint.

Self-contained: loads the model + dataset, embeds images, then produces three
plots:

  1. Poincaré disk (2-D) via HoroPCA — hyperbolic-native dimensionality
     reduction. Lorentz → Poincaré ball → 2-D via horospherical projections.

  2. UMAP 3-D coloured by generator class (real/FLUX/SD3/gemini …),
     anchors plotted as class-coloured stars.

  3. UMAP 3-D coloured by semantic class (COCO/FFHQ/…), anchors plotted as
     grey stars (anchors don't belong to any semantic).

The two UMAP plots share the same fitted UMAP model so they are point-by-point
comparable. UMAP is fitted on images only — anchors live at a different norm
scale and would distort the layout; they are placed at the per-class centroid
in UMAP space (semantically: "this anchor represents this cluster").

We deliberately do NOT use plain Euclidean PCA: the embeddings live in
hyperbolic space and Euclidean PCA would silently misrepresent radial
distances. If HoroPCA isn't available the script raises a hard error.

Usage:
    python -m tests.visualize_horopca \\
        --checkpoint   $WORK/checkpoints/attribution_k4_vitl14.pt \\
        --dataset_path $WORK/iab_dataset \\
        --captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --generators   real FLUX SD3 gemini \\
        --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \\
        --split        val \\
        --max_per_class 500 \\
        --output_dir   $WORK/viz/k4_hier

HoroPCA repo:
    Set HOROPCA_DIR env var, or clone to <repo>/external/HoroPCA, or
    $WORK/hyp_fine_tuning/horopca:
        git clone https://github.com/HazyResearch/HoroPCA <repo>/external/HoroPCA
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from models.attribution_clip import AttributionCLIP
from data.iab_clip_dataset import IABCLIPDataset
from geometry.lorentz import half_aperture


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset_path",  required=True)
    p.add_argument("--captions_dir",  required=True)
    p.add_argument("--generators",    nargs="+", required=True)
    p.add_argument("--semantics",     nargs="+",
                   default=["COCO", "cat", "dog", "wild", "FFHQ", "celebahq",
                             "bedroom", "church", "classroom", "ImageNet-1k"])
    p.add_argument("--split",         choices=["train", "val", "all"], default="val")
    p.add_argument("--val_frac",      type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max_per_class", type=int,   default=500,
                   help="Cap images per (generator, semantic). HoroPCA builds "
                        "internal (N, N) matrices — keep ≲ 5000 unless RAM lets you.")
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--output_dir",    required=True)
    return p.parse_args()


# ────────────────────────── geometry helpers ─────────────────────────────────

def lorentz_to_poincare(x_space: np.ndarray, curv: float) -> np.ndarray:
    """Stereographic map of Lorentz space-components onto the Poincaré ball."""
    x_time = np.sqrt(1.0 / curv + np.sum(x_space ** 2, axis=-1, keepdims=True))
    return x_space / (x_time + 1.0 / np.sqrt(curv))


def class_centroids(imgs_d: np.ndarray, gt, classes) -> np.ndarray:
    """Per-class centroid in (UMAP) space. Used to place anchor stars meaningfully."""
    cents = np.zeros((len(classes), imgs_d.shape[1]))
    for i, c in enumerate(classes):
        m = np.array([g == c for g in gt])
        if m.any():
            cents[i] = imgs_d[m].mean(axis=0)
    return cents


# ────────────────────────── HoroPCA loading ──────────────────────────────────

def _patch_torch_solve():
    """HoroPCA uses the removed torch.solve(B, A) → (solution, LU) API.
    Re-implement on top of torch.linalg.solve (the modern replacement)."""
    if not getattr(torch, "_horopca_solve_patched", False):
        def _solve(B, A):
            return torch.linalg.solve(A, B), None
        torch.solve = _solve
        torch._horopca_solve_patched = True


def _load_horopca():
    """Locate and import HoroPCA. Hard error if the repo isn't found."""
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        os.environ.get("HOROPCA_DIR"),
        str(repo_root / "external" / "HoroPCA"),
        os.path.expandvars("$WORK/hyp_fine_tuning/horopca"),
    ]
    horopca_path = next((p for p in candidates if p and Path(p).exists()), None)
    if horopca_path is None:
        raise FileNotFoundError(
            "HoroPCA repo not found. Set HOROPCA_DIR, or clone the repo to "
            "<repo>/external/HoroPCA, or $WORK/hyp_fine_tuning/horopca:\n"
            "  git clone https://github.com/HazyResearch/HoroPCA <repo>/external/HoroPCA"
        )
    if horopca_path not in sys.path:
        sys.path.insert(0, horopca_path)
    _patch_torch_solve()
    from learning.pca import HoroPCA   # type: ignore  (external repo)
    return HoroPCA


def run_horopca_2d(all_pts_poincare: np.ndarray) -> np.ndarray:
    """Fit HoroPCA → 2-D on (N, D) Poincaré-ball points. Returns (N, 2)."""
    HoroPCA = _load_horopca()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(all_pts_poincare, dtype=torch.float64, device=device)
    pca = HoroPCA(dim=all_pts_poincare.shape[1], n_components=2).double().to(device)
    pca.fit(X, iterative=False, optim=True)
    with torch.no_grad():
        return pca.map_to_ball(X).cpu().numpy()


# ────────────────────────── embedding extraction ─────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device):
    all_img, all_gt, all_sem = [], [], []
    for batch in tqdm(loader, desc="embedding"):
        pixel = batch["pixel_values"].to(device)
        x_img, _ = model.encode_image(pixel)
        all_img.append(x_img.cpu())
        all_gt.extend(batch["generator"])
        all_sem.extend(batch["semantic"])
    return torch.cat(all_img, dim=0).numpy(), all_gt, all_sem


# ────────────────────────── plotting ─────────────────────────────────────────

def _class_colors(classes):
    cmap = plt.colormaps.get_cmap("tab10" if len(classes) <= 10 else "tab20")
    return {c: cmap(i % cmap.N) for i, c in enumerate(classes)}


def plot_poincare_disk(imgs_2d, ancs_2d, gt, classes, out_path):
    """2-D HoroPCA scatter inside the unit disk."""
    _, ax = plt.subplots(figsize=(11, 11))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.2))

    colors = _class_colors(classes)
    for c in classes:
        m = np.array([g == c for g in gt])
        if m.any():
            ax.scatter(imgs_2d[m, 0], imgs_2d[m, 1], c=[colors[c]], s=6,
                       alpha=0.4, label=f"{c} ({m.sum()})")
    for i, c in enumerate(classes):
        ax.scatter(ancs_2d[i, 0], ancs_2d[i, 1], c=[colors[c]], s=700,
                   marker="*", edgecolors="black", linewidths=1.8, zorder=10,
                   label=f"{c} anchor")
        ax.annotate(c, (ancs_2d[i, 0], ancs_2d[i, 1]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=11, fontweight="bold")

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Poincaré disk projection (Lorentz → Poincaré ball → 2-D HoroPCA)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def compute_umap_3d(x_imgs: np.ndarray) -> np.ndarray:
    """3-D UMAP fitted on images only. HySAC-paper style (compact blobs)."""
    import umap
    reducer = umap.UMAP(
        n_neighbors=80, min_dist=0.7, spread=2.0,
        n_components=3, metric="euclidean", random_state=42,
    )
    return reducer.fit_transform(x_imgs)


def _plot_umap_3d(imgs_d, ancs_d, point_labels, point_classes,
                  anchor_names, anchor_color_by_class, title, out_path):
    point_colors = _class_colors(point_classes)
    anchor_colors = (_class_colors(anchor_names) if anchor_color_by_class
                     else {n: (0.35, 0.35, 0.35, 1.0) for n in anchor_names})

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    for c in point_classes:
        m = np.array([g == c for g in point_labels])
        if m.any():
            ax.scatter(imgs_d[m, 0], imgs_d[m, 1], imgs_d[m, 2],
                       c=[point_colors[c]], s=6, alpha=0.5,
                       label=f"{c} ({m.sum()})")
    for i, name in enumerate(anchor_names):
        ax.scatter(ancs_d[i, 0], ancs_d[i, 1], ancs_d[i, 2],
                   c=[anchor_colors[name]], s=600, marker="*",
                   edgecolors="black", linewidths=1.8,
                   label=f"anchor: {name}", depthshade=False)
        ax.text(ancs_d[i, 0], ancs_d[i, 1], ancs_d[i, 2], f"  {name}",
                fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def plot_umap_by_class(imgs_d, ancs_d, gt, classes, out_path):
    title = f"UMAP of hyperbolic embeddings — coloured by generator ({len(imgs_d)} images)"
    _plot_umap_3d(imgs_d, ancs_d, gt, classes, classes,
                  anchor_color_by_class=True, title=title, out_path=out_path)


def plot_umap_by_semantic(imgs_d, ancs_d, sem, classes_sem, anchor_names, out_path):
    title = f"UMAP of hyperbolic embeddings — coloured by semantic class ({len(imgs_d)} images)"
    _plot_umap_3d(imgs_d, ancs_d, sem, classes_sem, anchor_names,
                  anchor_color_by_class=False, title=title, out_path=out_path)


# ────────────────────────── main ─────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    clip_name = ckpt["clip_name"]
    class_names = ckpt["class_names"]
    anchor_texts = ckpt["anchor_texts"]
    curv = ckpt.get("curv", 1.0)
    min_radius = ckpt.get("min_radius", 0.1)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  classes: {class_names}")
    print(f"  curv={curv}  min_radius={min_radius}")

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

    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    dataset = IABCLIPDataset(
        root=args.dataset_path,
        captions_dir=args.captions_dir,
        generators=args.generators,
        semantics=args.semantics,
        processor_name=clip_name,
        max_per_class=args.max_per_class,
        split=args.split,
        val_frac=args.val_frac,
        seed=args.seed,
        include_uncaptioned=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ── Anchor embeddings ────────────────────────────────────────────────────
    tok = tokenizer(anchor_texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=77)
    with torch.no_grad():
        x_ancs_t, _ = model.encode_text(tok["input_ids"].to(device),
                                        tok["attention_mask"].to(device))
    x_ancs = x_ancs_t.cpu().numpy()

    # ── Image embeddings ─────────────────────────────────────────────────────
    x_imgs, gt, sem = extract_embeddings(model, loader, device)
    print(f"Embedded {len(x_imgs)} images, {len(x_ancs)} anchors "
          f"(hyperbolic_dim={x_imgs.shape[1]})")

    # ── Poincaré disk (2-D HoroPCA) ──────────────────────────────────────────
    p_imgs = lorentz_to_poincare(x_imgs, curv=curv)
    p_ancs = lorentz_to_poincare(x_ancs, curv=curv)
    all_pts = np.concatenate([p_imgs, p_ancs], axis=0)
    print(f"Running HoroPCA → 2-D on {len(all_pts)} points "
          f"(this can take a few minutes)…")
    coords_2d = run_horopca_2d(all_pts)
    imgs_2d = coords_2d[:len(p_imgs)]
    ancs_2d = coords_2d[len(p_imgs):]
    plot_poincare_disk(imgs_2d, ancs_2d, gt, class_names,
                       out_dir / "poincare_disk.png")

    # ── 3-D UMAP (single fit, two colourings) ────────────────────────────────
    print("Computing 3-D UMAP (fit on images, anchors at class centroids)…")
    imgs_d = compute_umap_3d(x_imgs)
    ancs_d = class_centroids(imgs_d, gt, class_names)
    plot_umap_by_class(imgs_d, ancs_d, gt, class_names,
                       out_dir / "umap_by_class.png")
    plot_umap_by_semantic(imgs_d, ancs_d, sem, args.semantics, class_names,
                          out_dir / "umap_by_semantic.png")

    # ── ψ summary printed for reference ──────────────────────────────────────
    psi = half_aperture(x_ancs_t.float(), curv=curv,
                        min_radius=min_radius).cpu().numpy()
    print(f"\nψ (half-aperture) per cone: "
          f"{dict(zip(class_names, [f'{p:.3f}' for p in psi]))}")
    print(f"All plots saved in {out_dir}/")


if __name__ == "__main__":
    main()
