"""
Visualise hyperbolic image embeddings from an AttributionCLIP checkpoint.

Produces three plots:

  1. UMAP (Euclidean projection of the 128-D Lorentz space components, ignores
     hyperbolic geometry but is the standard way to see clusters).

  2. Poincaré disk (Lorentz → Poincaré ball, PCA-projected to 2D). Preserves
     the hyperbolic radial structure: anchors with bigger norm sit closer to
     the boundary circle. Mimics the HoroPCA-style figures common in the
     hyperbolic-embedding literature.

  3. Per-class violin / box plot of `ξ to own anchor` vs `ξ to other anchors`.
     Shows the angular separation that makes cone classification work.

Usage:
    python -m tests.visualize_embeddings \\
        --checkpoint   $WORK/checkpoints/attribution_k4_vitl14.pt \\
        --dataset_path $WORK/iab_dataset \\
        --captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --generators   real FLUX SD3 gemini \\
        --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \\
        --split        val \\
        --max_per_class 500 \\
        --output_dir   $WORK/viz/k4_hier
"""
import argparse
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
from geometry.lorentz import half_aperture, oxy_angle


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
    p.add_argument("--split",         choices=["train", "val", "test", "all"], default="val")
    p.add_argument("--val_frac",      type=float, default=0.1)
    p.add_argument("--test_frac",     type=float, default=0.1)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max_per_class", type=int,   default=500,
                   help="Cap images per (generator, semantic) for faster UMAP.")
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--show_captions", action="store_true",
                   help="Also embed and plot caption embeddings (slower).")
    return p.parse_args()


# ── Geometry helpers ─────────────────────────────────────────────────────────

def lorentz_to_poincare(x_space: np.ndarray, curv: float = 1.0) -> np.ndarray:
    """Map (B, D) Lorentz space-components onto the Poincaré ball of the same dim.
    All output points have norm < 1/sqrt(curv).
    """
    x_time = np.sqrt(1.0 / curv + (x_space ** 2).sum(axis=-1, keepdims=True))
    return x_space / (1.0 + np.sqrt(curv) * x_time)


# ── Embedding extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device, with_captions=False):
    all_img, all_cap, all_gt, all_sem = [], [], [], []
    for batch in tqdm(loader, desc="embedding"):
        pixel = batch["pixel_values"].to(device)
        x_img, _ = model.encode_image(pixel)
        all_img.append(x_img.cpu())
        if with_captions:
            cap_ids = batch["input_ids"].to(device)
            cap_mask = batch["attention_mask"].to(device)
            x_cap, _ = model.encode_text(cap_ids, cap_mask)
            all_cap.append(x_cap.cpu())
        all_gt.extend(batch["generator"])
        all_sem.extend(batch["semantic"])
    x_imgs = torch.cat(all_img, dim=0).numpy()
    x_caps = torch.cat(all_cap, dim=0).numpy() if with_captions else None
    return x_imgs, x_caps, all_gt, all_sem


# ── Plotting ─────────────────────────────────────────────────────────────────

def _class_colors(classes):
    cmap = plt.colormaps.get_cmap("tab10")
    return {c: cmap(i % 10) for i, c in enumerate(classes)}


def plot_umap(x_imgs, x_ancs, x_caps, gt, classes, out_path, with_captions=False):
    # Fit on images only — anchors live at a very different norm scale than
    # images and would distort the layout if included in the fit. We project
    # them afterwards using the trained UMAP model so they appear in the right
    # neighbourhood of their cluster.
    # n_neighbors=50 (more global structure, less "spaghetti"),
    # min_dist=0.4 (cluster blobs, not threads).
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=50, min_dist=0.4, spread=1.5,
                            n_components=2, random_state=42)
        imgs_2d = reducer.fit_transform(x_imgs)
        ancs_2d = reducer.transform(x_ancs)
        caps_2d = reducer.transform(x_caps) if with_captions else None
    except ImportError:
        print("umap-learn not installed; falling back to sklearn TSNE on union.")
        from sklearn.manifold import TSNE
        all_pts = np.concatenate(
            [x_imgs, x_ancs] + ([x_caps] if with_captions else []), axis=0
        )
        all_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_pts)
        n_img, K = len(x_imgs), len(x_ancs)
        imgs_2d = all_2d[:n_img]
        ancs_2d = all_2d[n_img:n_img + K]
        caps_2d = all_2d[n_img + K:] if with_captions else None

    n_img, K = len(x_imgs), len(x_ancs)

    fig, ax = plt.subplots(figsize=(11, 10))
    colors = _class_colors(classes)
    for c in classes:
        m = np.array([g == c for g in gt])
        if m.any():
            ax.scatter(imgs_2d[m, 0], imgs_2d[m, 1], c=[colors[c]], s=6,
                       alpha=0.45, label=f"{c} ({m.sum()})")
    if with_captions and caps_2d is not None:
        for c in classes:
            m = np.array([g == c for g in gt])
            if m.any():
                ax.scatter(caps_2d[m, 0], caps_2d[m, 1], c=[colors[c]], s=20,
                           marker="^", alpha=0.5, edgecolors="black", linewidths=0.3)
    for i, c in enumerate(classes):
        ax.scatter(ancs_2d[i, 0], ancs_2d[i, 1], c=[colors[c]], s=600,
                   marker="*", edgecolors="black", linewidths=1.8, zorder=10,
                   label=f"{c} anchor")
        ax.annotate(c, (ancs_2d[i, 0], ancs_2d[i, 1]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=11, fontweight="bold", zorder=11)
    ax.set_title(f"UMAP of hyperbolic embeddings ({n_img} images, {K} anchors"
                 + (f", +{len(caps_2d)} captions" if with_captions else "") + ")")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def plot_poincare_disk(x_imgs, x_ancs, x_caps, gt, classes, out_path,
                       curv=1.0, with_captions=False):
    """Lorentz → Poincaré ball, then PCA to 2D, plotted inside the unit circle."""
    from sklearn.decomposition import PCA

    p_imgs = lorentz_to_poincare(x_imgs, curv=curv)
    p_ancs = lorentz_to_poincare(x_ancs, curv=curv)
    p_caps = lorentz_to_poincare(x_caps, curv=curv) if with_captions else None

    # Fit PCA on the union so anchors and images share axes.
    all_pts = np.concatenate(
        [p_imgs, p_ancs] + ([p_caps] if with_captions else []), axis=0
    )
    pca = PCA(n_components=2, random_state=42).fit(all_pts)

    imgs_2d = pca.transform(p_imgs)
    ancs_2d = pca.transform(p_ancs)
    caps_2d = pca.transform(p_caps) if with_captions else None

    # Rescale so points fit comfortably in the unit disk.
    max_r = np.max(np.linalg.norm(np.concatenate(
        [imgs_2d, ancs_2d] + ([caps_2d] if with_captions else []), axis=0
    ), axis=1))
    scale = 0.98 / max_r if max_r > 0 else 1.0
    imgs_2d, ancs_2d = imgs_2d * scale, ancs_2d * scale
    if caps_2d is not None:
        caps_2d = caps_2d * scale

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.2))

    colors = _class_colors(classes)
    for c in classes:
        m = np.array([g == c for g in gt])
        if m.any():
            ax.scatter(imgs_2d[m, 0], imgs_2d[m, 1], c=[colors[c]], s=6,
                       alpha=0.4, label=f"{c} ({m.sum()})")
    if with_captions and caps_2d is not None:
        for c in classes:
            m = np.array([g == c for g in gt])
            if m.any():
                ax.scatter(caps_2d[m, 0], caps_2d[m, 1], c=[colors[c]], s=20,
                           marker="^", alpha=0.45, edgecolors="black", linewidths=0.3)
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
    ax.set_title("Poincaré disk projection (Lorentz → Poincaré ball → 2-D PCA)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def plot_xi_distribution(xi_to_anchors, gt, classes, out_path):
    """Box plot of ξ(image, anchor) for own anchor vs other anchors per class."""
    K = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    gt_idx = np.array([cls_to_idx[g] for g in gt])

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * K), 5))
    positions, data, colors_list = [], [], []
    cmap = _class_colors(classes)
    for i, c in enumerate(classes):
        m = gt_idx == i
        if not m.any():
            continue
        own = xi_to_anchors[m, i]
        other = xi_to_anchors[m][:, [j for j in range(K) if j != i]].flatten()
        positions += [3 * i + 0.5, 3 * i + 1.5]
        data += [own, other]
        colors_list += [cmap[c], (0.7, 0.7, 0.7, 0.8)]
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                    showfliers=False)
    for patch, col in zip(bp["boxes"], colors_list):
        patch.set_facecolor(col)
    ax.set_xticks([3 * i + 1 for i in range(K)])
    ax.set_xticklabels(classes)
    ax.set_ylabel("exterior angle ξ (rad)")
    ax.set_title("ξ to own anchor (color) vs ξ to other anchors (grey)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def _pairwise_xi(apex, point, curv):
    """Pairwise oxy_angle: (A, D), (P, D) → (A, P)."""
    A, D = apex.shape; P = point.shape[0]
    apex_t  = apex.unsqueeze(1).expand(A, P, D).reshape(A * P, D)
    point_t = point.unsqueeze(0).expand(A, P, D).reshape(A * P, D)
    return oxy_angle(apex_t, point_t, curv=curv).reshape(A, P)


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
        include_uncaptioned=not args.show_captions,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Anchor embeddings
    tok = tokenizer(anchor_texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=77)
    with torch.no_grad():
        x_ancs_t, _ = model.encode_text(tok["input_ids"].to(device),
                                        tok["attention_mask"].to(device))
    x_ancs = x_ancs_t.cpu().numpy()

    x_imgs, x_caps, gt, sem = extract_embeddings(
        model, loader, device, with_captions=args.show_captions
    )
    print(f"Embedded {len(x_imgs)} images "
          + (f"+ {len(x_caps)} captions" if x_caps is not None else "")
          + f"; anchors: {x_ancs.shape}")

    # Pairwise ξ for the distribution plot
    with torch.no_grad():
        xi = _pairwise_xi(x_ancs_t.float(),
                          torch.from_numpy(x_imgs).to(device).float(),
                          curv=curv).T.cpu().numpy()  # (N, K)
    psi = half_aperture(x_ancs_t.float(), curv=curv, min_radius=min_radius).cpu().numpy()
    print(f"  ψ per cone: {dict(zip(class_names, [f'{p:.3f}' for p in psi]))}")

    # Plots
    plot_umap(x_imgs, x_ancs, x_caps, gt, class_names,
              out_dir / "umap.png", with_captions=args.show_captions)
    plot_poincare_disk(x_imgs, x_ancs, x_caps, gt, class_names,
                       out_dir / "poincare_disk.png", curv=curv,
                       with_captions=args.show_captions)
    plot_xi_distribution(xi, gt, class_names, out_dir / "xi_distribution.png")

    print(f"\nAll plots saved in {out_dir}/")


if __name__ == "__main__":
    main()
