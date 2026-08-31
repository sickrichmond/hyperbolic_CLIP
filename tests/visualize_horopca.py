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
        --checkpoint   $WORK/hyp_fine_tuning/checkpoints/attribution_k4_vitl14.pt \\
        --dataset_path $WORK/hyp_fine_tuning/iab_dataset \\
        --captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --generators   real FLUX SD3 gemini \\
        --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \\
        --split        val \\
        --max_per_class 500 \\
        --output_dir   $WORK/hyp_fine_tuning/viz/k4_hier

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
from geometry.lorentz import exp_map0, half_aperture


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
                   help="Cap images per (generator, semantic) loaded from disk.")
    p.add_argument("--horopca_max_points", type=int, default=1500,
                   help="Random subsample size used to FIT HoroPCA. The default "
                        "uses pairwise distances (memory O(N² · D) in fp64 ≈ "
                        "2 GB at N=1500, D=128), which is more stable for a "
                        "direct 128→2 fit than the Fréchet-variance variant. "
                        "Anchors are always kept.")
    p.add_argument("--horopca_steps", type=int, default=500,
                   help="Gradient steps for the HoroPCA fit (paper used 2000).")
    p.add_argument("--horopca_lr", type=float, default=5e-2,
                   help="Learning rate for HoroPCA Adam (paper default).")
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--no_center", action="store_true",
                   help="Disable Fréchet-mean centering before HoroPCA. "
                        "Centering is ON by default and is what stops the disk "
                        "from collapsing to a single off-centre blob (HoroPCA "
                        "assumes the data has Fréchet mean at the origin — see "
                        "the paper's Appendix C.1). Use this flag only to "
                        "reproduce the un-centered (broken) behaviour.")
    p.add_argument("--disk_zoom", type=float, default=0.0,
                   help="Axis half-width for the Poincaré disk plot. 0 (default) "
                        "auto-frames to the data extent so the cloud fills the "
                        "figure (Fig-4 style). Set e.g. 1.08 to always show the "
                        "full unit disk with its boundary circle.")
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


def run_horopca_2d(fit_pts: np.ndarray, project_pts: np.ndarray,
                   lr: float = 5e-2, max_steps: int = 500, pca=None):
    """Fit HoroPCA on `fit_pts`, project ALL of `project_pts`. Returns ((N, 2), pca).

    Pass a previously returned `pca` to SKIP the fit and only project. Two reasons:
    a fresh fit each time gives a different basis (rotations and reflections), so a
    sequence of frames is not comparable — and the fit is the expensive half.

    Uses the default pairwise-variance objective (more stable for direct
    high-dim → 2-D than the Fréchet variant, which collapses to a single
    point when initialised with zero mean weights). Memory is O(N² · D) in
    fp64 — at the default fit subsample N=1500, D=128 this is ~2 GB. The
    paper's reference hyperparameters are lr=5e-2 with ~2000 steps; 500 is
    a good compromise that converges in a few minutes on one A100.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_all = torch.as_tensor(project_pts, dtype=torch.float64, device=device)
    if pca is None:
        HoroPCA = _load_horopca()
        X_fit = torch.as_tensor(fit_pts, dtype=torch.float64, device=device)
        pca = HoroPCA(dim=fit_pts.shape[1], n_components=2,
                      lr=lr, max_steps=max_steps).double().to(device)
        with torch.enable_grad():
            pca.fit(X_fit, iterative=False, optim=True)
    with torch.no_grad():
        return pca.map_to_ball(X_all).cpu().numpy(), pca


# ────────────────────────── Poincaré centering ───────────────────────────────
# HoroPCA assumes the data has its Fréchet mean at the origin (see the class
# docstring in HoroPCA's learning/pca.py and the paper's Appendix C.1,
# "Centering"). Our CLIP image embeddings instead sit in a narrow cone far from
# the origin (mean pairwise cosine ≈ 0.9), so a projection onto a 2-D geodesic
# submanifold *through the origin* is dominated by that global offset and the
# whole cloud lands in a tiny off-centre blob — the "everything collapsed on one
# point" artefact. Mapping the Fréchet mean to the origin with a hyperbolic
# isometry (a circle inversion) fixes this and lets the conformal factor near the
# origin expand the fine structure so it fills the disk.

def _poincare_module():
    """Import HoroPCA's Poincaré helpers (ensures the repo is on sys.path)."""
    _load_horopca()  # idempotent; only needed for its sys.path side effect
    import geom.poincare as poincare  # type: ignore  (external repo)
    return poincare


def poincare_frechet_mean(points: np.ndarray, iters: int = 200,
                          lr: float = 0.3) -> np.ndarray:
    """Fréchet (Karcher) mean of Poincaré-ball points via Riemannian gradient
    descent. Returns a (1, D) array."""
    poincare = _poincare_module()
    x = torch.as_tensor(points, dtype=torch.float64)
    mu = poincare.project(x.mean(0, keepdim=True))
    for _ in range(iters):
        v = poincare.logmap(mu.expand_as(x), x).mean(0, keepdim=True)
        nxt = poincare.project(poincare.expmap(mu, lr * v))
        if not torch.isfinite(nxt).all():
            break
        mu = nxt
    return mu.numpy()


def center_poincare(points: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Apply the hyperbolic isometry (circle inversion) that maps `mu` to the
    origin. This is the HoroPCA/Appendix-C.1 centering step."""
    poincare = _poincare_module()
    x = torch.as_tensor(points, dtype=torch.float64)
    m = torch.as_tensor(mu, dtype=torch.float64)
    return poincare.project(poincare.reflect_at_zero(x, m)).numpy()


def _report_norms(tag: str, p: np.ndarray) -> None:
    """Print Poincaré-norm stats + how collinear the cloud is (the diagnostic
    that tells you whether centering is needed)."""
    n = np.linalg.norm(p, axis=1)
    mu = p.mean(0)
    mun = np.linalg.norm(mu)
    cos = float(((p @ mu) / (n * mun + 1e-12)).mean())
    hint = "  ← collinear cone, needs centering" if cos > 0.5 else ""
    print(f"  [{tag}] ||p||: med={np.median(n):.3f} max={n.max():.3f} | "
          f"||mean||={mun:.3f} mean-cos={cos:.3f}{hint}")


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


def plot_poincare_disk(imgs_2d, ancs_2d, gt, classes, out_path, zoom=0.0,
                       psi=None, origin_2d=None, title=None):
    """2-D HoroPCA scatter inside the unit disk.

    `zoom`: axis half-width. 0 ⇒ auto-frame to the data extent so the cloud fills
    the figure (the embeddings rarely reach the unit boundary, so the default
    full-disk view would show a tiny central blob even when correctly centred).

    `psi`: (K,) half-apertures in radians. When given, each anchor's entailment cone
    is shaded. `origin_2d` is where the MODEL's origin landed in this frame — ξ is an
    angle measured from the origin, and the Fréchet centering moves the origin away
    from the middle of the disk, so the cone axis is the direction anchor - origin,
    not the direction anchor - (0,0). Defaults to the centre if not given.

    psi is the TRUE high-dimensional aperture, not one recomputed from the projected
    radius: it is the aperture the model actually has, and watching it evolve is the
    point of the diagnostic.
    """
    _, ax = plt.subplots(figsize=(11, 11))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.2))

    max_r = float(np.linalg.norm(np.concatenate([imgs_2d, ancs_2d]), axis=1).max())
    lim = zoom if zoom > 0 else min(1.08, max(0.1, 1.15 * max_r))

    colors = _class_colors(classes)

    if psi is not None:
        # ponytail: euclidean wedge, exact only near the centre of the disk (geodesics
        # there are near-straight). If the anchors drift toward the boundary, replace
        # with the geodesic contour of oxy_angle == psi.
        o = np.zeros(2) if origin_2d is None else np.asarray(origin_2d).reshape(2)
        L = 3.0 * lim
        for i, c in enumerate(classes):
            radial = ancs_2d[i] - o
            if not np.any(radial):
                continue
            th = np.arctan2(radial[1], radial[0])
            edge = [ancs_2d[i] + L * np.array([np.cos(th + sgn * psi[i]),
                                               np.sin(th + sgn * psi[i])])
                    for sgn in (+1.0, -1.0)]
            ax.fill([ancs_2d[i, 0], edge[0][0], edge[1][0]],
                    [ancs_2d[i, 1], edge[0][1], edge[1][1]],
                    color=colors[c], alpha=0.08, lw=0, zorder=1)
            for e in edge:
                ax.plot([ancs_2d[i, 0], e[0]], [ancs_2d[i, 1], e[1]],
                        color=colors[c], lw=0.8, alpha=0.55, zorder=2)

    for c in classes:
        m = np.array([g == c for g in gt])
        if m.any():
            ax.scatter(imgs_2d[m, 0], imgs_2d[m, 1], c=[colors[c]], s=6,
                       alpha=0.4, label=f"{c} ({m.sum()})")
    if origin_2d is not None:
        # The MODEL's origin — the vertex every drawn angle is measured from, and the
        # point the cones open away from. The Fréchet centering puts the IMAGE centroid
        # at the middle of the disk, so the two are different places and "the data is in
        # the middle" is true by construction rather than a fact about the model. Without
        # this marker there is no way to read either statement off the figure.
        o = np.asarray(origin_2d).reshape(2)
        ax.scatter(*o, marker="+", s=260, c="black", linewidths=2.0, zorder=11,
                   label="model origin")
    for i, c in enumerate(classes):
        ax.scatter(ancs_2d[i, 0], ancs_2d[i, 1], c=[colors[c]], s=700,
                   marker="*", edgecolors="black", linewidths=1.8, zorder=10,
                   label=f"{c} anchor")
        ax.annotate(c, (ancs_2d[i, 0], ancs_2d[i, 1]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=11, fontweight="bold")

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    zoom_note = "" if lim >= 0.98 else f"  —  zoomed to max radius {max_r:.2f} of unit disk"
    head = "Poincaré disk projection (Lorentz → Poincaré ball → 2-D HoroPCA)"
    if title:
        head = f"{head}  —  {title}"
    if psi is not None:
        head += f"\ncones at the true 128-d ψ: {np.degrees(psi).min():.1f}°–{np.degrees(psi).max():.1f}°"
    ax.set_title(head + zoom_note)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def plot_epoch_snapshot(x_img, labels, x_anc, class_names, out_png, curv=1.0,
                        min_radius=0.1, state=None, seed=42, max_points=1500,
                        title=None, psi=None):
    """One Poincaré-disk frame per training epoch, cones included.

    `x_img` (N, D) Lorentz space-components, `labels` their integer class indices,
    `x_anc` (K, D) the anchors under the current weights.

    `state` is (pca, mu) returned by the previous call. Pass it back and the HoroPCA
    basis and the Fréchet-mean isometry are reused, which is what makes consecutive
    frames comparable — and skips the expensive half (the fit is O(N²·D) in fp64).
    Returns the state for the next epoch.

    `psi` (K,) in radians overrides the aperture. Pass it whenever the model's aperture
    is NOT `asin(2K/‖a‖)` — under `--loss axis` it is a free parameter, and recomputing
    it from the anchor norm here draws a cone the model does not have. Left None the
    coupled formula is used, which is correct for `--loss cone`.
    """
    x_anc_c = x_anc.detach().float().cpu()
    if psi is None:
        psi = half_aperture(x_anc_c, curv=curv, min_radius=min_radius).numpy()

    p_imgs = lorentz_to_poincare(np.asarray(x_img), curv=curv)
    p_ancs = lorentz_to_poincare(x_anc_c.numpy(), curv=curv)
    # The model's origin rides along: xi is an angle measured from it, and the
    # centering isometry below moves it off the middle of the disk.
    p_orig = lorentz_to_poincare(np.zeros((1, p_imgs.shape[1])), curv=curv)

    pca, mu = state if state is not None else (None, None)
    if mu is None:
        mu = poincare_frechet_mean(p_imgs)
    p_imgs, p_ancs, p_orig = (center_poincare(a, mu) for a in (p_imgs, p_ancs, p_orig))

    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(p_imgs), size=min(max_points, len(p_imgs)), replace=False)
    coords, pca = run_horopca_2d(
        np.concatenate([p_imgs[fit_idx], p_ancs], axis=0),
        np.concatenate([p_imgs, p_ancs, p_orig], axis=0),
        pca=pca,
    )
    n_i, n_a = len(p_imgs), len(p_ancs)
    imgs_2d, ancs_2d, orig_2d = coords[:n_i], coords[n_i:n_i + n_a], coords[n_i + n_a]

    plot_poincare_disk(imgs_2d, ancs_2d, [class_names[i] for i in labels],
                       class_names, out_png,
                       psi=psi, origin_2d=orig_2d, title=title)
    return pca, mu


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

    model = AttributionCLIP.from_checkpoint(ckpt).to(device)
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
    # Same contract as comparison/training/test_hypclip.py:load_anchors — free
    # anchors (image_centroid / text_free / random) live in 'anchor_tangent' and are
    # lifted with exp_map0; only text runs get re-encoded. Re-encoding a free-anchor
    # checkpoint would plot anchors the model never had.
    tangent = ckpt.get("anchor_tangent")
    if tangent is not None:
        print(f"  anchors: learned '{ckpt.get('anchor_init', 'free')}' (anchor_tangent)")
        with torch.no_grad():
            x_ancs_t = exp_map0(tangent.float().to(device), curv=curv)
    else:
        print(f"  anchors: re-encoded from the checkpoint's own anchor_texts")
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

    # Centre at the Fréchet mean so HoroPCA's through-the-origin projection isn't
    # dominated by the global offset of the cone (otherwise the disk collapses to
    # one blob). The SAME isometry is applied to the anchors so they stay in the
    # image cloud's frame. Disable with --no_center to see the broken behaviour.
    _report_norms("before centering", p_imgs)
    if not args.no_center:
        mu = poincare_frechet_mean(p_imgs)
        p_imgs = center_poincare(p_imgs, mu)
        p_ancs = center_poincare(p_ancs, mu)
        _report_norms("after centering ", p_imgs)
    else:
        print("  centering DISABLED (--no_center): expect a collapsed/off-centre disk")

    # Fit HoroPCA on a random subsample (anchors always included), then project
    # every image so the plot still shows the full val set.
    rng = np.random.default_rng(args.seed)
    n_keep = min(args.horopca_max_points, len(p_imgs))
    fit_idx = rng.choice(len(p_imgs), size=n_keep, replace=False)
    fit_pts = np.concatenate([p_imgs[fit_idx], p_ancs], axis=0)
    project_pts = np.concatenate([p_imgs, p_ancs], axis=0)
    print(f"Running HoroPCA → 2-D: fit on {len(fit_pts)} points "
          f"({args.horopca_steps} steps, lr={args.horopca_lr}), "
          f"projecting {len(project_pts)}…")
    coords_2d, _ = run_horopca_2d(fit_pts, project_pts,
                                  lr=args.horopca_lr,
                                  max_steps=args.horopca_steps)
    imgs_2d = coords_2d[:len(p_imgs)]
    ancs_2d = coords_2d[len(p_imgs):]
    _report_norms("2-D projection ", imgs_2d)
    plot_poincare_disk(imgs_2d, ancs_2d, gt, class_names,
                       out_dir / "poincare_disk.png", zoom=args.disk_zoom)

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
