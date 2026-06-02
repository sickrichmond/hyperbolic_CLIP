"""
Visualise Lorentz-space embeddings with HoroPCA + UMAP.

Pipeline:
  1. Load embeddings from .npz (saved by tests/extract_embeddings.py).
  2. Convert Lorentz space components → Poincaré-ball coordinates (HoroPCA's
     native input space).
  3. HoroPCA: reduce D → n_pca dims (hyperbolic PCA on horocycles).
  4. UMAP: project the HoroPCA output to 3D with euclidean metric.
  5. 3D scatter plot, colored by class (style matching the HySAC paper).

Requires:
  - HoroPCA cloned at external/HoroPCA (so we can import its modules)
  - umap-learn installed in the env
  - matplotlib

Usage:
    python -m tests.visualize_horopca \\
        --embeddings $WORK/embeddings/val_hier.npz \\
        --output     $WORK/figures/val_hier_horopca.png \\
        --n_pca      8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless on compute nodes
import matplotlib.pyplot as plt

# Make the HoroPCA repo importable
REPO_ROOT = Path(__file__).resolve().parents[1]
HOROPCA_PATH = REPO_ROOT / "external" / "HoroPCA"
if not HOROPCA_PATH.exists():
    sys.exit(
        f"HoroPCA not found at {HOROPCA_PATH}.\n"
        f"Clone it first:\n"
        f"  git clone https://github.com/HazyResearch/HoroPCA {HOROPCA_PATH}"
    )
sys.path.insert(0, str(HOROPCA_PATH))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", required=True, help=".npz from extract_embeddings.py")
    p.add_argument("--output_prefix", required=True,
                   help="Prefix for the two PNGs produced; "
                        "<prefix>_by_class.png and <prefix>_by_semantic.png are written.")
    p.add_argument("--n_pca",      type=int,   default=8,
                   help="Output dimension of HoroPCA before UMAP.")
    p.add_argument("--n_neighbors", type=int,  default=30, help="UMAP n_neighbors.")
    p.add_argument("--min_dist",    type=float, default=0.1, help="UMAP min_dist.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--max_points",  type=int, default=3000,
                   help="Subsample to this many points before HoroPCA. HoroPCA "
                        "builds (N, N) bilinear matrices internally, so memory "
                        "scales quadratically. Keep ≲ 5000 unless you have RAM.")
    return p.parse_args()


def lorentz_to_poincare(x_space: np.ndarray, curv: float) -> np.ndarray:
    """
    Lorentz space components → Poincaré-ball coordinates.

    Lorentz point (with curvature c):  x_time = sqrt(1/c + ||x_space||²)
    Stereographic projection from (-1/√c, 0) onto the disk gives:
        x_ball = x_space / (x_time + 1/√c)
    Result lies in the open ball of radius 1/√c.
    """
    x_time = np.sqrt(1.0 / curv + np.sum(x_space ** 2, axis=-1, keepdims=True))
    return x_space / (x_time + 1.0 / np.sqrt(curv))


def _patch_torch_solve():
    """HoroPCA calls the removed torch.solve(B, A) → (solution, LU). Reimplement
    it on top of torch.linalg.solve(A, B), which is the recommended replacement."""
    import torch
    if not getattr(torch, "_horopca_solve_patched", False):
        def _solve(B, A):
            return torch.linalg.solve(A, B), None
        torch.solve = _solve
        torch._horopca_solve_patched = True


def run_horopca(x_ball: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    """Apply HoroPCA to Poincaré-ball points, return (N, n_components).

    fit_optim runs hundreds of gradient steps with autograd through asinh/acosh.
    On CPU this is very slow (>1h for N=3000) — we move everything to GPU when
    available."""
    import torch
    _patch_torch_solve()
    from learning.pca import HoroPCA   # type: ignore  (lives in external/HoroPCA)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Force fp64 — HoroPCA's Minkowski ops mix dtypes and asinh/acosh benefit
    # from higher precision.
    X = torch.as_tensor(x_ball, dtype=torch.float64, device=device)
    pca = HoroPCA(dim=x_ball.shape[1], n_components=n_components).double().to(device)
    pca.fit(X, iterative=False, optim=True)
    with torch.no_grad():
        Z = pca.map_to_ball(X).cpu().numpy()
    return Z


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data = np.load(args.embeddings, allow_pickle=True)
    lorentz       = data["lorentz"]                 # (N, D)
    anchors       = data["anchors"]                 # (K, D)
    labels        = data["labels"]                  # (N,)
    class_names   = list(data["class_names"])
    generators    = list(data["generators"])
    semantics     = list(data["semantics"])
    curv          = float(data["curv"][0])

    print(f"Loaded {len(lorentz)} embeddings ({lorentz.shape[1]}D), "
          f"{len(anchors)} anchors, curv={curv}")

    # Optionally subsample for speed (UMAP scales as O(N log N) but the matplotlib
    # scatter is the real bottleneck for plotting > tens of thousands of points).
    if args.max_points and len(lorentz) > args.max_points:
        idx = rng.choice(len(lorentz), size=args.max_points, replace=False)
        lorentz, labels = lorentz[idx], labels[idx]
        generators = [generators[i] for i in idx]
        semantics  = [semantics[i]  for i in idx]
        print(f"Subsampled to {args.max_points} points")

    # ─── Lorentz → Poincaré ───────────────────────────────────────────────────
    ball_imgs    = lorentz_to_poincare(lorentz, curv=curv)
    ball_anchors = lorentz_to_poincare(anchors, curv=curv)

    # ─── HoroPCA ──────────────────────────────────────────────────────────────
    print(f"Running HoroPCA → {args.n_pca}D ...")
    X_all = np.concatenate([ball_imgs, ball_anchors], axis=0)  # fit jointly so
                                                                # anchors share basis
    Z_all = run_horopca(X_all, n_components=args.n_pca, seed=args.seed)
    Z_imgs    = Z_all[:len(ball_imgs)]
    Z_anchors = Z_all[len(ball_imgs):]

    # ─── UMAP → 3D ────────────────────────────────────────────────────────────
    print(f"Running UMAP → 3D (n_neighbors={args.n_neighbors}, min_dist={args.min_dist}) ...")
    import umap
    reducer = umap.UMAP(n_components=3, n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist, random_state=args.seed,
                        metric="euclidean")
    Y_all = reducer.fit_transform(np.concatenate([Z_imgs, Z_anchors], axis=0))
    Y_imgs    = Y_all[:len(Z_imgs)]
    Y_anchors = Y_all[len(Z_imgs):]

    # ─── Two plots from the SAME UMAP layout ─────────────────────────────────
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # By class (real vs FLUX) — HySAC palette
    if len(class_names) == 2:
        palette = {"real": "#E53935", "FLUX": "#7E57C2"}
        class_colors = [palette.get(n, plt.get_cmap("tab10")(i))
                        for i, n in enumerate(class_names)]
    else:
        class_colors = [plt.get_cmap("tab10")(i) for i in range(len(class_names))]
    _plot_3d(Y_imgs, Y_anchors,
             groups=labels, group_names=class_names, colors=class_colors,
             class_names=class_names, anchor_colors=class_colors,
             out=out_prefix.with_name(out_prefix.name + "_by_class.png"))

    # By semantic (10 classes)
    uniq_sem = sorted(set(semantics))
    sem_to_idx = {s: i for i, s in enumerate(uniq_sem)}
    sem_groups = np.array([sem_to_idx[s] for s in semantics])
    sem_cmap = plt.get_cmap("tab10" if len(uniq_sem) <= 10 else "tab20")
    sem_colors = [sem_cmap(i) for i in range(len(uniq_sem))]
    _plot_3d(Y_imgs, Y_anchors,
             groups=sem_groups, group_names=uniq_sem, colors=sem_colors,
             class_names=class_names, anchor_colors=["#444"] * len(class_names),
             out=out_prefix.with_name(out_prefix.name + "_by_semantic.png"))


def _plot_3d(Y_imgs, Y_anchors, *, groups, group_names, colors,
             class_names, anchor_colors, out: Path) -> None:
    """Render a single 3D scatter and save it to `out`."""
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for k, name in enumerate(group_names):
        mask = (groups == k)
        ax.scatter(Y_imgs[mask, 0], Y_imgs[mask, 1], Y_imgs[mask, 2],
                   s=6, alpha=0.55, color=colors[k],
                   label=f"{name} ({mask.sum()})", linewidth=0)

    for k, name in enumerate(class_names):
        ax.scatter(Y_anchors[k, 0], Y_anchors[k, 1], Y_anchors[k, 2],
                   marker="*", s=320, edgecolor="black", linewidth=1.5,
                   color=anchor_colors[k], depthshade=False,
                   label=f"anchor: {name}")

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    ax.set_facecolor("white")
    ax.xaxis.pane.set_edgecolor("lightgrey")
    ax.yaxis.pane.set_edgecolor("lightgrey")
    ax.zaxis.pane.set_edgecolor("lightgrey")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()

    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved figure → {out}")


if __name__ == "__main__":
    main()
