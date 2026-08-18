"""Is this feature space tree-like enough to deserve a hyperbolic model?

The whole premise of HypCLIP is that hyperbolic space fits the data better than a
sphere. Nobody ever tested it: `grep -rE 'hyperbolicity|gromov|delta_hyp'` over the
repo returns nothing. This is the standard justification measure in the literature
(Khrulkov et al., "Hyperbolic Image Embeddings", CVPR 2020, §5.1) and it is the
first thing a reviewer will ask for.

Gromov's four-point delta, computed from a fixed base point w (Fournier et al.):

    G[i][j] = ½ (d(i,w) + d(j,w) − d(i,j))          the Gromov product
    δ       = max (G ⊗ G − G),   (G ⊗ G)[i][j] = max_k min(G[i][k], G[k][j])
    δ_rel   = 2δ / diam

δ_rel ∈ [0, 1]. **0 = an exact tree metric** (hyperbolic geometry is the right
embedding space); **1 = maximally non-tree-like** (a 4-cycle hits exactly 1).
Khrulkov reports ≈0.2–0.3 for natural image datasets — a value in that band means
this data is no more tree-like than any other image set, and the geometry needs to
be justified by what it *does* (hierarchy, open-set) rather than by how the
features are shaped.

Runs on the feature caches `scripts/extract_clip_features.py` already writes, so it
needs no forward pass of its own:

    python -m tests.probe_hyperbolicity --selfcheck
    python -m tests.probe_hyperbolicity \\
        frozen=$WORK/hyp_fine_tuning/clip_features_frozen \\
        lora=$WORK/hyp_fine_tuning/clip_features_lora \\
        projection=$WORK/hyp_fine_tuning/clip_features_projection

CPU is enough at n=1500; a GPU makes the O(n³) min-max product instant.
"""
import argparse
import sys
from pathlib import Path

import torch


def delta_hyperbolicity(D):
    """(δ, δ_rel) from a square distance matrix, base point = the row-0 point.

    A fixed base point is the usual trade: the true δ maximises over all of them,
    but δ_w ≤ 2δ for any w, so the number is an estimate of the right order and the
    O(n⁴) version buys nothing at this scale.
    """
    row = D[0]
    G = 0.5 * (row.unsqueeze(1) + row.unsqueeze(0) - D)

    # (G ⊗ G)[i][j] = max_k min(G[i][k], G[k][j]), accumulated one k at a time so
    # the n³ intermediate never exists.
    maxmin = torch.full_like(G, float("-inf"))
    for k in range(G.shape[0]):
        maxmin = torch.maximum(maxmin, torch.minimum(G[:, k].unsqueeze(1), G[k].unsqueeze(0)))

    delta = (maxmin - G).max().item()
    diam = D.max().item()
    return delta, 2 * delta / diam


def sample_distances(path, n, seed, device):
    """n random rows of a cached feature tensor -> their Euclidean distance matrix."""
    cache = Path(path)
    if cache.is_dir():
        cache = cache / "clip_features_val.pt"
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    X = blob["X"].float()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:n]
    X = X[idx].to(device)
    return torch.cdist(X, X), tuple(X.shape), blob.get("source", "?")


def selfcheck():
    """Two metrics with a δ_rel known in closed form, so the estimator is pinned down."""
    # A star tree: 4 leaves on a centre, unit edges. Tree metrics have δ = 0 exactly.
    tree = torch.tensor([[0., 2, 2, 2], [2, 0, 2, 2], [2, 2, 0, 2], [2, 2, 2, 0]])
    d, rel = delta_hyperbolicity(tree)
    assert abs(d) < 1e-6 and abs(rel) < 1e-6, (d, rel)

    # The 4-cycle: the textbook worst case, δ = 1 with diam 2 -> δ_rel = 1.
    cycle = torch.tensor([[0., 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
    d, rel = delta_hyperbolicity(cycle)
    assert abs(d - 1.0) < 1e-6 and abs(rel - 1.0) < 1e-6, (d, rel)

    # A caterpillar tree (path of 3 internal nodes, one leaf each) is still δ = 0,
    # and unlike the star it has a non-trivial diameter.
    path = torch.tensor([[0., 3, 4, 5], [3, 0, 3, 4], [4, 3, 0, 3], [5, 4, 3, 0]])
    d, _ = delta_hyperbolicity(path)
    assert abs(d) < 1e-6, d
    print("selfcheck ok: tree δ_rel=0.000, 4-cycle δ_rel=1.000")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="*", metavar="label=DIR_OR_PT",
                   help="feature caches from scripts.extract_clip_features")
    p.add_argument("-n", type=int, default=1500, help="points sampled per source")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()

    if args.selfcheck:
        return selfcheck()
    if not args.sources:
        sys.exit("usage: probe_hyperbolicity.py label=DIR [label=DIR ...]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{args.n} points per source, seed {args.seed}, on {device}\n")
    print(f"{'source':16s} {'dim':>6s} {'diam':>8s} {'δ':>8s} {'δ_rel':>8s}")
    for spec in args.sources:
        label, _, path = spec.partition("=")
        D, shape, desc = sample_distances(path or label, args.n, args.seed, device)
        delta, rel = delta_hyperbolicity(D)
        print(f"{label:16s} {shape[1]:6d} {D.max():8.3f} {delta:8.4f} {rel:8.4f}   {desc}")

    print("\n0 = exact tree metric, 1 = a 4-cycle. Khrulkov et al. (CVPR 2020) report\n"
          "≈0.2-0.3 for natural image datasets; landing there means the geometry has to\n"
          "be justified by hierarchy and open-set, not by the shape of the features.")


if __name__ == "__main__":
    main()
