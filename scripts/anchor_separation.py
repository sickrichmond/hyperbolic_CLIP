"""Report high-dimensional anchor angles and angular cone-overlap diagnostics.

Read saved anchor tangents and derive depth-coupled apertures. Angular-cap overlap
is tested against the sum of pairwise half-apertures;
it is not inferred from a 2-D projection.

Usage: python scripts/anchor_separation.py checkpoint.pt [more.pt ...]
"""
import sys
import torch
import torch.nn.functional as F

for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("loss", "cone") != "cone":
        raise ValueError(f"{path}: only entailment-cone checkpoints are supported")
    t = ck["anchor_tangent"]                      # (K, D); exp_map0 is radial, so the
    d = F.normalize(t.float(), dim=-1)            # tangent direction IS the axis direction
    K = d.shape[0]
    cos = (d @ d.T).clamp(-1 + 1e-6, 1 - 1e-6)
    iu = torch.triu_indices(K, K, offset=1)
    ang = torch.rad2deg(torch.arccos(cos[iu[0], iu[1]]))

    # exp_map0 and half_aperture have cancelling sqrt(curvature) factors.
    rc = ck.get("curv", 1.0) ** 0.5
    sin_psi = (2.0 * ck["min_radius"]
               / torch.sinh(rc * t.float().norm(dim=-1))).clamp(max=1.0)
    psi = torch.rad2deg(torch.arcsin(sin_psi.clamp(max=1.0)))
    print(f"\n{path}   K={K}")
    print(f"  sep  min {ang.min():.1f}  mean {ang.mean():.1f}  max {ang.max():.1f} deg")
    print(f"  psi  min {psi.min():.1f}  mean {psi.mean():.1f}  max {psi.max():.1f} deg"
          f"   spread {psi.max()-psi.min():.1f}"
          f"   (depth-coupled)")
    # A pair overlaps when its axes are closer than the sum of the two apertures.
    need = psi[iu[0]] + psi[iu[1]]
    print(f"  overlapping pairs: {(ang < need).sum()}/{len(ang)}"
          f"   worst deficit {(need - ang).max():.1f} deg")
    # The pairwise ratio tests angular-cap separation. The aggregate ratio
    # agrees with it only for equal apertures.
    print(f"  criterion 1a (< 1):  pairwise {(need / ang).max():.1f}"
          f"   legacy 2psi/min-angle {2 * psi.mean() / ang.min():.1f}")

    names = ck.get("class_names")
    if names is not None:
        k = min(5, len(ang))
        for v, i in zip(*ang.topk(k, largest=False)):
            print(f"    closest: {names[iu[0][i]]:<14} {names[iu[1][i]]:<14} {v:.1f} deg")
