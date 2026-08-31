"""Pairwise anchor angles from a checkpoint.

The Poincare snapshots cannot answer "do the cones overlap": 22 near-orthogonal
directions in 128-d all project onto ~one point of any 2-D basis, which is why the
INIT frame — known mean angle 89.9 deg — already looks like a single blob with fully
overlapping cones. The angles have to be read from the anchors themselves.

Usage:  python scripts/anchor_separation.py <ckpt.pt> [more.pt ...]
"""
import sys
import torch
import torch.nn.functional as F

for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    t = ck["anchor_tangent"]                      # (K, D); exp_map0 is radial, so the
    d = F.normalize(t.float(), dim=-1)            # tangent direction IS the axis direction
    K = d.shape[0]
    cos = (d @ d.T).clamp(-1 + 1e-6, 1 - 1e-6)
    iu = torch.triu_indices(K, K, offset=1)
    ang = torch.rad2deg(torch.arccos(cos[iu[0], iu[1]]))

    # --loss axis stores a free aperture; --loss cone couples it to the depth, so
    # re-derive it there rather than printing nothing (the coupled psi is what was in
    # force during a cone run, and the overlap check below is the whole point).
    sin_psi = ck.get("anchor_sin_psi")
    if sin_psi is None:
        # half_aperture divides by ||x||*sqrt(c) and exp_map0 gives ||x|| =
        # sinh(sqrt(c)*||t||)/sqrt(c), so the two sqrt(c) cancel exactly.
        rc = ck.get("curv", 1.0) ** 0.5
        sin_psi = (2.0 * ck["min_radius"]
                   / torch.sinh(rc * t.float().norm(dim=-1))).clamp(max=1.0)
    psi = torch.rad2deg(torch.arcsin(sin_psi.clamp(max=1.0)))
    print(f"\n{path}   K={K}")
    print(f"  sep  min {ang.min():.1f}  mean {ang.mean():.1f}  max {ang.max():.1f} deg"
          f"   (random 128-d init ~ 90; simplex ideal {torch.rad2deg(torch.arccos(torch.tensor(-1/(K-1)))):.1f})")
    print(f"  psi  min {psi.min():.1f}  mean {psi.mean():.1f}  max {psi.max():.1f} deg"
          f"   spread {psi.max()-psi.min():.1f}"
          f"   ({'free' if ck.get('anchor_sin_psi') is not None else 'coupled, re-derived'})")
    # A pair overlaps when its axes are closer than the sum of the two apertures.
    need = psi[iu[0]] + psi[iu[1]]
    print(f"  overlapping pairs: {(ang < need).sum()}/{len(ang)}"
          f"   worst deficit {(need - ang).max():.1f} deg")
    # Stop criterion 1a, both forms. The PAIRWISE max(need/ang) is the exact
    # disjointness ratio; 2*mean(psi)/min(ang) is what the epoch line and every
    # recorded number use (sweepwin 12.8, Phase B `flat` 1.2), so it stays for
    # comparability. They coincide only while psi is uniform — which it is today,
    # and stops being the moment the aperture actually spreads.
    print(f"  criterion 1a (< 1):  pairwise {(need / ang).max():.1f}"
          f"   legacy 2psi/min-angle {2 * psi.mean() / ang.min():.1f}")

    names = ck.get("class_names")
    if names is not None:
        k = min(5, len(ang))
        for v, i in zip(*ang.topk(k, largest=False)):
            print(f"    closest: {names[iu[0][i]]:<14} {names[iu[1][i]]:<14} {v:.1f} deg")
