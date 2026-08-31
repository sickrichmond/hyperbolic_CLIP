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

    psi = ck.get("anchor_sin_psi")
    psi = torch.rad2deg(torch.arcsin(psi.clamp(max=1.0))) if psi is not None else None
    print(f"\n{path}   K={K}")
    print(f"  sep  min {ang.min():.1f}  mean {ang.mean():.1f}  max {ang.max():.1f} deg"
          f"   (random 128-d init ~ 90; simplex ideal {torch.rad2deg(torch.arccos(torch.tensor(-1/(K-1)))):.1f})")
    if psi is not None:
        print(f"  psi  min {psi.min():.1f}  mean {psi.mean():.1f}  max {psi.max():.1f} deg"
              f"   spread {psi.max()-psi.min():.1f}")
        # A pair overlaps when their axes are closer than the sum of the two apertures.
        s = psi[iu[0]] + psi[iu[1]]
        print(f"  overlapping pairs: {(ang < s).sum()}/{len(ang)}"
              f"   worst deficit {(s - ang).max():.1f} deg")

    names = ck.get("class_names")
    if names is not None:
        k = min(5, len(ang))
        for v, i in zip(*ang.topk(k, largest=False)):
            print(f"    closest: {names[iu[0][i]]:<14} {names[iu[1][i]]:<14} {v:.1f} deg")
