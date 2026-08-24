"""How much of the sphere do the class anchors actually occupy — hyperbolic or euclidean?

This is the mechanism behind the robustness gap. On the cone model every image sits
within 0.4° of its nearest anchor (max_c cos median 0.999976), which sounds like a
clean collapse onto 22 well-separated rays — but only if the anchors themselves are
spread apart. They are not: all 22 fit inside 9°, with the closest pair 2.1° apart.
A perturbation of one degree crosses into the neighbouring class, which is exactly
what JPEG does to the cone model (AUC 0.64) and not to the euclidean one (0.94).

So the question this answers is: does the euclidean model spread its anchors, or is
it packed just as tightly and the difference lies elsewhere?

Anchors are compared by DIRECTION in the projected space — the space the classifier
actually decides in. For the hyperbolic model that is the hyperboloid's space
component (exp_map0 is radial, so direction is preserved from the tangent space); for
the euclidean model it is the unit-sphere embedding. Both are read through the same
`load_anchors` the evaluators use, so the checkpoint's own prompts are honoured.

    IAB_EXCLUDE_GENERATORS=dalle3 python -m tests.probe_anchor_spread \\
        $CK/attribution_22cls_sweepwin_vitl14.pt \\
        $CK/attribution_22cls_promptsA_vitl14.pt \\
        $CK/attribution_22cls_euclidean_d128_vitl14.pt

No images, no GPU needed: it only encodes 22 sentences per checkpoint. IAB_EXCLUDE_GENERATORS
must select the same label space the checkpoint was trained on (e.g. `dalle3,infinity`
for a held-out run), otherwise load_anchors says so and stops.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path

os.environ.setdefault("IAB_EXCLUDE_GENERATORS", "dalle3")

import torch
import torch.nn.functional as F

from comparison.training.test_hypclip import harness_class_names
from geometry.lorentz import half_aperture


def load_any(path, device):
    """(model, anchors, ψ or None, description, clip_name) for either geometry.

    Shared with tests/probe_degradation_shift.py — the checkpoint-loading contract
    (which model class, which load_anchors, the anchor permutation) lives here once.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    geometry = ckpt.get("geometry", "hyperbolic")

    if geometry == "euclidean":
        from comparison.training.test_euclidean import load_anchors
        from models.euclidean_attribution_clip import EuclideanAttributionCLIP
        model = EuclideanAttributionCLIP(
            clip_name=ckpt["clip_name"], lora_r=ckpt.get("lora_r", 8),
            lora_alpha=ckpt.get("lora_alpha", 16),
            embed_dim=ckpt.get("embed_dim", 128)).to(device)
        model.clip.load_state_dict(ckpt["lora_state"])
        model.projection.load_state_dict(ckpt["projection"])
        model.logit_scale.data = ckpt["logit_scale"].to(device)
        model.eval()
        x_anc = load_anchors(ckpt, model, device)
        # The CE turns cosine gaps into logit gaps by this factor, so it is what says
        # whether tight packing is actually a problem for THIS model.
        scale = min(model.logit_scale.exp().item(), 100.0)
        return model, x_anc, None, f"euclidean, logit_scale={scale:.1f}", ckpt["clip_name"]

    from comparison.training.test_hypclip import load_anchors
    from models.attribution_clip import AttributionCLIP
    curv = ckpt.get("curv", 1.0)
    model = AttributionCLIP.from_checkpoint(ckpt).to(device)
    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection"])
    model.eval()
    x_anc = load_anchors(ckpt, model, curv, device)
    psi = half_aperture(x_anc, curv=curv, min_radius=ckpt.get("min_radius", 0.1))
    return (model, x_anc, psi,
            f"hyperbolic, curv={curv:g}, ‖x_anc‖≈{x_anc.norm(dim=-1).mean():.2f}",
            ckpt["clip_name"])


def report(path, device, names, dump=None):
    model, x_anc, psi, desc, _ = load_any(path, device)
    # Detached and the model dropped before the next checkpoint: three ViT-L/14s alive
    # at once is an OOM kill on a login node, and x_anc would otherwise keep the whole
    # text-encoder graph reachable.
    x_anc = x_anc.detach()
    K = x_anc.shape[0]
    if K != len(names):
        raise ValueError(f"{os.path.basename(path)}: {K} anchors but the active label map has "
                         f"{len(names)} — set IAB_EXCLUDE_GENERATORS to match this checkpoint")

    d = F.normalize(x_anc, dim=-1)
    A = (d @ d.T).clamp(-1.0, 1.0)
    i, j = torch.triu_indices(K, K, offset=1)
    off = A[i, j]
    deg = lambda c: math.degrees(math.acos(min(max(float(c), -1.0), 1.0)))

    print(f"\n═════ {os.path.basename(path)}   ({desc})")
    print(f"  anchor↔anchor cos : max={off.max():.4f}  mean={off.mean():.4f}  min={off.min():.4f}")
    print(f"  as angles         : closest pair {deg(off.max()):.1f}°  "
          f"mean {deg(off.mean()):.1f}°  WIDEST PAIR {deg(off.min()):.1f}°")
    if psi is not None:
        print(f"  ψ                 : mean={psi.mean():.4f} rad ({math.degrees(psi.mean()):.1f}°)  "
              f"spread={psi.max() - psi.min():.4f}")
        # Entailment cones are disjoint only where the apexes are further apart than
        # ψ_c + ψ_c'. The trainer checks this at init, but only for free anchors
        # (train_attribution.py:494 gates the whole block on anchor_init != "text"),
        # so no text-anchor run has ever printed it. Ratio > 1 = nominally overlapping.
        closest = deg(off.max())
        ratio = f"{2 * math.degrees(psi.mean()) / closest:.1f}" if closest else "inf"
        print(f"  2ψ / closest pair : {ratio}"
              f"   (>1 means the cones overlap at the anchors' own depth)")
    for r in off.argsort(descending=True)[:3]:
        print(f"      {off[r]:.4f}  ({deg(off[r]):4.1f}°)  {names[i[r]]} ↔ {names[j[r]]}")

    if dump:
        # The full angle matrix is the second, independent source for
        # comparison.training.scripts.extract_tree — which is pure stdlib and so
        # cannot encode prompts itself.
        out = Path(dump) / (Path(path).stem + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "class_names": names,
            "angles_deg": [[deg(c) for c in row] for row in A.tolist()],
        }) + "\n")
        print(f"  dumped angle matrix -> {out}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return deg(off.min()), deg(off.max())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--dump", metavar="DIR",
                   help="write <DIR>/<ckpt>.json with the full angle matrix, for extract_tree")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = harness_class_names()
    print(f"{len(names)} classes in the active label map "
          f"(IAB_EXCLUDE_GENERATORS={os.environ['IAB_EXCLUDE_GENERATORS']!r})")

    rows = []
    for path in args.checkpoints:
        rows.append((os.path.basename(path), *report(path, device, names, args.dump)))

    print(f"\n{'checkpoint':52s} {'widest':>8s} {'closest':>8s}")
    for name, widest, closest in rows:
        print(f"{name:52s} {widest:7.1f}° {closest:7.1f}°")
    print("\n'widest' is how much of the sphere the class directions span; 'closest' is the\n"
          "margin the two most confusable classes have. Images sit ~0.4° from their anchor,\n"
          "so a 'closest' of a couple of degrees means a small perturbation changes the class.")


if __name__ == "__main__":
    main()
