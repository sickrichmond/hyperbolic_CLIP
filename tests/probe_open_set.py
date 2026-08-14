"""Do the cones buy anything on an UNKNOWN generator, where cosine cannot?

Closed-set is settled: `argmin ξ` and `argmax cos` agree at 0.9998
(tests/probe_cone_vs_cosine.py) and a plain nn.Linear on the same 128-d vectors
beats the cones 99.2 vs 98.86. So on 22-way accuracy the geometry buys nothing.

One degree of freedom is left, and it is the only one the sphere does not have:
the NORM of the image embedding. cos is scale-invariant by construction; ξ is not
(oxy_angle divides by ‖x‖). In entailment-cone terms an image that belongs to no
leaf should stay near the origin — generic, uncommitted — and therefore fall
outside every cone. That is a claim about open-set rejection, and it is testable
for free: `dalle3` is on disk and excluded from training by IAB_EXCLUDE_GENERATORS.

Four scores over the SAME embeddings, oriented so higher == "unknown":

    xi_min    min_c ξ_c                the model's own confidence
    neg_cos   -max_c cos(x, a_c)       the same thing without the geometry
    neg_norm  -‖x_img‖                 the radius alone, no anchors at all
    margin    min_c (ξ_c - ψ_c)        the PARAMETER-FREE cone rule: >0 == outside
                                       every cone == "none of the above"

Read `AUROC`. xi_min meaningfully above neg_cos is the contribution — the geometry
rejecting what cosine cannot. Equal, and the conclusion is that a scalar
--target_norm disables depth by construction and the fix is a per-hierarchy-level
target_norm, not a better threshold. `margin` matters separately: it is the only
score here that needs no held-out unknowns to pick a threshold.

    IAB_EXCLUDE_GENERATORS=dalle3 python -m tests.probe_open_set \\
        $WORK/hyp_fine_tuning/checkpoints/attribution_22cls_sweepwin_vitl14.pt

Standalone: touches no repo file. GPU node. `--selfcheck` needs neither GPU nor data.
"""
import argparse
import os

os.environ.setdefault("IAB_EXCLUDE_GENERATORS", "dalle3")   # 22-class anchors

import json

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.iab_clip_dataset import IABCLIPDataset
from geometry.lorentz import half_aperture, oxy_angle
from models.attribution_clip import AttributionCLIP

SEMANTICS = ["COCO", "cat", "dog", "wild", "FFHQ", "celebahq", "bedroom", "church",
             "classroom", "ImageNet-1k"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?")
    p.add_argument("--dataset_path", default=os.environ.get("FAST", "") + "/datasets/iab_dataset")
    p.add_argument("--captions_dir", default=os.environ.get("WORK", "") + "/hyp_fine_tuning/iab_captions")
    p.add_argument("--manifest", default=os.environ.get("WORK", "") + "/hyp_fine_tuning/split_manifest_22cls.json",
                   help="known images are the manifest's VAL split — held out, and the "
                        "same images the linear probes were scored on")
    p.add_argument("--unknown", default="dalle3",
                   help="the held-out generator. Must NOT be in the checkpoint's label "
                        "space, i.e. must be in IAB_EXCLUDE_GENERATORS.")
    p.add_argument("--n", type=int, default=8000, help="images per side")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--anchors_only", action="store_true",
                   help="print the anchor geometry and stop — no images, ~30s")
    return p.parse_args()


def score_batch(x_img, x_anc, psi, curv):
    """(B, D) image points -> the four scores, all oriented higher == unknown."""
    B, K = x_img.shape[0], x_anc.shape[0]
    xi = oxy_angle(x_anc.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1),
                   x_img.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1),
                   curv=curv).reshape(B, K)
    cos = F.normalize(x_img, dim=-1) @ F.normalize(x_anc, dim=-1).T
    return {
        "xi_min": xi.min(1).values,
        "neg_cos": -cos.max(1).values,
        "neg_norm": -x_img.norm(dim=-1),
        "margin": (xi - psi.unsqueeze(0)).min(1).values,
        "pred": xi.argmin(1).float(),      # not a score — the diagnostic below
    }


@torch.no_grad()
def collect(loader, model, x_anc, psi, curv, device, desc):
    out = {}
    for batch in tqdm(loader, desc=desc, leave=False):
        x_img, _ = model.encode_image(batch["pixel_values"].to(device))
        for k, v in score_batch(x_img, x_anc, psi, curv).items():
            out.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def subset(ds, n, seed=0):
    if len(ds) <= n:
        return ds
    g = torch.Generator().manual_seed(seed)
    return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())


def _selfcheck():
    """The score orientations, on hand-built points. No GPU, no data."""
    curv = 1.0
    from geometry.lorentz import exp_map0
    anc = exp_map0(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), curv=curv)
    psi = half_aperture(anc, curv=curv)
    # DEEPER along anchor 0's ray (inside its cone) vs near the origin, between the two.
    # Not the anchor itself: oxy_angle(x, x) is the degenerate case and returns π/2.
    img = exp_map0(torch.tensor([[3.0, 0.0], [0.2, 0.2]]), curv=curv)
    s = score_batch(img, anc, psi, curv)
    assert s["xi_min"][0] < s["xi_min"][1], s["xi_min"]
    assert s["neg_cos"][0] < s["neg_cos"][1], s["neg_cos"]
    assert s["neg_norm"][0] < s["neg_norm"][1], s["neg_norm"]
    assert s["margin"][0] < 0 < s["margin"][1], s["margin"]   # inside vs outside every cone
    print(f"selfcheck OK  (ψ={psi.tolist()}, margins={s['margin'].tolist()})")


def main():
    args = parse_args()
    if args.selfcheck:
        return _selfcheck()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from comparison.training.test_hypclip import harness_class_names, load_anchors
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    curv = ckpt.get("curv", 1.0)
    model = AttributionCLIP(
        clip_name=ckpt["clip_name"], lora_r=ckpt.get("lora_r", 8),
        lora_alpha=ckpt.get("lora_alpha", 16),
        hyperbolic_dim=ckpt.get("hyperbolic_dim", 128), curv=curv,
    ).to(device)
    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection"])
    model.eval()

    x_anc = load_anchors(ckpt, model, curv, device)
    if args.unknown in harness_class_names():
        raise ValueError(
            f"'{args.unknown}' is in the label space, so it is not unknown to this model. "
            f"Set IAB_EXCLUDE_GENERATORS to include it (e.g. dalle3,{args.unknown}).")
    psi = half_aperture(x_anc, curv=curv, min_radius=ckpt.get("min_radius", 0.1))
    print(f"ψ: min={psi.min():.4f} max={psi.max():.4f} spread={psi.max() - psi.min():.4f}")

    # Decisive for reading max_c cos(x, a_c) ≈ 1: it is only evidence of per-class
    # collapse if the ANCHORS are spread apart. If the anchors are themselves mutually
    # collinear, everything lives in one narrow ray bundle and the classifier separates
    # on the 5th decimal — a very different, and much worse, picture.
    names = harness_class_names()
    A = F.normalize(x_anc, dim=-1) @ F.normalize(x_anc, dim=-1).T
    off = A[~torch.eye(len(names), dtype=torch.bool, device=A.device)]
    print(f"anchor↔anchor cos (projected space): max={off.max():.4f} "
          f"mean={off.mean():.4f} min={off.min():.4f}")
    i, j = torch.triu_indices(len(names), len(names), offset=1)
    for r in A[i, j].argsort(descending=True)[:3]:
        print(f"    {A[i[r], j[r]]:.4f}  {names[i[r]]} ↔ {names[j[r]]}")
    if args.anchors_only:
        return

    with open(args.manifest) as f:
        val_paths = set(json.load(f)["val"])
    common = dict(root=args.dataset_path, captions_dir=args.captions_dir,
                  semantics=SEMANTICS, processor_name=ckpt["clip_name"], split="all",
                  require_caption=False, include_uncaptioned=True)
    # `names` is the model's own label space, so the known side is exactly the classes
    # it was trained on — held out via the manifest's val split.
    known_ds = IABCLIPDataset(**common, generators=names, include_paths=val_paths)
    unknown_ds = IABCLIPDataset(**common, generators=[args.unknown], max_per_class=args.n)
    print(f"known ({len(names)} classes, manifest val): {len(known_ds)} → "
          f"{min(len(known_ds), args.n)}   unknown ({args.unknown}): {len(unknown_ds)}")

    def run(ds, desc):
        loader = DataLoader(subset(ds, args.n), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
        return collect(loader, model, x_anc, psi, curv, device, desc)

    k = run(known_ds, "known")
    u = run(unknown_ds, "unknown")

    n_k, n_u = len(k["xi_min"]), len(u["xi_min"])
    y = torch.cat([torch.zeros(n_k), torch.ones(n_u)]).numpy()
    print(f"\n{n_k} known vs {n_u} unknown        (AUROC 0.5 = the score is blind)")
    print(f"{'score':10s} {'AUROC':>7s} {'FPR@95TPR':>10s}   mean known / mean unknown")
    for name in ("xi_min", "neg_cos", "neg_norm", "margin"):
        s = torch.cat([k[name], u[name]])
        thr = torch.quantile(u[name], 0.05)          # threshold catching 95% of unknowns
        fpr = (k[name] >= thr).float().mean().item()  # known wrongly rejected there
        print(f"{name:10s} {roc_auc_score(y, s.numpy()):7.4f} {fpr:10.4f}   "
              f"{k[name].mean():+.4f} / {u[name].mean():+.4f}")

    # The control: if ξ ranks images exactly as cosine does, it cannot beat it, and any
    # AUROC gap above is noise. Only ‖x_img‖ can make them differ.
    rank = lambda t: t.argsort().argsort().float()
    rho = torch.corrcoef(torch.stack([
        rank(torch.cat([k["xi_min"], u["xi_min"]])),
        rank(torch.cat([k["neg_cos"], u["neg_cos"]]))]))[0, 1]
    print(f"\nSpearman(xi_min, neg_cos) = {rho:.4f}   "
          f"(1.0 == same ranking == the norm carries nothing)")

    rej_k = (k["margin"] > 0).float().mean().item()
    rej_u = (u["margin"] > 0).float().mean().item()
    print(f"parameter-free rule ξ_c > ψ_c ∀c: rejects {rej_u:.1%} of unknown, "
          f"{rej_k:.1%} of known   (ψ mean {psi.mean():.4f})")

    # max cos prints as -1.0000 for both sides at 4 dp. If it really is saturated, every
    # image sits on top of an anchor direction and there is no room left for a novelty
    # score — so look at it with enough digits to tell saturation from a tie.
    print("\nmax_c cos(x, a_c), quantiles:")
    for lbl, d in (("known", k), ("unknown", u)):
        q = torch.quantile(-d["neg_cos"], torch.tensor([0.01, 0.5, 0.99]))
        print(f"  {lbl:8s} p01={q[0]:.6f}  median={q[1]:.6f}  p99={q[2]:.6f}")

    # THE control on whether dalle3 is a fair unknown at all: if it lands overwhelmingly
    # on one known class, it is a near-duplicate of that generator, not a novel one, and
    # confident predictions are the correct behaviour rather than an OSR failure.
    print("\nwhere argmin ξ sends them (top 5):")
    for lbl, d in (("known", k), ("unknown", u)):
        h = torch.bincount(d["pred"].long(), minlength=len(names))
        top = h.argsort(descending=True)[:5]
        print(f"  {lbl:8s} " + "  ".join(
            f"{names[i]}={h[i] / len(d['pred']):.1%}" for i in top if h[i] > 0))


if __name__ == "__main__":
    main()
