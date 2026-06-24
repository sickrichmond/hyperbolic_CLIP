"""
Hierarchical (generator + family) attribution with novelty detection.

For each image we compute the exterior angle ξ to every generator anchor and to
every family anchor, then apply a two-level geometric back-off with a tunable
abstention margin τ:

    g* = argmin_k ξ_gen[k]
    if ξ_gen[g*] < ψ_gen[g*] - τ:        → name the generator  ("it is SD3_5")
    elif ξ_fam[f*] < ψ_fam[f*]:           → name the family     ("looks like a SD")
                                            (f* = argmin_m ξ_fam[m])
    else:                                 → "novel / unknown"

The intended use is the leave-one-out generalisation test: train with a
generator held out (e.g. --holdout_generators SD3_5 in train_attribution.py),
then evaluate INCLUDING that generator here. A held-out generator that the model
has never seen should not sit confidently inside any seen generator cone, so it
abstains and is routed to its family cone — measured by `family_routing`.

Usage:
    python -m tests.eval_attribution_hierarchical \\
        --checkpoint   $WORK/hyp_fine_tuning/checkpoints/attribution_holdout_sd.pt \\
        --dataset_path $WORK/hyp_fine_tuning/iab_dataset \\
        --captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --generators   real SD1_5 SD2_1 SD3 SD3_5 SDXL FLUX \\
        --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \\
        --split        val --tau 0.05
"""
import argparse
import warnings
from collections import Counter

import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from models.attribution_clip import AttributionCLIP
from data.iab_clip_dataset import IABCLIPDataset
from data.families import family_of, build_family_structures
from geometry.lorentz import half_aperture, oxy_angle


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset_path",  required=True)
    p.add_argument("--captions_dir",  required=True)
    p.add_argument("--generators",    nargs="+", required=True,
                   help="Generators to LOAD for eval. Include any held-out "
                        "generator here so it can be scored for novelty.")
    p.add_argument("--semantics",     nargs="+",
                   default=["COCO", "cat", "dog", "wild", "FFHQ", "celebahq",
                             "bedroom", "church", "classroom", "ImageNet-1k"])
    p.add_argument("--split",         choices=["train", "val", "all"], default="val")
    p.add_argument("--val_frac",      type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max_per_class", type=int,   default=None)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--tau",           type=float, default=0.0,
                   help="Abstention margin (radians): name the generator only "
                        "if ξ_gen < ψ_gen - τ. Larger τ ⇒ more abstention.")
    p.add_argument("--tau_sweep",     nargs="+", type=float,
                   default=[0.0, 0.02, 0.05, 0.1, 0.15, 0.2],
                   help="τ values for the held-out routing trade-off table.")
    return p.parse_args()


def _pairwise_xi(anchors: torch.Tensor, x_img: torch.Tensor, curv: float) -> torch.Tensor:
    """xi[b, k] = oxy_angle(anchor_k, img_b). anchors (K,D), x_img (B,D) → (B, K)."""
    B, K = x_img.shape[0], anchors.shape[0]
    a = anchors.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1)
    i = x_img.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    return oxy_angle(a, i, curv=curv).reshape(B, K)


def decide(xi_gen_row, xi_fam_row, psi_gen, psi_fam, class_names, family_names, tau):
    """Two-level geometric back-off for a single image.

    Returns (level, name) where level ∈ {"gen", "fam", "novel"}.
    """
    g = int(torch.argmin(xi_gen_row))
    if xi_gen_row[g] < psi_gen[g] - tau:
        return "gen", class_names[g]
    f = int(torch.argmin(xi_fam_row))
    if xi_fam_row[f] < psi_fam[f]:
        return "fam", family_names[f]
    return "novel", None


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    clip_name = ckpt["clip_name"]
    class_names: list[str] = ckpt["class_names"]          # seen generators
    anchor_texts: list[str] = ckpt["anchor_texts"]
    curv = ckpt.get("curv", 1.0)
    min_radius = ckpt.get("min_radius", 0.1)

    # Family structures: prefer the checkpoint's (trained) ones; fall back to
    # rebuilding them from the seen generators for older flat checkpoints.
    family_names = ckpt.get("family_names")
    family_anchor_texts = ckpt.get("family_anchor_texts")
    if family_names is None or family_anchor_texts is None:
        family_names, family_anchor_texts, _ = build_family_structures(class_names)
        print("NOTE: checkpoint has no family fields — rebuilt from class_names "
              "(family cones were NOT trained).")

    holdout = ckpt.get("holdout_generators", []) or []

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  clip_name={clip_name}  hyperbolic_dim={ckpt.get('hyperbolic_dim')}  curv={curv}")
    print(f"  trained val_balanced={100*ckpt.get('val_balanced', 0):.1f}%  "
          f"val_family={100*ckpt.get('val_family_acc', 0):.1f}%  epoch={ckpt.get('epoch')}")
    print(f"  seen generators ({len(class_names)}): {class_names}")
    print(f"  families ({len(family_names)}): {family_names}")
    print(f"  held out at training: {holdout if holdout else '(none)'}")
    print(f"  abstention τ = {args.tau}")

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
        include_uncaptioned=True,   # eval is image-only
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ── Encode generator and family anchors ─────────────────────────────────────
    def encode(texts):
        tok = tokenizer(texts, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=77)
        x, _ = model.encode_text(tok["input_ids"].to(device),
                                 tok["attention_mask"].to(device))
        return x

    x_gen = encode(anchor_texts)
    x_fam = encode(family_anchor_texts)
    psi_gen = half_aperture(x_gen, curv=curv, min_radius=min_radius).cpu()  # (K,)
    psi_fam = half_aperture(x_fam, curv=curv, min_radius=min_radius).cpu()  # (F,)

    # ── Embed images, collect ξ to both levels ──────────────────────────────────
    all_xi_gen, all_xi_fam, all_gt = [], [], []
    for batch in tqdm(loader, desc=f"eval ({args.split})"):
        pixel = batch["pixel_values"].to(device)
        x_img, _ = model.encode_image(pixel)
        all_xi_gen.append(_pairwise_xi(x_gen, x_img, curv).cpu())
        all_xi_fam.append(_pairwise_xi(x_fam, x_img, curv).cpu())
        all_gt.extend(batch["generator"])
    xi_gen = torch.cat(all_xi_gen, dim=0)   # (N, K)
    xi_fam = torch.cat(all_xi_fam, dim=0)   # (N, F)

    seen = set(class_names)
    heldout_present = sorted({g for g in all_gt if g not in seen})

    print("\n--- Family half-apertures ψ ---")
    for f, p_v in zip(family_names, psi_fam.tolist()):
        print(f"    family[{f:18s}]: ψ={p_v:.3f}")

    # ── Decisions at the chosen τ ────────────────────────────────────────────────
    decisions = [
        decide(xi_gen[i], xi_fam[i], psi_gen, psi_fam, class_names, family_names, args.tau)
        for i in range(len(all_gt))
    ]

    # ── Seen-generator metrics ───────────────────────────────────────────────────
    print(f"\n=== SEEN generators @ τ={args.tau} ({args.split} split) ===")
    per_gen = {c: {"n": 0, "gen_hit": 0, "fam_hit": 0, "abstain": 0} for c in class_names}
    for (lvl, name), gt in zip(decisions, all_gt):
        if gt not in seen:
            continue
        s = per_gen[gt]
        s["n"] += 1
        pred_fam = name if lvl == "fam" else (family_of(name) if lvl == "gen" else None)
        s["gen_hit"] += int(lvl == "gen" and name == gt)
        s["fam_hit"] += int(pred_fam == family_of(gt))
        s["abstain"] += int(lvl != "gen")
    gen_accs, fam_accs = [], []
    for c in class_names:
        s = per_gen[c]
        if s["n"] == 0:
            continue
        ga, fa, ab = s["gen_hit"]/s["n"], s["fam_hit"]/s["n"], s["abstain"]/s["n"]
        gen_accs.append(ga); fam_accs.append(fa)
        print(f"  {c:14s}: gen={100*ga:5.1f}%  family={100*fa:5.1f}%  "
              f"abstain={100*ab:5.1f}%  (n={s['n']})")
    if gen_accs:
        print(f"  {'balanced':14s}: gen={100*sum(gen_accs)/len(gen_accs):5.1f}%  "
              f"family={100*sum(fam_accs)/len(fam_accs):5.1f}%")

    # ── Held-out novelty metrics ─────────────────────────────────────────────────
    if heldout_present:
        print(f"\n=== HELD-OUT generators (novelty) @ τ={args.tau} ===")
        print("  family_route = abstained to generator level AND routed to the correct family")
        for g in heldout_present:
            idx = [i for i, gt in enumerate(all_gt) if gt == g]
            true_fam = family_of(g)
            n = len(idx)
            abstain = sum(decisions[i][0] != "gen" for i in idx)
            fam_correct = sum(
                (decisions[i][1] if decisions[i][0] == "fam"
                 else (family_of(decisions[i][1]) if decisions[i][0] == "gen" else None)) == true_fam
                for i in idx
            )
            fam_route = sum(decisions[i][0] == "fam" and decisions[i][1] == true_fam for i in idx)
            outcomes = Counter(
                f"{lvl}:{decisions[i][1]}" if (lvl := decisions[i][0]) != "novel" else "novel"
                for i in idx
            )
            print(f"  {g:14s} (true family={true_fam}, n={n}):")
            print(f"      abstain={100*abstain/n:5.1f}%  "
                  f"family_correct={100*fam_correct/n:5.1f}%  "
                  f"family_route={100*fam_route/n:5.1f}%")
            top = ", ".join(f"{k}={v}" for k, v in outcomes.most_common(5))
            print(f"      routed to: {top}")

        # τ sweep — naming-vs-abstain trade-off over the held-out set as a whole
        print("\n--- Held-out routing vs τ ---")
        print(f"  {'τ':>6s}  {'abstain%':>9s}  {'family_correct%':>16s}  {'family_route%':>14s}")
        heldout_set = set(heldout_present)
        ho_idx = [i for i, gt in enumerate(all_gt) if gt in heldout_set]
        for tau in args.tau_sweep:
            decs = [decide(xi_gen[i], xi_fam[i], psi_gen, psi_fam,
                           class_names, family_names, tau) for i in ho_idx]
            n = len(ho_idx)
            if n == 0:
                break
            abstain = sum(d[0] != "gen" for d in decs)
            fam_corr = sum(
                (d[1] if d[0] == "fam" else (family_of(d[1]) if d[0] == "gen" else None))
                == family_of(all_gt[i])
                for d, i in zip(decs, ho_idx)
            )
            fam_route = sum(d[0] == "fam" and d[1] == family_of(all_gt[i])
                            for d, i in zip(decs, ho_idx))
            print(f"  {tau:6.2f}  {100*abstain/n:9.1f}  {100*fam_corr/n:16.1f}  "
                  f"{100*fam_route/n:14.1f}")
    else:
        print("\n(no held-out generators present in this eval set — "
              "pass them via --generators to measure novelty)")


if __name__ == "__main__":
    main()
