"""
Image-only attribution evaluation with entailment cones.

For each image, projects to Lorentz space and picks the anchor cone with the
smallest exterior angle (i.e., the cone the image sits most deeply inside).
No captions are used at inference — matches the in-the-wild scenario.

Usage:
    python -m tests.eval_attribution \\
        --checkpoint  $WORK/checkpoints/attribution_FLUX_vitl14.pt \\
        --dataset_path $WORK/iab_dataset \\
        --captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --generators   real FLUX \\
        --semantics    COCO cat dog wild FFHQ celebahq bedroom church classroom ImageNet-1k \\
        --split        val
"""
import argparse
import warnings

import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from tqdm import tqdm

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
    p.add_argument("--generators",    nargs="+", default=["real", "FLUX"])
    p.add_argument("--semantics",     nargs="+",
                   default=["COCO", "cat", "dog", "wild", "FFHQ", "celebahq",
                             "bedroom", "church", "classroom", "ImageNet-1k"])
    p.add_argument("--split",         choices=["train", "val", "all"], default="val",
                   help="Which split of the dataset to evaluate on.")
    p.add_argument("--val_frac",      type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--batch_size",    type=int, default=128)
    p.add_argument("--num_workers",   type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    clip_name = ckpt["clip_name"]
    class_names: list[str] = ckpt["class_names"]
    anchor_texts: list[str] = ckpt["anchor_texts"]
    curv = ckpt.get("curv", 1.0)
    min_radius = ckpt.get("min_radius", 0.1)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  clip_name={clip_name}  hyperbolic_dim={ckpt.get('hyperbolic_dim')}  curv={curv}")
    print(f"  trained val_balanced={100*ckpt.get('val_balanced', 0):.1f}%  epoch={ckpt.get('epoch')}")
    print(f"  classes:")
    for i, (c, t) in enumerate(zip(class_names, anchor_texts)):
        print(f"    [{i}] {c:8s} → \"{t}\"")

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
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ── Encode anchors ────────────────────────────────────────────────────────
    tok = tokenizer(anchor_texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=77)
    x_anc, _ = model.encode_text(tok["input_ids"].to(device),
                                 tok["attention_mask"].to(device))
    psi = half_aperture(x_anc, curv=curv, min_radius=min_radius)

    # ── Embed images & classify ───────────────────────────────────────────────
    all_pred, all_gt, all_sem = [], [], []
    all_xi_to_anchors = []
    for batch in tqdm(loader, desc=f"eval ({args.split})"):
        pixel = batch["pixel_values"].to(device)
        x_img, _ = model.encode_image(pixel)

        B, K = x_img.shape[0], x_anc.shape[0]
        x_anc_tiled = x_anc.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1)
        x_img_tiled = x_img.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        xi = oxy_angle(x_anc_tiled, x_img_tiled, curv=curv).reshape(B, K).cpu()
        all_xi_to_anchors.append(xi)

        pred_idx = xi.argmin(dim=1)
        all_pred.extend(class_names[i] for i in pred_idx.tolist())
        all_gt.extend(batch["generator"])
        all_sem.extend(batch["semantic"])

    xi_all = torch.cat(all_xi_to_anchors, dim=0)   # (N, K)
    psi_cpu = psi.cpu()

    # ── Metrics ───────────────────────────────────────────────────────────────
    total = len(all_gt)
    correct = sum(p == g for p, g in zip(all_pred, all_gt))
    print(f"\n=== Image-only cone classification ({args.split} split) ===")
    print(f"Total samples:    {total}")
    print(f"Overall accuracy: {100*correct/total:.1f}%  ({correct}/{total})")

    print("\n--- Per-class accuracy ---")
    per_class = {}
    for c in class_names:
        idx = [i for i, g in enumerate(all_gt) if g == c]
        if not idx:
            continue
        hits = sum(1 for i in idx if all_pred[i] == c)
        per_class[c] = hits / len(idx)
        print(f"  {c:10s}: {100*per_class[c]:5.1f}%  ({hits}/{len(idx)})")
    print(f"  {'balanced avg':10s}: {100 * sum(per_class.values()) / len(per_class):5.1f}%")

    print("\n--- Per-semantic accuracy ---")
    by_sem: dict[str, list[bool]] = {}
    for pred, gt, sem in zip(all_pred, all_gt, all_sem):
        by_sem.setdefault(sem, []).append(pred == gt)
    for sem, hits in sorted(by_sem.items()):
        print(f"  {sem:15s}: {100*sum(hits)/len(hits):5.1f}%  ({sum(hits)}/{len(hits)})")

    print("\n--- Mean exterior angle ξ to each anchor (lower = deeper inside cone) ---")
    print(f"  Half-aperture ψ:")
    for c, p_v in zip(class_names, psi_cpu.tolist()):
        print(f"    cone[{c:8s}]: ψ={p_v:.3f}")
    for j, c_anc in enumerate(class_names):
        for c_gt in class_names:
            mask = torch.tensor([g == c_gt for g in all_gt])
            if mask.any():
                m = xi_all[mask, j].mean().item()
                inside = (xi_all[mask, j] < psi_cpu[j]).float().mean().item()
                print(f"  {c_gt:8s} images → cone[{c_anc:8s}]:  ξ̄={m:.3f}  inside={100*inside:.1f}%")


if __name__ == "__main__":
    main()
