"""
EUCLIDEAN-baseline attribution evaluation (image-only).

Counterpart to tests.eval_attribution for the Euclidean ablation. For each image
it projects to the unit sphere and picks the anchor with the highest COSINE
SIMILARITY (instead of the smallest entailment-cone exterior angle). All metrics
— overall / per-class / balanced accuracy, confusion matrix, precision/recall/F1,
per-semantic accuracy — are reported in the same format as the hyperbolic eval so
the two can be compared line-by-line.

Usage:
    python -m tests.eval_attribution_euclidean \\
        --checkpoint   $WORK/checkpoints/attribution_euclidean_vitl14.pt \\
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

from models.euclidean_attribution_clip import EuclideanAttributionCLIP
from data.iab_clip_dataset import IABCLIPDataset


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
    embed_dim = ckpt.get("embed_dim", 128)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  clip_name={clip_name}  embed_dim={embed_dim}  geometry={ckpt.get('geometry', 'euclidean')}")
    print(f"  trained val_balanced={100*ckpt.get('val_balanced', 0):.1f}%  epoch={ckpt.get('epoch')}")
    print(f"  classes:")
    for i, (c, t) in enumerate(zip(class_names, anchor_texts)):
        print(f"    [{i}] {c:8s} → \"{t}\"")

    model = EuclideanAttributionCLIP(
        clip_name=clip_name,
        lora_r=ckpt.get("lora_r", 8),
        lora_alpha=ckpt.get("lora_alpha", 16),
        embed_dim=embed_dim,
    ).to(device)
    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection"])
    if "logit_scale" in ckpt:
        with torch.no_grad():
            model.logit_scale.copy_(ckpt["logit_scale"].to(device))
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
        include_uncaptioned=True,   # eval is image-only — caption not needed
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ── Encode anchors ────────────────────────────────────────────────────────
    tok = tokenizer(anchor_texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=77)
    x_anc = model.encode_text(tok["input_ids"].to(device),
                              tok["attention_mask"].to(device))         # (K, D)

    # ── Embed images & classify by max cosine similarity ──────────────────────
    all_pred, all_gt, all_sem = [], [], []
    all_cos_to_anchors = []
    for batch in tqdm(loader, desc=f"eval ({args.split})"):
        pixel = batch["pixel_values"].to(device)
        x_img = model.encode_image(pixel)                               # (B, D)
        cos = (x_img @ x_anc.t()).cpu()                                 # (B, K)
        all_cos_to_anchors.append(cos)

        pred_idx = cos.argmax(dim=1)
        all_pred.extend(class_names[i] for i in pred_idx.tolist())
        all_gt.extend(batch["generator"])
        all_sem.extend(batch["semantic"])

    cos_all = torch.cat(all_cos_to_anchors, dim=0)   # (N, K)

    # ── Metrics ───────────────────────────────────────────────────────────────
    total = len(all_gt)
    correct = sum(p == g for p, g in zip(all_pred, all_gt))
    print(f"\n=== Image-only cosine classification ({args.split} split) ===")
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

    # ── Confusion matrix + precision / recall / F1 (per class) ───────────────
    print("\n--- Confusion matrix (rows = ground truth, cols = prediction) ---")
    K = len(class_names)
    cmat = [[0] * K for _ in range(K)]
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    for pred, gt in zip(all_pred, all_gt):
        cmat[name_to_idx[gt]][name_to_idx[pred]] += 1
    header = "             " + "  ".join(f"{c:>8s}" for c in class_names)
    print(header)
    for i, c in enumerate(class_names):
        row = f"  gt={c:8s} " + "  ".join(f"{cmat[i][j]:>8d}" for j in range(K))
        print(row)

    print("\n--- Precision / Recall / F1 per class ---")
    p_list, r_list, f_list = [], [], []
    for c in class_names:
        i = name_to_idx[c]
        tp = cmat[i][i]
        fp = sum(cmat[r][i] for r in range(K) if r != i)   # predicted c but gt was something else
        fn = sum(cmat[i][r] for r in range(K) if r != i)   # gt was c but predicted something else
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        p_list.append(precision); r_list.append(recall); f_list.append(f1)
        print(f"  {c:10s}: P={100*precision:5.1f}%  R={100*recall:5.1f}%  F1={100*f1:5.1f}%  "
              f"(TP={tp} FP={fp} FN={fn})")
    print(f"  {'macro avg':10s}: P={100*sum(p_list)/K:5.1f}%  R={100*sum(r_list)/K:5.1f}%  F1={100*sum(f_list)/K:5.1f}%")

    print("\n--- Per-semantic accuracy ---")
    by_sem: dict[str, list[bool]] = {}
    for pred, gt, sem in zip(all_pred, all_gt, all_sem):
        by_sem.setdefault(sem, []).append(pred == gt)
    for sem, hits in sorted(by_sem.items()):
        print(f"  {sem:15s}: {100*sum(hits)/len(hits):5.1f}%  ({sum(hits)}/{len(hits)})")

    print("\n--- Mean cosine similarity to each anchor (higher = closer) ---")
    for j, c_anc in enumerate(class_names):
        for c_gt in class_names:
            mask = torch.tensor([g == c_gt for g in all_gt])
            if mask.any():
                m = cos_all[mask, j].mean().item()
                print(f"  {c_gt:8s} images → anchor[{c_anc:8s}]:  cos̄={m:.3f}")


if __name__ == "__main__":
    main()
