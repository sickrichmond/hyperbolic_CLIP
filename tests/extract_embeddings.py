"""
Dump image embeddings of a checkpoint to a .npz file.

Runs the val split through the image encoder (image-only — no captions needed
at inference) and writes:
  - lorentz:     (N, D)  image embeddings, space components on the hyperboloid
  - anchors:     (K, D)  class-anchor embeddings (same Lorentz space)
  - labels:      (N,)    int class labels in [0, K)
  - class_names: list[str]  e.g. ["real", "FLUX"]
  - generators:  list[str]  per-sample generator string
  - semantics:   list[str]  per-sample semantic class string

The downstream HoroPCA + UMAP visualisation works on these.

Usage:
    python -m tests.extract_embeddings \\
        --checkpoint   $WORK/checkpoints/attribution_FLUX_vitl14_hier.pt \\
        --dataset_path $WORK/iab_dataset \\
        --captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --output       $WORK/embeddings/val_hier.npz
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from models.attribution_clip import AttributionCLIP
from data.iab_clip_dataset import IABCLIPDataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset_path",  required=True)
    p.add_argument("--captions_dir",  required=True)
    p.add_argument("--output",        required=True,
                   help=".npz file to write")
    p.add_argument("--generators",    nargs="+", default=["real", "FLUX"])
    p.add_argument("--semantics",     nargs="+",
                   default=["COCO", "cat", "dog", "wild", "FFHQ", "celebahq",
                             "bedroom", "church", "classroom", "ImageNet-1k"])
    p.add_argument("--split",         choices=["train", "val", "all"], default="val")
    p.add_argument("--val_frac",      type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max_per_class", type=int,   default=None)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--num_workers",   type=int,   default=4)
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

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  classes: {class_names}")
    print(f"  curv:    {curv}  hyperbolic_dim={ckpt.get('hyperbolic_dim')}")

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
        include_uncaptioned=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Anchors
    tok = tokenizer(anchor_texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=77)
    x_anc, _ = model.encode_text(tok["input_ids"].to(device),
                                 tok["attention_mask"].to(device))
    anchors = x_anc.detach().cpu().numpy()                # (K, D)

    # Image embeddings
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    all_emb, all_lbl, all_gen, all_sem = [], [], [], []
    for batch in tqdm(loader, desc=f"embedding ({args.split})"):
        pixel = batch["pixel_values"].to(device)
        x_img, _ = model.encode_image(pixel)
        all_emb.append(x_img.detach().cpu().numpy())
        all_lbl.extend(name_to_idx[g] for g in batch["generator"])
        all_gen.extend(batch["generator"])
        all_sem.extend(batch["semantic"])

    embeddings = np.concatenate(all_emb, axis=0)          # (N, D)
    labels = np.array(all_lbl, dtype=np.int64)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        lorentz=embeddings,
        anchors=anchors,
        labels=labels,
        class_names=np.array(class_names),
        anchor_texts=np.array(anchor_texts),
        generators=np.array(all_gen),
        semantics=np.array(all_sem),
        curv=np.array([curv]),
    )
    print(f"\nSaved {len(embeddings)} embeddings ({embeddings.shape[1]}D Lorentz) "
          f"+ {len(anchors)} anchors → {out_path}")


if __name__ == "__main__":
    main()
