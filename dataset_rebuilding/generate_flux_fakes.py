"""
Pilot: regenerate FAKE images with FLUX from the new dense captions.

For every REAL image we have a new detailed caption (dataset_rebuilding/
caption_real_images.py). Here we feed that caption to FLUX and save the
generated fake, so the synthetic counterpart is now produced from a rich prompt
(harder to attribute) instead of IAB's thin original caption.

Output layout (NON-destructive — a separate root, the original iab_dataset is
untouched). Each fake is named after its REAL image's stem, so real↔fake pair
trivially by stem:

    <out_root>/FLUX/COCO/000000.png            ← from real real/COCO/000000.jpg
    <out_root>/FLUX/AnimalFace/cat/flickr_cat_000003.png
    ...

This mirrors the IAB `<generator>/<semantic>` layout, so the result can later be
pointed at by the dataset loader (or paired by stem with the reals).

Resumable: existing PNGs are skipped. Safe to relaunch after a SLURM timeout.

Example:
    python dataset_rebuilding/generate_flux_fakes.py \\
        --captions_dir $WORK/iab_captions_detailed_clean \\
        --dataset_path $WORK/iab_dataset \\
        --out_root     $WORK/iab_recap_dataset \\
        --model        black-forest-labs/FLUX.1-schnell \\
        --max_per_class 100        # pilot subset; drop to do all 2000/class
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# IAB layout (mirrors caption_real_images.py / data/iab_clip_dataset.py).
SEMANTIC_DIR = {
    "COCO":        (None,         "COCO"),
    "ImageNet-1k": (None,         "ImageNet-1k"),
    "cat":         ("AnimalFace", "cat"),
    "dog":         ("AnimalFace", "dog"),
    "wild":        ("AnimalFace", "wild"),
    "FFHQ":        ("HumanFace",  "FFHQ"),
    "celebahq":    ("HumanFace",  "celebahq"),
    "bedroom":     ("Scene",      "bedroom"),
    "church":      ("Scene",      "church"),
    "classroom":   ("Scene",      "classroom"),
}
OUTPUT_CSV = {
    "COCO":        "COCO.csv",
    "cat":         "AnimalFace_cat.csv",
    "dog":         "AnimalFace_dog.csv",
    "wild":        "AnimalFace_wild.csv",
    "FFHQ":        "HumanFace_FFHQ.csv",
    "celebahq":    "HumanFace_celebahq.csv",
    "bedroom":     "Scene_LSUN_bedroom.csv",
    "church":      "Scene_LSUN_church.csv",
    "classroom":   "Scene_LSUN_classroom.csv",
    "ImageNet-1k": "imagenet-1k.csv",
}
ALL_SEMANTICS = list(SEMANTIC_DIR.keys())
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captions_dir", required=True,
                   help="Dir with the new caption CSVs (use the *_clean dir).")
    p.add_argument("--dataset_path", required=True,
                   help="IAB root (contains real/), to enumerate real stems.")
    p.add_argument("--out_root", required=True,
                   help="Output root; fakes go to <out_root>/FLUX/<semantic>/.")
    p.add_argument("--model", default="black-forest-labs/FLUX.1-schnell",
                   help="HF id. Default FLUX.1-schnell (ungated, fast — the model "
                        "IAB used). For FLUX.1-dev set --steps 28 --guidance 3.5 "
                        "--max_seq_len 512.")
    p.add_argument("--semantics", nargs="+", default=ALL_SEMANTICS,
                   choices=ALL_SEMANTICS)
    p.add_argument("--steps", type=int, default=4,
                   help="Inference steps (schnell ~4; FLUX.1-dev ~28).")
    p.add_argument("--guidance", type=float, default=0.0,
                   help="Guidance scale (schnell 0.0; FLUX.1-dev ~3.5).")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--max_seq_len", type=int, default=256,
                   help="T5 max sequence length (schnell caps at 256; dev up to 512).")
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Cap fakes per class (pilot).")
    p.add_argument("--limit", type=int, default=None,
                   help="Global cap on images generated this run.")
    p.add_argument("--base_seed", type=int, default=0,
                   help="Per-image seed = base_seed + stable hash(stem).")
    p.add_argument("--cpu_offload", action="store_true",
                   help="Enable model CPU offload if VRAM is tight (slower).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _img_dir(root: Path, semantic: str) -> Path:
    sup, leaf = SEMANTIC_DIR[semantic]
    return (root / "real" / leaf) if sup is None else (root / "real" / sup / leaf)


def _out_dir(out_root: Path, semantic: str) -> Path:
    sup, leaf = SEMANTIC_DIR[semantic]
    return (out_root / "FLUX" / leaf) if sup is None else (out_root / "FLUX" / sup / leaf)


def _load_caps_by_stem(csv_path: Path) -> dict[str, str]:
    """stem -> caption (last column); skips header / stray header lines."""
    caps: dict[str, str] = {}
    if not csv_path.exists():
        return caps
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] == "ImgPath":
                continue
            caps[Path(row[0]).stem] = row[-1].strip()
    return caps


def _stable_seed(base: int, stem: str) -> int:
    # Deterministic across runs/processes (Python's hash() is salted).
    h = 0
    for ch in stem:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (base + h) & 0x7FFFFFFF


def build_pipeline(args):
    from diffusers import FluxPipeline
    dtype = torch.bfloat16
    print(f"Loading {args.model} (bf16)…")
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — this must run on a GPU node.")
    caps_dir = Path(args.captions_dir)
    root = Path(args.dataset_path)
    out_root = Path(args.out_root)

    pipe = build_pipeline(args)

    total_done = total_skip = total_nocap = 0
    remaining = args.limit
    t0 = time.time()
    for sem in args.semantics:
        if remaining is not None and remaining <= 0:
            break
        img_dir = _img_dir(root, sem)
        if not img_dir.exists():
            print(f"[{sem}] real dir missing, skipping: {img_dir}")
            continue
        caps = _load_caps_by_stem(caps_dir / OUTPUT_CSV[sem])
        out_dir = _out_dir(out_root, sem)
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = sorted(p for p in img_dir.iterdir() if p.suffix in IMAGE_EXTS)
        if args.max_per_class:
            imgs = imgs[:args.max_per_class]

        n_done = n_skip = n_nocap = 0
        for img in tqdm(imgs, desc=sem):
            if remaining is not None and remaining <= 0:
                break
            out = out_dir / f"{img.stem}.png"
            if out.exists() and not args.overwrite:
                n_skip += 1
                continue
            cap = caps.get(img.stem, "")
            if not cap:
                n_nocap += 1
                continue
            gen = torch.Generator("cuda").manual_seed(_stable_seed(args.base_seed, img.stem))
            image = pipe(
                cap,
                height=args.height, width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                max_sequence_length=args.max_seq_len,
                generator=gen,
            ).images[0]
            image.save(out)
            n_done += 1
            if remaining is not None:
                remaining -= 1

        rate = n_done / max(time.time() - t0, 1e-6)
        print(f"[{sem}] generated {n_done}, skipped {n_skip} (exist), "
              f"{n_nocap} without caption  | ~{rate:.2f} img/s cumulative")
        total_done += n_done; total_skip += n_skip; total_nocap += n_nocap

    dt = time.time() - t0
    print(f"\nDone. Generated {total_done} fakes in {dt/60:.1f} min "
          f"({total_done/max(dt,1e-6):.2f} img/s); skipped {total_skip}; "
          f"{total_nocap} reals had no caption.")
    print(f"Output under: {out_root}/FLUX/")


if __name__ == "__main__":
    main()
