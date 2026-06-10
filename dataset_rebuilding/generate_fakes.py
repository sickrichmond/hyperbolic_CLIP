"""
Regenerate FAKE images with a chosen diffusion model from the new dense captions.

Generalises generate_flux_fakes.py to several generator families so we can
rebuild the IAB diffusion-only fake set (real SD3 SD3_5 SDXL FLUX) from richer
prompts and measure how attribution performance drops vs the original IAB fakes.

Each real image's new caption (dataset_rebuilding/check_captions.py → *_clean) is
fed to the generator; the fake is saved NON-destructively, named after the REAL
image's stem so real↔fake pair by stem, under the IAB-style layout:

    <out_root>/<GENERATOR>/COCO/000000.png
    <out_root>/<GENERATOR>/AnimalFace/cat/flickr_cat_000003.png
    ...

Resumable: existing PNGs are skipped.

Example (run one generator per GPU/job — see slurm_gen.sh):
    python dataset_rebuilding/generate_fakes.py --generator SD3 \\
        --captions_dir $WORK/iab_captions_detailed_clean \\
        --dataset_path $WORK/iab_dataset \\
        --out_root     $WORK/iab_recap_dataset \\
        --max_per_class 100
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# ── Generator registry: label → pipeline kind + HF id + default params ────────
# pipeline kind drives which diffusers class + which call kwargs are valid.
GENERATORS = {
    "FLUX":  dict(kind="flux", model="black-forest-labs/FLUX.1-schnell",
                  steps=4,  guidance=0.0, max_seq_len=256),   # gated
    "SD3":   dict(kind="sd3",  model="stabilityai/stable-diffusion-3-medium-diffusers",
                  steps=28, guidance=7.0, max_seq_len=256),   # gated
    "SD3_5": dict(kind="sd3",  model="stabilityai/stable-diffusion-3.5-large",
                  steps=28, guidance=3.5, max_seq_len=256),   # gated
    "SDXL":  dict(kind="sdxl", model="stabilityai/stable-diffusion-xl-base-1.0",
                  steps=30, guidance=5.0, max_seq_len=None),  # ungated; CLIP 77-tok only
}

# IAB layout (mirrors data/iab_clip_dataset.py).
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
    p.add_argument("--generator", required=True, choices=list(GENERATORS),
                   help="Which generator family to use (sets pipeline + defaults).")
    p.add_argument("--captions_dir", required=True,
                   help="Dir with the new caption CSVs (use the *_clean dir).")
    p.add_argument("--dataset_path", required=True,
                   help="IAB root (contains real/), to enumerate real stems.")
    p.add_argument("--out_root", required=True,
                   help="Output root; fakes go to <out_root>/<generator>/<semantic>/.")
    p.add_argument("--model", default=None, help="Override the HF model id.")
    p.add_argument("--semantics", nargs="+", default=ALL_SEMANTICS,
                   choices=ALL_SEMANTICS)
    # These default to the per-generator registry values when left unset.
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--max_seq_len", type=int, default=None,
                   help="T5 max sequence length (flux/sd3 only; ignored for sdxl).")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Cap fakes per class (pilot).")
    p.add_argument("--limit", type=int, default=None,
                   help="Global cap on images generated this run.")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--cpu_offload", action="store_true",
                   help="Enable model CPU offload if VRAM is tight (slower).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _img_dir(root: Path, semantic: str) -> Path:
    sup, leaf = SEMANTIC_DIR[semantic]
    return (root / "real" / leaf) if sup is None else (root / "real" / sup / leaf)


def _out_dir(out_root: Path, generator: str, semantic: str) -> Path:
    sup, leaf = SEMANTIC_DIR[semantic]
    base = out_root / generator
    return (base / leaf) if sup is None else (base / sup / leaf)


def _load_caps_by_stem(csv_path: Path) -> dict[str, str]:
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
    h = 0
    for ch in stem:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (base + h) & 0x7FFFFFFF


def build_pipeline(kind: str, model: str, cpu_offload: bool):
    from diffusers import (FluxPipeline, StableDiffusion3Pipeline,
                           StableDiffusionXLPipeline)
    cls = {"flux": FluxPipeline, "sd3": StableDiffusion3Pipeline,
           "sdxl": StableDiffusionXLPipeline}[kind]
    print(f"Loading {model} ({kind}, bf16)…")
    pipe = cls.from_pretrained(model, torch_dtype=torch.bfloat16)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — this must run on a GPU node.")

    cfg = GENERATORS[args.generator]
    kind = cfg["kind"]
    model = args.model or cfg["model"]
    steps = args.steps if args.steps is not None else cfg["steps"]
    guidance = args.guidance if args.guidance is not None else cfg["guidance"]
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else cfg["max_seq_len"]

    caps_dir = Path(args.captions_dir)
    root = Path(args.dataset_path)
    out_root = Path(args.out_root)
    print(f"Generator={args.generator} model={model} steps={steps} "
          f"guidance={guidance} max_seq_len={max_seq_len} {args.width}x{args.height}")

    pipe = build_pipeline(kind, model, args.cpu_offload)

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
        out_dir = _out_dir(out_root, args.generator, sem)
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = sorted(p for p in img_dir.iterdir() if p.suffix in IMAGE_EXTS)
        if args.max_per_class:
            imgs = imgs[:args.max_per_class]

        n_done = n_skip = n_nocap = 0
        for img in tqdm(imgs, desc=f"{args.generator}:{sem}"):
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
            kwargs = dict(
                height=args.height, width=args.width,
                num_inference_steps=steps, guidance_scale=guidance,
                generator=torch.Generator("cuda").manual_seed(
                    _stable_seed(args.base_seed, img.stem)),
            )
            if kind in ("flux", "sd3") and max_seq_len:
                kwargs["max_sequence_length"] = max_seq_len
            image = pipe(cap, **kwargs).images[0]
            image.save(out)
            n_done += 1
            if remaining is not None:
                remaining -= 1

        rate = n_done / max(time.time() - t0, 1e-6)
        print(f"[{sem}] generated {n_done}, skipped {n_skip} (exist), "
              f"{n_nocap} without caption  | ~{rate:.2f} img/s cumulative")
        total_done += n_done; total_skip += n_skip; total_nocap += n_nocap

    dt = time.time() - t0
    print(f"\nDone {args.generator}. Generated {total_done} in {dt/60:.1f} min "
          f"({total_done/max(dt,1e-6):.2f} img/s); skipped {total_skip}; "
          f"{total_nocap} reals had no caption.")
    print(f"Output under: {out_root}/{args.generator}/")


if __name__ == "__main__":
    main()
