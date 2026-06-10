"""
Regenerate FAKE images with a chosen diffusion model from the new dense captions.

Two naming modes:

  --naming iab   (default): mirror the ORIGINAL IAB fake set 1:1 — iterate the
      rows N of the fake-source caption CSV and emit `{prefix}_p{N}_i{K}.png`
      (K = 0..variants-1, IAB uses 2), using OUR detailed caption for the real
      image at row N. Same generators, same filenames, same structure as IAB;
      the ONLY difference is the richer prompt. This is what you want for an
      apples-to-apples attribution comparison.

  --naming stem : one fake per REAL image, named `<stem>.png` (the round-1 form).

Defaults (model / steps / guidance / dtype / resolution) MATCH the IAB paper
(arXiv 2605.12967) per generator, so nothing varies except the prompt.

Output (non-destructive — choose a fresh --out_root for each round), IAB layout:

    <out_root>/<GEN>/COCO/COCO-new_p0_i0.png
    <out_root>/<GEN>/AnimalFace/cat/AnimalFace_cat_p0_i0.png
    ...

Resumable: existing PNGs are skipped.

Example (full set, IAB naming — see slurm_gen.sh):
    python dataset_rebuilding/generate_fakes.py --generator SD3 \\
        --captions_dir      $WORK/iab_captions_detailed_clean \\
        --fake_src_captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --out_root          $WORK/iab_recap_dataset_v2
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# ── Generator registry — defaults MATCH ImageAttributionBench (arXiv 2605.12967):
# SD3-medium, SD3.5-MEDIUM, SDXL-base-1.0 at 1024²; FLUX.1-schnell at 512².
# (Sources: dataset_construction/t2i_generator/diffuser_models/SDModel.py; paper.)
GENERATORS = {
    "FLUX":  dict(kind="flux", model="black-forest-labs/FLUX.1-schnell",
                  dtype="bf16", steps=4,  guidance=0.0, max_seq_len=256,
                  height=512,  width=512),
    "SD3":   dict(kind="sd3",  model="stabilityai/stable-diffusion-3-medium-diffusers",
                  dtype="fp16", steps=28, guidance=7.0, max_seq_len=256,
                  height=1024, width=1024),
    "SD3_5": dict(kind="sd3",  model="stabilityai/stable-diffusion-3.5-medium",
                  dtype="bf16", steps=40, guidance=4.5, max_seq_len=256,
                  height=1024, width=1024),
    "SDXL":  dict(kind="sdxl", model="stabilityai/stable-diffusion-xl-base-1.0",
                  dtype="fp16", steps=30, guidance=5.0, max_seq_len=None,
                  height=1024, width=1024),
}

# IAB layout.
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
# detailed-caption CSV (keyed by stem) per semantic.
DETAILED_CSV = {
    "COCO": "COCO.csv", "cat": "AnimalFace_cat.csv", "dog": "AnimalFace_dog.csv",
    "wild": "AnimalFace_wild.csv", "FFHQ": "HumanFace_FFHQ.csv",
    "celebahq": "HumanFace_celebahq.csv", "bedroom": "Scene_LSUN_bedroom.csv",
    "church": "Scene_LSUN_church.csv", "classroom": "Scene_LSUN_classroom.csv",
    "ImageNet-1k": "imagenet-1k.csv",
}
# IAB naming: semantic → (fake filename prefix, fake-source CSV used to index rows N).
FAKE_SRC = {
    "COCO":        ("COCO-new",             "COCO-new.csv"),
    "cat":         ("AnimalFace_cat",        "AnimalFace_cat.csv"),
    "dog":         ("AnimalFace_dog",        "AnimalFace_dog.csv"),
    "wild":        ("AnimalFace_wild",       "AnimalFace_wild.csv"),
    "FFHQ":        ("HumanFace_FFHQ",        "HumanFace_FFHQ.csv"),
    "celebahq":    ("HumanFace_celebahq",    "HumanFace_celebahq.csv"),
    "bedroom":     ("Scene_LSUN_bedroom",    "Scene_LSUN_bedroom.csv"),
    "church":      ("Scene_LSUN_church",     "Scene_LSUN_church.csv"),
    "classroom":   ("Scene_LSUN_classroom",  "Scene_LSUN_classroom.csv"),
    "ImageNet-1k": ("imagenet-1k-new",       "imagenet-1k-new.csv"),
}
ALL_SEMANTICS = list(SEMANTIC_DIR.keys())
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generator", required=True, choices=list(GENERATORS))
    p.add_argument("--captions_dir", required=True,
                   help="Detailed caption CSVs, keyed by stem (the *_clean dir).")
    p.add_argument("--out_root", required=True,
                   help="Output root; fakes go to <out_root>/<generator>/<semantic>/.")
    p.add_argument("--naming", choices=["iab", "stem"], default="iab",
                   help="iab: {prefix}_p{N}_i{K}.png (mirror IAB); stem: <stem>.png.")
    p.add_argument("--variants", type=int, default=2,
                   help="Images per prompt for --naming iab (IAB uses 2 → i0,i1).")
    p.add_argument("--fake_src_captions_dir", default=None,
                   help="Dir with the original IAB CSVs (COCO-new.csv, ...) used to "
                        "map row N → real stem. Required for --naming iab.")
    p.add_argument("--dataset_path", default=None,
                   help="IAB root (real/...). Required for --naming stem.")
    p.add_argument("--model", default=None, help="Override the HF model id.")
    p.add_argument("--dtype", default=None, choices=["fp16", "bf16"])
    p.add_argument("--semantics", nargs="+", default=ALL_SEMANTICS,
                   choices=ALL_SEMANTICS)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Cap prompts (rows/stems) per class — for a quick subset.")
    p.add_argument("--limit", type=int, default=None,
                   help="Global cap on images generated this run.")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--cpu_offload", action="store_true")
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


def _row_stems(csv_path: Path) -> list[str]:
    """Real-image stem per row index N (the fake `_p{N}` indexes into this CSV)."""
    stems: list[str] = []
    if not csv_path.exists():
        return stems
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)                       # header
        for row in reader:
            stems.append(Path(row[0]).stem if row else "")
    return stems


def _stable_seed(base: int, key: str) -> int:
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (base + h) & 0x7FFFFFFF


def build_pipeline(kind: str, model: str, dtype: str, cpu_offload: bool):
    from diffusers import (FluxPipeline, StableDiffusion3Pipeline,
                           StableDiffusionXLPipeline)
    cls = {"flux": FluxPipeline, "sd3": StableDiffusion3Pipeline,
           "sdxl": StableDiffusionXLPipeline}[kind]
    td = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    print(f"Loading {model} ({kind}, {dtype})…")
    pipe = cls.from_pretrained(model, torch_dtype=td)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _tasks_for_semantic(args, semantic: str):
    """Yield (out_path, caption, seed_key) for one semantic, per naming mode.
    Returns (tasks, n_nocap)."""
    out_dir = _out_dir(Path(args.out_root), args.generator, semantic)
    detailed = _load_caps_by_stem(Path(args.captions_dir) / DETAILED_CSV[semantic])
    tasks, n_nocap = [], 0

    if args.naming == "iab":
        prefix, src_csv = FAKE_SRC[semantic]
        stems = _row_stems(Path(args.fake_src_captions_dir) / src_csv)
        if args.max_per_class:
            stems = stems[:args.max_per_class]
        for n, stem in enumerate(stems):
            cap = detailed.get(stem, "")
            if not cap:
                n_nocap += 1
                continue
            for k in range(args.variants):
                tasks.append((out_dir / f"{prefix}_p{n}_i{k}.png", cap, f"{stem}:{k}"))
    else:  # stem
        img_dir = _img_dir(Path(args.dataset_path), semantic)
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix in IMAGE_EXTS) \
            if img_dir.exists() else []
        if args.max_per_class:
            imgs = imgs[:args.max_per_class]
        for img in imgs:
            cap = detailed.get(img.stem, "")
            if not cap:
                n_nocap += 1
                continue
            tasks.append((out_dir / f"{img.stem}.png", cap, img.stem))
    return out_dir, tasks, n_nocap


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — this must run on a GPU node.")
    if args.naming == "iab" and not args.fake_src_captions_dir:
        sys.exit("--naming iab requires --fake_src_captions_dir (the original IAB CSVs).")
    if args.naming == "stem" and not args.dataset_path:
        sys.exit("--naming stem requires --dataset_path.")

    cfg = GENERATORS[args.generator]
    kind = cfg["kind"]
    model = args.model or cfg["model"]
    steps = args.steps if args.steps is not None else cfg["steps"]
    guidance = args.guidance if args.guidance is not None else cfg["guidance"]
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else cfg["max_seq_len"]
    dtype = args.dtype or cfg["dtype"]
    height = args.height if args.height is not None else cfg["height"]
    width = args.width if args.width is not None else cfg["width"]
    print(f"Generator={args.generator} naming={args.naming} model={model} dtype={dtype} "
          f"steps={steps} guidance={guidance} max_seq_len={max_seq_len} {width}x{height}")

    pipe = build_pipeline(kind, model, dtype, args.cpu_offload)

    total_done = total_skip = total_nocap = 0
    remaining = args.limit
    t0 = time.time()
    for sem in args.semantics:
        if remaining is not None and remaining <= 0:
            break
        out_dir, tasks, n_nocap = _tasks_for_semantic(args, sem)
        total_nocap += n_nocap
        if not tasks:
            print(f"[{sem}] no tasks (nocap={n_nocap}) — skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        n_done = n_skip = 0
        for out, cap, seed_key in tqdm(tasks, desc=f"{args.generator}:{sem}"):
            if remaining is not None and remaining <= 0:
                break
            if out.exists() and not args.overwrite:
                n_skip += 1
                continue
            gen = torch.Generator("cuda").manual_seed(_stable_seed(args.base_seed, seed_key))
            kwargs = dict(height=height, width=width, num_inference_steps=steps,
                          guidance_scale=guidance, generator=gen)
            if kind in ("flux", "sd3") and max_seq_len:
                kwargs["max_sequence_length"] = max_seq_len
            pipe(cap, **kwargs).images[0].save(out)
            n_done += 1
            if remaining is not None:
                remaining -= 1

        rate = n_done / max(time.time() - t0, 1e-6)
        print(f"[{sem}] generated {n_done}, skipped {n_skip} (exist), "
              f"{n_nocap} prompts w/o caption | ~{rate:.2f} img/s cumulative")
        total_done += n_done; total_skip += n_skip

    dt = time.time() - t0
    print(f"\nDone {args.generator}. Generated {total_done} in {dt/60:.1f} min "
          f"({total_done/max(dt,1e-6):.2f} img/s); skipped {total_skip}; "
          f"{total_nocap} prompts had no caption.")
    print(f"Output under: {args.out_root}/{args.generator}/")


if __name__ == "__main__":
    main()
