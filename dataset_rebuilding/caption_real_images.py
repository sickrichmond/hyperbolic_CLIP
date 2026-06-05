"""
Re-caption the ImageAttributionBench **real** images with Qwen3.5-9B (vision)
served by a local Ollama server.

Why
---
The original IAB captions are thin (~20-50 words) and generic, so the fakes
generated from them are easy to spot. This script produces dense, per-class
captions (see `caption_prompts.py`) that will later be used to regenerate
harder synthetic counterparts.

Output
------
One CSV per semantic class, written to `--output_dir`, with the SAME filenames
and column schema as the original IAB caption CSVs so it is a drop-in
replacement for `IABCLIPDataset(captions_dir=...)`:

    COCO.csv                 ImgPath,Caption
    AnimalFace_cat.csv       ImgPath,Caption
    ...
    imagenet-1k.csv          ImgPath,Label,Caption   (Label carried over from the
                                                      original CSV by filename stem)

The loader keys captions by `Path(ImgPath).stem`, so ImgPath only needs a
matching stem — we write the relative path `real/<...>/<file>` for readability.

Resumability
------------
Captions are flushed to disk row-by-row. Re-running skips images whose stem is
already present in the output CSV, and images that errored out (not written) are
retried. Safe to relaunch after a SLURM timeout. Use --overwrite to start fresh.

This script talks to Ollama over HTTP and needs no GPU itself — the SLURM job
(`slurm_caption.sh`) starts `ollama serve` on the compute node's GPU first.

Example
-------
    python dataset_rebuilding/caption_real_images.py \\
        --dataset_path $WORK/iab_dataset \\
        --output_dir   $WORK/iab_captions_detailed \\
        --orig_captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --model qwen3.5:9b \\
        --num_workers 4
"""
import argparse
import base64
import csv
import io
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# Allow `python dataset_rebuilding/caption_real_images.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_prompts import PROMPT_BY_SEMANTIC  # noqa: E402

# ── IAB layout (mirrors data/iab_clip_dataset.py, kept local to avoid importing
#    torch/transformers on the Ollama node) ────────────────────────────────────
# semantic → (super_category | None, leaf_dir)
SEMANTIC_DIR: dict[str, tuple[str | None, str]] = {
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

# semantic → output CSV filename (mirrors the original IAB caption filenames so
# the new dir is a drop-in replacement).
OUTPUT_CSV: dict[str, str] = {
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

# Classes whose CSV carries an extra Label column (ImgPath,Label,Caption).
THREE_COLUMN = {"ImageNet-1k"}

ALL_SEMANTICS = list(SEMANTIC_DIR.keys())
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG"}

# Cleanup patterns for model output.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_PREAMBLE_RE = re.compile(
    r"^\s*(here(?:'s| is)?\s+(?:the\s+)?(?:final\s+)?caption\s*[:\-]?\s*|caption\s*[:\-]\s*)",
    re.IGNORECASE,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", required=True,
                   help="Root of the extracted IAB dataset (contains real/).")
    p.add_argument("--output_dir", required=True,
                   help="Directory for the new caption CSVs.")
    p.add_argument("--orig_captions_dir", default=None,
                   help="Original IAB caption dir — used only to carry over the "
                        "ImageNet Label column. Optional.")
    p.add_argument("--semantics", nargs="+", default=ALL_SEMANTICS,
                   choices=ALL_SEMANTICS)
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--ollama_host", default="127.0.0.1:11434")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Concurrent requests to Ollama. Match OLLAMA_NUM_PARALLEL.")
    p.add_argument("--max_image_side", type=int, default=1024,
                   help="Downscale longer image side to this before sending (0=off).")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--num_predict", type=int, default=256,
                   help="Max tokens for the caption.")
    p.add_argument("--request_timeout", type=int, default=300)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Cap images per class (debug).")
    p.add_argument("--limit", type=int, default=None,
                   help="Global cap on number of images this run (debug).")
    p.add_argument("--overwrite", action="store_true",
                   help="Ignore and overwrite any existing output CSVs.")
    p.add_argument("--smoke", action="store_true",
                   help="Caption a single image per requested class, print it, exit.")
    return p.parse_args()


# ── Ollama client ─────────────────────────────────────────────────────────────

def _encode_image(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _clean_caption(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _OPEN_THINK_RE.sub("", text)          # unterminated <think>
    text = _PREAMBLE_RE.sub("", text.strip())
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)             # collapse newlines/whitespace
    return text.strip()


def caption_image(host: str, model: str, prompt: str, img_path: Path,
                  *, max_side: int, temperature: float, num_predict: int,
                  timeout: int, max_retries: int) -> str:
    """Return a cleaned caption for one image. Raises on repeated failure."""
    b64 = _encode_image(img_path, max_side)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "think": False,                          # disable reasoning if supported
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": num_predict,
        },
    }
    url = f"http://{host}/api/chat"
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code == 400 and "think" in payload:
                payload.pop("think")             # older Ollama: retry without it
                r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            content = r.json()["message"]["content"]
            cap = _clean_caption(content)
            if cap:
                return cap
            last_err = RuntimeError("empty caption")
        except Exception as e:                   # noqa: BLE001 — retry any failure
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after {max_retries} retries: {last_err}")


def wait_for_ollama(host: str, model: str, timeout: int = 600) -> None:
    """Block until the Ollama server answers and the model is loadable."""
    url = f"http://{host}/api/tags"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                names = {m["name"] for m in r.json().get("models", [])}
                # Tags may appear with or without an explicit ':latest'.
                if any(n == model or n.split(":")[0] == model.split(":")[0]
                       for n in names) or not names:
                    print(f"Ollama is up at {host} (models: {sorted(names) or 'none listed'})")
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"Ollama not reachable / model '{model}' not available at {host}")


# ── Per-class processing ──────────────────────────────────────────────────────

def _img_dir(root: Path, semantic: str) -> Path:
    sup, leaf = SEMANTIC_DIR[semantic]
    return (root / "real" / leaf) if sup is None else (root / "real" / sup / leaf)


def _rel_imgpath(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _load_existing_stems(csv_path: Path) -> set[str]:
    done: set[str] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)                       # header
        for row in reader:
            if row:
                done.add(Path(row[0]).stem)
    return done


def _load_label_map(orig_dir: Path | None, csv_name: str) -> dict[str, str]:
    """stem → Label, read from the original 3-column ImageNet CSV (if available)."""
    if orig_dir is None:
        return {}
    path = orig_dir / csv_name
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 3:
                labels[Path(row[0]).stem] = row[1]
    return labels


def process_semantic(args, semantic: str) -> tuple[int, int]:
    """Caption all (remaining) real images for one semantic class.
    Returns (n_written, n_failed)."""
    root = Path(args.dataset_path)
    img_dir = _img_dir(root, semantic)
    if not img_dir.exists():
        print(f"[{semantic}] image dir not found, skipping: {img_dir}")
        return 0, 0

    prompt = PROMPT_BY_SEMANTIC[semantic]
    csv_name = OUTPUT_CSV[semantic]
    out_path = Path(args.output_dir) / csv_name
    three_col = semantic in THREE_COLUMN
    label_map = _load_label_map(
        Path(args.orig_captions_dir) if args.orig_captions_dir else None, csv_name
    ) if three_col else {}

    images = sorted(p for p in img_dir.iterdir() if p.suffix in IMAGE_EXTS)
    if args.max_per_class:
        images = images[:args.max_per_class]

    if args.overwrite and out_path.exists():
        out_path.unlink()
    done = _load_existing_stems(out_path)
    todo = [p for p in images if p.stem not in done]

    header_needed = not out_path.exists()
    print(f"[{semantic}] {len(images)} images, {len(done)} already done, "
          f"{len(todo)} to caption → {out_path.name}")
    if not todo:
        return 0, 0

    n_written = n_failed = 0
    with open(out_path, "a", newline="") as f, \
            ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["ImgPath", "Label", "Caption"] if three_col
                            else ["ImgPath", "Caption"])
            f.flush()

        futures = {
            ex.submit(
                caption_image, args.ollama_host, args.model, prompt, p,
                max_side=args.max_image_side, temperature=args.temperature,
                num_predict=args.num_predict, timeout=args.request_timeout,
                max_retries=args.max_retries,
            ): p
            for p in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=semantic):
            p = futures[fut]
            try:
                caption = fut.result()
            except Exception as e:               # noqa: BLE001
                n_failed += 1
                tqdm.write(f"  ! {p.name}: {e}")
                continue
            rel = _rel_imgpath(root, p)
            if three_col:
                writer.writerow([rel, label_map.get(p.stem, ""), caption])
            else:
                writer.writerow([rel, caption])
            f.flush()                            # row-level durability for resume
            n_written += 1

    print(f"[{semantic}] wrote {n_written}, failed {n_failed}")
    return n_written, n_failed


def run_smoke(args) -> None:
    root = Path(args.dataset_path)
    for semantic in args.semantics:
        img_dir = _img_dir(root, semantic)
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix in IMAGE_EXTS) \
            if img_dir.exists() else []
        if not imgs:
            print(f"[{semantic}] no images found at {img_dir}")
            continue
        cap = caption_image(
            args.ollama_host, args.model, PROMPT_BY_SEMANTIC[semantic], imgs[0],
            max_side=args.max_image_side, temperature=args.temperature,
            num_predict=args.num_predict, timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        print(f"\n=== {semantic} :: {imgs[0].name} ===\n{cap}\n")


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Model: {args.model} @ {args.ollama_host} | workers={args.num_workers}")
    wait_for_ollama(args.ollama_host, args.model)

    if args.smoke:
        run_smoke(args)
        return

    total_written = total_failed = 0
    remaining = args.limit
    for semantic in args.semantics:
        if remaining is not None and remaining <= 0:
            break
        # Honour the global --limit by temporarily capping this class.
        if remaining is not None:
            saved = args.max_per_class
            args.max_per_class = (remaining if saved is None
                                  else min(saved, remaining))
        w, fa = process_semantic(args, semantic)
        if remaining is not None:
            args.max_per_class = saved
            remaining -= w
        total_written += w
        total_failed += fa

    print(f"\nDone. Wrote {total_written} captions, {total_failed} failed "
          f"(failed images will be retried on the next run).")
    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
