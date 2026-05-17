"""
Build a paired real/fake dataset from OpenFake for L_pair training.

Pass 1 (metadata only, zero local disk):
    For each of the 608 train parquet files, read only the 'prompt' and 'label'
    columns via HTTP range requests using HfFileSystem + pyarrow column projection.
    Collects per-prompt real/fake counts. No full-file download needed.

Pass 2 (streaming, zero temp disk):
    Stream the full OpenFake train split via HuggingFace streaming API.
    Save only images whose prompt appears in keep_prompts (i.e. has ≥1 real AND
    ≥1 fake). Images are thumbnailed to 256×256 JPEG and saved to output_dir.

The result is a dataset where every prompt_id maps to ≥1 real and ≥1 fake image,
suitable for building (real, fake) pairs for the L_pair loss.

Run in tmux — pass1 takes ~4-5h, pass2 ~6-10h depending on bandwidth.
Progress is saved to pass1_stats.parquet after pass1 so if pass2 fails, pass1
can be skipped on restart.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset, Image as DSImage
from huggingface_hub import HfFileSystem
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

REPO_ID = "ComplexDataLab/OpenFake"
REAL_LABEL = "real"
FAKE_LABEL = "fake"
MIN_PROMPT_LEN = 10


def list_parquet_files(config: str, split: str) -> list[str]:
    fs = HfFileSystem()
    folder = f"datasets/{REPO_ID}/{config}"
    all_files = fs.ls(folder, detail=False)
    files = sorted([
        f for f in all_files
        if f.endswith(".parquet") and f"/{split}-" in f
    ])
    return files


def pass1_http_column_projection(
    parquet_paths: list[str],
    stats_cache: Path,
) -> dict[str, dict[str, int]]:
    """
    Read only prompt+label from each parquet via HTTP range requests.
    No local disk usage — pyarrow fetches only the relevant column chunks.
    Saves incremental progress to stats_cache so pass1 can be resumed.
    """
    if stats_cache.exists():
        print(f"Found cached pass1 stats at {stats_cache}, loading...")
        df_stats = pd.read_parquet(stats_cache)
        stats: dict[str, dict[str, int]] = {}
        for row in df_stats.itertuples():
            stats[row.prompt] = {"real": row.real, "fake": row.fake}
        print(f"Loaded {len(stats)} prompts from cache.")
        return stats

    fs = HfFileSystem()
    stats = defaultdict(lambda: {"real": 0, "fake": 0})
    n_errors = 0

    for path in tqdm(parquet_paths, desc="pass1 (HTTP column projection)"):
        try:
            with fs.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                table = pf.read(columns=["prompt", "label"])

            prompts = table.column("prompt").to_pylist()
            labels = table.column("label").to_pylist()
            for prompt, label in zip(prompts, labels):
                if not isinstance(prompt, str):
                    continue
                prompt = prompt.strip()
                if len(prompt) < MIN_PROMPT_LEN:
                    continue
                if label == REAL_LABEL:
                    stats[prompt]["real"] += 1
                elif label == FAKE_LABEL:
                    stats[prompt]["fake"] += 1
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                fname = path.split("/")[-1]
                print(f"\n  Warning: failed {fname}: {e}")

    if n_errors > 0:
        print(f"\nTotal parquet read errors: {n_errors}")

    # Save to cache so pass2-only restarts skip pass1
    records = [
        {"prompt": p, "real": c["real"], "fake": c["fake"]}
        for p, c in stats.items()
    ]
    pd.DataFrame(records).to_parquet(stats_cache, index=False)
    print(f"Saved pass1 stats ({len(stats)} prompts) to {stats_cache}")
    return dict(stats)


def select_paired_prompts(
    stats: dict[str, dict[str, int]],
    max_prompts: int | None,
    seed: int,
) -> set[str]:
    paired = [
        p for p, c in stats.items()
        if c["real"] >= 1 and c["fake"] >= 1
    ]
    print(f"Prompts with ≥1 real AND ≥1 fake: {len(paired)}")
    if max_prompts is not None and len(paired) > max_prompts:
        rng = random.Random(seed)
        paired = rng.sample(paired, max_prompts)
        print(f"Subsampled to {max_prompts} prompts.")
    return set(paired)


def pass2_streaming(
    keep_prompts: set[str],
    output_dir: Path,
    config: str,
    split: str,
    max_fakes_per_prompt: int | None,
) -> pd.DataFrame:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt → integer id mapping (for PairedPromptBatchSampler)
    prompt_to_id = {p: i for i, p in enumerate(sorted(keep_prompts))}

    fake_counts: dict[str, int] = defaultdict(int)
    records: list[dict] = []
    img_idx = 0
    n_errors = 0

    print(f"\nStreaming {REPO_ID} config={config} split={split} ...")
    ds = load_dataset(REPO_ID, config, split=split, streaming=True)
    # Disable automatic image decoding by HF datasets: corrupt EXIF in remote
    # PNGs caused PIL to crash inside the datasets worker thread, corrupting
    # the GIL and killing the process. With decode=False, image bytes are
    # returned raw and we decode them ourselves inside try/except below.
    ds = ds.cast_column("image", DSImage(decode=False))

    for row in tqdm(ds, desc="pass2 (streaming)"):
        try:
            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                continue
            prompt = prompt.strip()
            if prompt not in keep_prompts:
                continue

            label = row.get("label")
            if label not in (REAL_LABEL, FAKE_LABEL):
                continue

            if label == FAKE_LABEL and max_fakes_per_prompt is not None:
                if fake_counts[prompt] >= max_fakes_per_prompt:
                    continue
                fake_counts[prompt] += 1

            img = row.get("image")
            if not isinstance(img, Image.Image):
                if isinstance(img, dict) and "bytes" in img:
                    img = Image.open(io.BytesIO(img["bytes"]))
                else:
                    continue

            img = img.convert("RGB")
            img.thumbnail((256, 256), Image.LANCZOS)
            fname = f"img_{img_idx:08d}.jpg"
            img.save(images_dir / fname, format="JPEG", quality=80, optimize=True)

            records.append({
                "image_path": f"images/{fname}",
                "prompt": prompt,
                "prompt_id": prompt_to_id[prompt],
                "label": label,
                "model": str(row.get("model") or ""),
                "type": str(row.get("type") or ""),
                "release_date": str(row.get("release_date") or ""),
            })
            img_idx += 1

            # Save incremental manifest every 5000 images
            if img_idx % 5000 == 0:
                pd.DataFrame(records).to_parquet(output_dir / "manifest.parquet", index=False)
                print(f"\n  Saved incremental manifest ({img_idx} images so far)")

        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"\n  Skipped corrupt image #{n_errors}: {e}")

    if n_errors > 0:
        print(f"\nTotal skipped images: {n_errors}")

    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="core", choices=["core", "reddit"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_prompts", type=int, default=None,
                    help="Cap on number of paired prompts. Default: all.")
    ap.add_argument("--max_fakes_per_prompt", type=int, default=None,
                    help="Cap on fake images per prompt. Default: all.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_pass1", action="store_true",
                    help="Skip pass1 if pass1_stats.parquet already exists.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_cache = output_dir / "pass1_stats.parquet"

    # --- Pass 1 ---
    print("=== Pass 1: HTTP column projection (no local disk) ===")
    parquet_files = list_parquet_files(args.config, args.split)
    print(f"Found {len(parquet_files)} parquet files")
    stats = pass1_http_column_projection(parquet_files, stats_cache)
    print(f"Distinct prompts seen: {len(stats)}")

    keep_prompts = select_paired_prompts(stats, args.max_prompts, args.seed)
    print(f"Target paired prompts: {len(keep_prompts)}")

    # Summary of what to expect
    n_expected_real = sum(stats[p]["real"] for p in keep_prompts)
    n_expected_fake = sum(stats[p]["fake"] for p in keep_prompts)
    print(f"Expected images (may overlap across shards): "
          f"~{n_expected_real} real, ~{n_expected_fake} fake")

    # --- Pass 2 ---
    print("\n=== Pass 2: Streaming images ===")
    manifest = pass2_streaming(
        keep_prompts=keep_prompts,
        output_dir=output_dir,
        config=args.config,
        split=args.split,
        max_fakes_per_prompt=args.max_fakes_per_prompt,
    )

    manifest.to_parquet(output_dir / "manifest.parquet", index=False)

    n_real = (manifest["label"] == REAL_LABEL).sum()
    n_fake = (manifest["label"] == FAKE_LABEL).sum()
    n_prompts = manifest["prompt_id"].nunique()

    print(f"\n=== Done ===")
    print(f"Total: {len(manifest)} images ({n_real} real, {n_fake} fake) "
          f"across {n_prompts} paired prompts")
    print(f"Dataset: {output_dir}")
    total_mb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1e6
    print(f"Disk usage: {total_mb:.0f} MB")

    print("\nFake distribution by generator (top 20):")
    print(manifest[manifest.label == FAKE_LABEL]["model"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
