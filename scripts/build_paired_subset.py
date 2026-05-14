"""
Build a paired subset of OpenFake where each prompt has at least one real
and one fake image. Two passes: scan metadata via pyarrow, then save images
via HF streaming.

Used only when there is enough disk to scan the full dataset. For limited
disk, prefer build_from_single_shard.py.
"""
from __future__ import annotations

import argparse
import io
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image
from tqdm import tqdm


REAL_LABEL = "real"
FAKE_LABEL = "fake"
MIN_PROMPT_LEN = 10


def list_parquet_files(repo_id: str, config: str, split: str) -> list[str]:
    fs = HfFileSystem()
    folder = f"datasets/{repo_id}/{config}"
    all_files = fs.ls(folder, detail=False)
    parquet_files = [
        f for f in all_files
        if f.endswith(".parquet") and f"/{split}-" in f
    ]
    parquet_files.sort()
    return parquet_files


def pass1_via_pyarrow(
    repo_id: str, config: str, split: str, cache_dir: Path,
) -> dict:
    parquet_paths_remote = list_parquet_files(repo_id, config, split)
    print(f"Found {len(parquet_paths_remote)} parquet files for {config}/{split}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"real": 0, "fake": 0})

    for remote_path in tqdm(parquet_paths_remote, desc="pass 1 (parquet files)"):
        rel_in_repo = remote_path.split(f"{repo_id}/", 1)[1]
        local_path = hf_hub_download(
            repo_id=repo_id, filename=rel_in_repo, repo_type="dataset",
            cache_dir=str(cache_dir),
        )

        try:
            table = pq.read_table(local_path, columns=["prompt", "label"])
        except Exception as e:
            print(f"  Failed to read {rel_in_repo}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            continue

        prompts = table.column("prompt").to_pylist()
        labels = table.column("label").to_pylist()
        for prompt, label in zip(prompts, labels):
            if not isinstance(prompt, str):
                continue
            prompt = prompt.strip()
            if len(prompt) < MIN_PROMPT_LEN or prompt.lower() == "nan":
                continue
            if label == REAL_LABEL:
                stats[prompt]["real"] += 1
            elif label == FAKE_LABEL:
                stats[prompt]["fake"] += 1

        try:
            real_path = os.path.realpath(local_path)
            if os.path.exists(real_path):
                os.remove(real_path)
            if os.path.islink(local_path):
                os.remove(local_path)
        except OSError as e:
            print(f"  Warning: could not delete {local_path}: {e}")

    return dict(stats)


def select_paired_prompts(stats: dict, max_prompts: int | None, seed: int = 42) -> set[str]:
    paired = [p for p, c in stats.items() if c["real"] >= 1 and c["fake"] >= 1]
    print(f"Paired prompts (>=1 real, >=1 fake): {len(paired)}")
    if max_prompts is not None and len(paired) > max_prompts:
        rng = random.Random(seed)
        paired = rng.sample(paired, max_prompts)
        print(f"Subsampled to {len(paired)} prompts.")
    return set(paired)


def pass2_save_images(
    streaming_ds, keep_prompts: set[str], output_dir: Path,
    max_fakes_per_prompt: int | None = None, seed: int = 42,
) -> pd.DataFrame:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    prompt_to_id = {p: i for i, p in enumerate(sorted(keep_prompts))}
    fake_counts: dict[str, int] = defaultdict(int)
    records: list[dict] = []
    next_img_idx = 0
    n_errors = 0

    for row in tqdm(streaming_ds, desc="pass 2 (images)"):
        try:
            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                continue
            prompt = prompt.strip()
            if prompt not in keep_prompts:
                continue

            label = row.get("label")
            if label == FAKE_LABEL and max_fakes_per_prompt is not None:
                if fake_counts[prompt] >= max_fakes_per_prompt:
                    continue
                fake_counts[prompt] += 1

            img = row["image"]
            if not isinstance(img, Image.Image):
                if isinstance(img, dict) and "bytes" in img:
                    img = Image.open(io.BytesIO(img["bytes"]))
                else:
                    continue

            img = img.convert("RGB")
            img.thumbnail((256, 256), Image.LANCZOS)
            fname = f"img_{next_img_idx:07d}.jpg"
            fpath = images_dir / fname
            img.save(fpath, format="JPEG", quality=80, optimize=True)

            records.append({
                "image_path": str(fpath.relative_to(output_dir)),
                "prompt": prompt,
                "prompt_id": prompt_to_id[prompt],
                "label": label,
                "model": row.get("model", ""),
                "type": row.get("type", ""),
                "release_date": row.get("release_date", ""),
            })
            next_img_idx += 1

        except (OSError, SyntaxError, ValueError) as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"\n  Skipped corrupt image (#{n_errors}): "
                      f"{type(e).__name__}: {e}")

    if n_errors > 0:
        print(f"\nTotal skipped images due to errors: {n_errors}")
    return pd.DataFrame(records)


def clear_hf_dataset_cache():
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        return
    cache_path = Path(hf_home) / "datasets"
    if cache_path.exists():
        print(f"\nClearing dataset cache at {cache_path}...")
        shutil.rmtree(cache_path, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--max_prompts", type=int, default=10_000)
    ap.add_argument("--max_fakes_per_prompt", type=int, default=None)
    ap.add_argument("--config", type=str, default="core", choices=["core", "reddit"])
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clear_cache_between_passes", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "ComplexDataLab/OpenFake"

    hf_home = os.environ.get("HF_HOME", "")
    parquet_cache_dir = (
        Path(hf_home) / "parquet_pass1" if hf_home else Path("./parquet_pass1")
    )

    print("\n=== Pass 1: scanning metadata via pyarrow ===")
    stats = pass1_via_pyarrow(repo_id, args.config, args.split, parquet_cache_dir)
    print(f"Distinct valid prompts seen: {len(stats)}")

    keep_prompts = select_paired_prompts(stats, args.max_prompts, args.seed)

    if args.clear_cache_between_passes:
        clear_hf_dataset_cache()

    print(f"\nOpening stream for pass 2 (config={args.config}, split={args.split})...")
    ds2 = load_dataset(repo_id, args.config, split=args.split, streaming=True)

    print("\n=== Pass 2: saving images ===")
    manifest = pass2_save_images(
        ds2, keep_prompts, output_dir,
        max_fakes_per_prompt=args.max_fakes_per_prompt, seed=args.seed,
    )

    manifest_path = output_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    n_real = (manifest["label"] == REAL_LABEL).sum()
    n_fake = (manifest["label"] == FAKE_LABEL).sum()
    n_prompts = manifest["prompt_id"].nunique()
    print("\n=== Done ===")
    print(f"Saved {len(manifest)} images "
          f"({n_real} real, {n_fake} fake) across {n_prompts} prompts.")


if __name__ == "__main__":
    main()