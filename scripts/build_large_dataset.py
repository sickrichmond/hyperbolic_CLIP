"""
Build a large real/fake dataset by processing one parquet file per OpenFake
super-shard (train-XXXX-of-00032-00000.parquet, X in [start_shard, end_shard]).

Each parquet is downloaded, images extracted, then the parquet is deleted from
the HF cache to free disk. With 32 super-shards the final dataset is ~120k
images (~1.5 GB) while never holding more than ~6 GB on disk at once.

Optionally incorporates an existing openfake_simple/ directory (shard 00000
already extracted) to skip one re-download.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

REPO_ID = "ComplexDataLab/OpenFake"
REAL_LABEL = "real"
FAKE_LABEL = "fake"


def shard_filename(shard_idx: int) -> str:
    return f"core/train-{shard_idx:05d}-of-00032-00000.parquet"


def delete_from_hf_cache(local_path: str) -> None:
    try:
        real_path = os.path.realpath(local_path)
        if os.path.exists(real_path):
            os.remove(real_path)
        if os.path.islink(local_path):
            os.remove(local_path)
    except OSError as e:
        print(f"  Warning: could not delete cached shard: {e}")


def process_parquet(
    parquet_path: str,
    images_dir: Path,
    img_offset: int,
    per_model_counts: dict[str, int],
    max_real: int | None,
    max_fake: int | None,
    max_per_fake_model: int | None,
    shard_idx: int,
) -> list[dict]:
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    records: list[dict] = []
    n_real_saved = 0
    n_fake_saved = 0
    n_errors = 0
    img_counter = img_offset

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  shard {shard_idx:02d}", leave=False):
        try:
            label = row.get("label")
            if label not in (REAL_LABEL, FAKE_LABEL):
                continue

            if label == REAL_LABEL:
                if max_real is not None and n_real_saved >= max_real:
                    continue
            else:
                if max_fake is not None and n_fake_saved >= max_fake:
                    continue
                model = str(row.get("model") or "")
                if max_per_fake_model is not None:
                    if per_model_counts.get(model, 0) >= max_per_fake_model:
                        continue
                    per_model_counts[model] = per_model_counts.get(model, 0) + 1

            img_data = row.get("image")
            if isinstance(img_data, dict) and "bytes" in img_data:
                img = Image.open(io.BytesIO(img_data["bytes"]))
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data))
            else:
                continue

            img = img.convert("RGB")
            img.thumbnail((256, 256), Image.LANCZOS)

            fname = f"img_{img_counter:08d}.jpg"
            img.save(images_dir / fname, format="JPEG", quality=80, optimize=True)

            records.append({
                "image_path": f"images/{fname}",
                "prompt": str(row.get("prompt") or ""),
                "prompt_id": img_counter,
                "label": label,
                "model": str(row.get("model") or ""),
                "type": str(row.get("type") or ""),
                "release_date": str(row.get("release_date") or ""),
                "shard": shard_idx,
            })

            img_counter += 1
            if label == REAL_LABEL:
                n_real_saved += 1
            else:
                n_fake_saved += 1

        except (OSError, SyntaxError, ValueError) as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"\n  Skipped corrupt image: {type(e).__name__}: {e}")

    return records


def import_existing(existing_dir: Path, output_dir: Path) -> list[dict]:
    """Copy images from an existing openfake_simple/ into output_dir."""
    manifest_path = existing_dir / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.parquet in {existing_dir}")

    existing_df = pd.read_parquet(manifest_path)
    images_src = existing_dir / "images"
    images_dst = output_dir / "images"

    print(f"Copying {len(existing_df)} images from {existing_dir}...")
    records: list[dict] = []
    for i, row in enumerate(tqdm(existing_df.itertuples(), total=len(existing_df),
                                  desc="  copy shard 00 (existing)", leave=False)):
        src = existing_dir / row.image_path
        fname = f"img_{i:08d}.jpg"
        dst = images_dst / fname
        shutil.copy2(src, dst)
        records.append({
            "image_path": f"images/{fname}",
            "prompt": str(getattr(row, "prompt", "") or ""),
            "prompt_id": i,
            "label": row.label,
            "model": str(getattr(row, "model", "") or ""),
            "type": str(getattr(row, "type", "") or ""),
            "release_date": str(getattr(row, "release_date", "") or ""),
            "shard": 0,
        })

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--start_shard", type=int, default=0,
                    help="First super-shard index (default 0)")
    ap.add_argument("--end_shard", type=int, default=31,
                    help="Last super-shard index inclusive (default 31)")
    ap.add_argument("--existing_dir", type=str, default=None,
                    help="Path to existing openfake_simple/ to incorporate as "
                         "shard 00 without re-downloading")
    ap.add_argument("--max_real", type=int, default=None,
                    help="Max real images per shard")
    ap.add_argument("--max_fake", type=int, default=None,
                    help="Max fake images per shard")
    ap.add_argument("--max_per_fake_model", type=int, default=None,
                    help="Global cap on fake images per generator model")
    ap.add_argument("--delete_after", action="store_true", default=True,
                    help="Delete parquet from HF cache after each shard (default True)")
    ap.add_argument("--no_delete_after", dest="delete_after", action="store_false")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    per_model_counts: dict[str, int] = defaultdict(int)

    # --- Shard 00: use existing or download ---
    if args.existing_dir and args.start_shard == 0:
        existing_dir = Path(args.existing_dir)
        shard0_records = import_existing(existing_dir, output_dir)
        # Populate per_model_counts from existing data
        for r in shard0_records:
            if r["label"] == FAKE_LABEL and r["model"]:
                per_model_counts[r["model"]] += 1
        all_records.extend(shard0_records)
        print(f"  Incorporated {len(shard0_records)} images from existing dir.")
        first_download_shard = 1
    else:
        first_download_shard = args.start_shard

    # --- Remaining shards ---
    shard_range = range(first_download_shard, args.end_shard + 1)
    total_shards = len(shard_range) + (1 if args.existing_dir and args.start_shard == 0 else 0)
    print(f"\nProcessing {len(shard_range)} shard(s) to download "
          f"(total including existing: {total_shards})...")

    for shard_idx in shard_range:
        fname = shard_filename(shard_idx)
        print(f"\n[{shard_idx}/{args.end_shard}] Downloading {fname}...")
        try:
            parquet_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  ERROR downloading shard {shard_idx}: {e}")
            continue

        img_offset = len(all_records)
        records = process_parquet(
            parquet_path=parquet_path,
            images_dir=images_dir,
            img_offset=img_offset,
            per_model_counts=per_model_counts,
            max_real=args.max_real,
            max_fake=args.max_fake,
            max_per_fake_model=args.max_per_fake_model,
            shard_idx=shard_idx,
        )
        all_records.extend(records)

        n_real = sum(1 for r in records if r["label"] == REAL_LABEL)
        n_fake = sum(1 for r in records if r["label"] == FAKE_LABEL)
        print(f"  Extracted {len(records)} images ({n_real} real, {n_fake} fake). "
              f"Running total: {len(all_records)}")

        if args.delete_after:
            delete_from_hf_cache(parquet_path)
            print(f"  Deleted cached parquet.")

        # Save incremental manifest after each shard (allows resume inspection)
        _save_manifest(all_records, output_dir)

    # --- Final summary ---
    manifest = pd.DataFrame(all_records)
    _save_manifest(all_records, output_dir)

    n_real_total = (manifest["label"] == REAL_LABEL).sum()
    n_fake_total = (manifest["label"] == FAKE_LABEL).sum()
    print(f"\n=== Done ===")
    print(f"Total: {len(manifest)} images ({n_real_total} real, {n_fake_total} fake)")
    print(f"Dataset: {output_dir}")
    print(f"Disk usage: {_dir_size_mb(output_dir):.0f} MB")

    print("\nFake breakdown by generator (top 30):")
    top = manifest[manifest["label"] == FAKE_LABEL]["model"].value_counts().head(30)
    print(top.to_string())


def _save_manifest(records: list[dict], output_dir: Path) -> None:
    pd.DataFrame(records).to_parquet(output_dir / "manifest.parquet", index=False)


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1e6


if __name__ == "__main__":
    main()
