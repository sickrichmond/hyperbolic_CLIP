"""
Build a small real/fake subset of OpenFake from a single train shard.
No prompt pairing required: just save all real and fake images.
"""
from __future__ import annotations

import argparse
import io
import warnings
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

REAL_LABEL = "real"
FAKE_LABEL = "fake"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--shard_filename", type=str,
                    default="core/train-00000-of-00032-00000.parquet")
    ap.add_argument("--max_real", type=int, default=None)
    ap.add_argument("--max_fake", type=int, default=None)
    ap.add_argument("--max_per_fake_model", type=int, default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading shard: {args.shard_filename}")
    parquet_path = hf_hub_download(
        repo_id="ComplexDataLab/OpenFake",
        filename=args.shard_filename,
        repo_type="dataset",
    )

    print("Reading parquet...")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    print(f"Shard contains {len(df)} rows")
    print(f"Labels: {df['label'].value_counts().to_dict()}")

    n_real_saved = 0
    n_fake_saved = 0
    fake_model_counts: dict[str, int] = {}
    records: list[dict] = []
    n_errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="processing"):
        try:
            label = row["label"]

            if label == REAL_LABEL:
                if args.max_real is not None and n_real_saved >= args.max_real:
                    continue
            elif label == FAKE_LABEL:
                if args.max_fake is not None and n_fake_saved >= args.max_fake:
                    continue
                model = row.get("model", "")
                if args.max_per_fake_model is not None:
                    if fake_model_counts.get(model, 0) >= args.max_per_fake_model:
                        continue
                    fake_model_counts[model] = fake_model_counts.get(model, 0) + 1
            else:
                continue

            img_data = row["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                img = Image.open(io.BytesIO(img_data["bytes"]))
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data))
            else:
                continue

            img = img.convert("RGB")
            img.thumbnail((256, 256), Image.LANCZOS)

            if label == REAL_LABEL:
                fname = f"real_{n_real_saved:06d}.jpg"
                n_real_saved += 1
            else:
                fname = f"fake_{n_fake_saved:06d}.jpg"
                n_fake_saved += 1

            fpath = images_dir / fname
            img.save(fpath, format="JPEG", quality=80, optimize=True)

            records.append({
                "image_path": str(fpath.relative_to(output_dir)),
                "prompt": str(row.get("prompt") or ""),
                "prompt_id": len(records),
                "label": label,
                "model": row.get("model", ""),
                "type": row.get("type", ""),
                "release_date": str(row.get("release_date", "")),
            })

        except (OSError, SyntaxError, ValueError) as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"\n  Skipped corrupt image (#{n_errors}): "
                      f"{type(e).__name__}: {e}")

    manifest = pd.DataFrame(records)
    manifest_path = output_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    print("\n=== Done ===")
    print(f"Saved {len(manifest)} images ({n_real_saved} real, {n_fake_saved} fake)")
    print(f"Manifest: {manifest_path}")
    if n_errors > 0:
        print(f"Skipped {n_errors} corrupt images.")

    print("\nFake breakdown by generator:")
    print(manifest[manifest['label'] == 'fake']['model'].value_counts())


if __name__ == "__main__":
    main()