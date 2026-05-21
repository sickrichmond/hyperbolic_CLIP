"""
Linear probe evaluation: train a logistic regression on the 128D tangent
embeddings from a trained checkpoint, then evaluate on the OOD test set.

This is a fairer evaluation than threshold-on-norm for models trained with
L_pair, which encode the real/fake distinction angularly. The probe can
exploit all 128 dimensions, including the angular structure.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m tests.linear_probe_eval \
        --checkpoint step1_checkpoint_paired.pt \
        --train_dataset /mnt/data3/rtrebiani/openfake_paired

Embeddings are cached to <cache_dir>/<ckpt_stem>_{train,test}.npz so re-running
the script with different classifier settings does not require recomputing.
"""
from __future__ import annotations

import argparse
import io
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from models.clip_lora import HyperbolicCLIP
from data.dataset import OpenFakePairedDataset


class TestSetDataset(Dataset):
    """OpenFake test split, decoded into a fixed-size in-memory dataset."""

    def __init__(
        self,
        parquet_path: str,
        processor_name: str = "openai/clip-vit-base-patch32",
        max_real: int = 500,
        max_fake: int = 500,
    ):
        table = pq.read_table(parquet_path, columns=["image", "label", "model"])
        df = table.to_pandas()
        self.processor = CLIPImageProcessor.from_pretrained(processor_name)
        self.samples = []
        n_real, n_fake = 0, 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="decoding test"):
            try:
                label = row["label"]
                if label == "real" and n_real >= max_real:
                    continue
                if label == "fake" and n_fake >= max_fake:
                    continue
                img_data = row["image"]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img = Image.open(io.BytesIO(img_data["bytes"]))
                elif isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data))
                else:
                    continue
                img = img.convert("RGB")
                pixel = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
                self.samples.append({
                    "pixel_values": pixel,
                    "is_real": label == "real",
                    "model": str(row.get("model") or ""),
                })
                if label == "real":
                    n_real += 1
                else:
                    n_fake += 1
            except (OSError, SyntaxError, ValueError):
                pass
        print(f"Built test dataset: {n_real} real, {n_fake} fake")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "pixel_values": s["pixel_values"],
            "is_real": torch.tensor(s["is_real"], dtype=torch.bool),
            "model": s["model"],
        }


def extract_embeddings(model, loader, device, desc="extracting"):
    model.eval()
    feats, labels, gen_models = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            pixel = batch["pixel_values"].to(device)
            _, tangent = model(pixel)
            feats.append(tangent.cpu().numpy())
            labels.append(batch["is_real"].numpy())
            if "model" in batch:
                gen_models.extend(batch["model"])
    return np.concatenate(feats), np.concatenate(labels), gen_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_dataset", required=True,
                    help="Path to training dataset (manifest.parquet + images/)")
    ap.add_argument("--cache_dir", default="/mnt/data3/rtrebiani/probe_embeddings")
    ap.add_argument("--C", type=float, default=1.0,
                    help="LogisticRegression inverse regularization strength")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda"
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = Path(args.checkpoint).stem
    train_cache = cache_dir / f"{ckpt_stem}_train.npz"
    test_cache = cache_dir / f"{ckpt_stem}_test.npz"

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HyperbolicCLIP(hyperbolic_dim=128, curv=ckpt["curv"]).to(device)
    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection_state"])

    # --- 1. Train embeddings ---
    if train_cache.exists():
        print(f"Loading cached train embeddings: {train_cache}")
        data = np.load(train_cache)
        X_train, y_real_train = data["X"], data["y_real"]
    else:
        print(f"Extracting train embeddings from {args.train_dataset}...")
        dataset = OpenFakePairedDataset(root=args.train_dataset)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        X_train, y_real_train, _ = extract_embeddings(model, loader, device, "train")
        np.savez(train_cache, X=X_train, y_real=y_real_train)
        print(f"Saved cache: {train_cache} ({X_train.nbytes/1e6:.0f} MB)")

    y_train = (~y_real_train.astype(bool)).astype(int)  # 1 if fake, 0 if real

    # --- 2. Train classifier ---
    print(f"\nTraining LogisticRegression on {X_train.shape[0]} samples × {X_train.shape[1]}D...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, C=args.C, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    train_acc = clf.score(X_train_s, y_train)
    print(f"Train accuracy (on probe training data): {train_acc:.4f}")

    # --- 3. Test embeddings ---
    if test_cache.exists():
        print(f"\nLoading cached test embeddings: {test_cache}")
        data = np.load(test_cache, allow_pickle=True)
        X_test, y_real_test = data["X"], data["y_real"]
        test_models = data["models"].tolist()
    else:
        print(f"\nDownloading and extracting test set embeddings...")
        parquet_path = hf_hub_download(
            repo_id="ComplexDataLab/OpenFake",
            filename="core/test-00000-of-00013.parquet",
            repo_type="dataset",
        )
        test_set = TestSetDataset(parquet_path, max_real=500, max_fake=500)
        test_loader = DataLoader(
            test_set, batch_size=64, shuffle=False, num_workers=2
        )
        X_test, y_real_test, test_models = extract_embeddings(
            model, test_loader, device, "test"
        )
        np.savez(test_cache, X=X_test, y_real=y_real_test,
                 models=np.array(test_models, dtype=object))
        print(f"Saved cache: {test_cache}")

    y_test = (~y_real_test.astype(bool)).astype(int)

    # --- 4. Evaluate ---
    X_test_s = scaler.transform(X_test)
    y_pred = clf.predict(X_test_s)
    acc = float((y_pred == y_test).mean())

    real_mask = y_test == 0
    fake_mask = y_test == 1
    real_misclas = float((y_pred[real_mask] == 1).mean())
    fake_misclas = float((y_pred[fake_mask] == 0).mean())

    print(f"\n=== Linear probe OOD results ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Probe train: {args.train_dataset} ({X_train.shape[0]} samples)")
    print(f"Test set:    {len(y_test)} samples "
          f"({int(real_mask.sum())} real, {int(fake_mask.sum())} fake)")
    print(f"OOD accuracy:               {acc:.4f}  ({int((y_pred == y_test).sum())}/{len(y_test)})")
    print(f"Real misclassified as fake: {real_misclas:.4f}")
    print(f"Fake misclassified as real: {fake_misclas:.4f}")

    # --- 5. Per-generator on fakes ---
    print(f"\n=== Per-generator accuracy on fakes ===")
    fake_idx = np.where(fake_mask)[0]
    by_model: dict[str, list[bool]] = {}
    for idx in fake_idx:
        m = test_models[idx] if idx < len(test_models) else ""
        by_model.setdefault(m, []).append(bool(y_pred[idx] == 1))

    for m, hits in sorted(by_model.items(), key=lambda x: -len(x[1])):
        hits_arr = np.array(hits)
        print(f"  {m:30s}: n={len(hits):4d}, "
              f"detected={hits_arr.sum()}/{len(hits)} "
              f"({100*hits_arr.mean():.0f}%)")


if __name__ == "__main__":
    main()
