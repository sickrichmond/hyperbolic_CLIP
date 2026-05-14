"""
Evaluate the trained step-1 model on the OpenFake test set (out-of-distribution).
"""
import io
import warnings

import torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from models.clip_lora import HyperbolicCLIP


class TestSetDataset(Dataset):
    def __init__(self, parquet_path: str,
                 processor_name: str = "openai/clip-vit-base-patch32",
                 max_real: int = 500, max_fake: int = 500):
        print(f"Reading {parquet_path}...")
        table = pq.read_table(parquet_path, columns=["image", "label", "model"])
        df = table.to_pandas()
        print(f"Total rows: {len(df)}")

        self.processor = CLIPImageProcessor.from_pretrained(processor_name)
        self.samples = []
        n_real, n_fake, n_errors = 0, 0, 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="decoding"):
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
                    "model": row.get("model", ""),
                })
                if label == "real":
                    n_real += 1
                else:
                    n_fake += 1
            except (OSError, SyntaxError, ValueError):
                n_errors += 1

        print(f"Built dataset: {n_real} real, {n_fake} fake, {n_errors} errors")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "pixel_values": s["pixel_values"],
            "is_real": torch.tensor(s["is_real"], dtype=torch.bool),
            "model": s["model"],
        }


def main():
    device = "cuda"
    ckpt = torch.load("step1_checkpoint.pt", map_location=device, weights_only=False)

    model = HyperbolicCLIP(hyperbolic_dim=128, curv=ckpt["curv"]).to(device)
    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection_state"])
    model.eval()

    parquet_path = hf_hub_download(
        repo_id="ComplexDataLab/OpenFake",
        filename="core/test-00000-of-00013.parquet",
        repo_type="dataset",
    )

    dataset = TestSetDataset(parquet_path, max_real=500, max_fake=500)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    all_dist, all_label, all_model = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            pixel = batch["pixel_values"].to(device)
            _, tangent = model(pixel)
            all_dist.append(tangent.norm(dim=-1).cpu())
            all_label.append(batch["is_real"])
            all_model.extend(batch["model"])

    dist = torch.cat(all_dist)
    label = torch.cat(all_label)
    real_d = dist[label]
    fake_d = dist[~label]

    print(f"\n=== OOD evaluation on test set ===")
    print(f"Real:   mean={real_d.mean():.3f}, std={real_d.std():.3f}, "
          f"min={real_d.min():.3f}, max={real_d.max():.3f}")
    print(f"Fake:   mean={fake_d.mean():.3f}, std={fake_d.std():.3f}, "
          f"min={fake_d.min():.3f}, max={fake_d.max():.3f}")

    threshold = 1.25
    pred_real = dist < threshold
    correct = (pred_real == label).sum().item()
    total = len(label)
    print(f"\nAccuracy at threshold {threshold}: {correct}/{total} = "
          f"{100*correct/total:.1f}%")
    print(f"Real misclassified as fake: {100*(real_d >= threshold).float().mean():.1f}%")
    print(f"Fake misclassified as real: {100*(fake_d < threshold).float().mean():.1f}%")

    print("\n=== Per-generator accuracy on fakes ===")
    fake_by_model: dict[str, list[float]] = {}
    for d, l, m in zip(dist.tolist(), label.tolist(), all_model):
        if not l:
            fake_by_model.setdefault(m, []).append(d)

    for m, ds in sorted(fake_by_model.items(), key=lambda x: -len(x[1])):
        ds_t = torch.tensor(ds)
        correct = (ds_t >= threshold).sum().item()
        print(f"  {m:30s}: n={len(ds):4d}, "
              f"detected={correct}/{len(ds)} "
              f"({100*correct/len(ds):.0f}%), "
              f"mean_dist={ds_t.mean():.2f}")


if __name__ == "__main__":
    main()