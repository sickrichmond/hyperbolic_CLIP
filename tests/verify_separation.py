"""
Reload trained model, compute distances on the training dataset,
verify real/fake remain well-separated.
"""
import torch
from torch.utils.data import DataLoader

from models.clip_lora import HyperbolicCLIP
from data.dataset import OpenFakePairedDataset


def main():
    device = "cuda"
    ckpt = torch.load("step1_checkpoint.pt", map_location=device, weights_only=False)

    model = HyperbolicCLIP(hyperbolic_dim=128, curv=ckpt["curv"]).to(device)
    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection_state"])
    model.eval()

    dataset = OpenFakePairedDataset(root="/mnt/data3/rtrebiani/openfake_simple")
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    all_dist, all_label = [], []
    with torch.no_grad():
        for batch in loader:
            pixel = batch["pixel_values"].to(device)
            _, tangent = model(pixel)
            all_dist.append(tangent.norm(dim=-1).cpu())
            all_label.append(batch["is_real"])

    dist = torch.cat(all_dist)
    label = torch.cat(all_label)
    real_d = dist[label]
    fake_d = dist[~label]

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
    print(f"Real misclassified: {100*(real_d >= threshold).float().mean():.1f}%")
    print(f"Fake misclassified: {100*(fake_d < threshold).float().mean():.1f}%")


if __name__ == "__main__":
    main()