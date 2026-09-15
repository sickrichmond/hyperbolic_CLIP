"""Compare checkpoint classification with nearest-anchor cosine classification.

Use the same fixed-seed subset of up to 8,000 harness test images for both
rules. Cone checkpoints minimize the exterior angle xi. Report accuracy,
prediction agreement and per-class disagreements. Observed agreement alone
does not establish a cause or a benefit from geometry.

Usage: IAB_EXCLUDE_GENERATORS=dalle3 python -m tests.probe_cone_vs_cosine CHECKPOINT
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from comparison.dataset.ImageAttributionDataset.dataloader import get_dataloader
from comparison.training.test_hypclip import harness_class_names, load_anchors
from geometry.lorentz import half_aperture, oxy_angle
from models.attribution_clip import AttributionCLIP

N_IMAGES = 8000          # fixed-seed subset, comparable across checkpoints
BATCH = 64


def main(ckpt_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    curv = ckpt.get('curv', 1.0)

    model = AttributionCLIP.from_checkpoint(ckpt).to(device)
    model.clip.load_state_dict(ckpt['lora_state'])
    model.projection.load_state_dict(ckpt['projection'])
    model.eval()

    x_anc = load_anchors(ckpt, model, curv, device)
    K = x_anc.shape[0]
    names = harness_class_names()
    psi = half_aperture(x_anc, curv=curv, min_radius=ckpt.get('min_radius', 0.1))
    print(f"\ncone half-apertures ψ: min={psi.min():.4f}  max={psi.max():.4f}  "
          f"mean={psi.mean():.4f}  spread={psi.max() - psi.min():.4f}")
    print(f"anchor norms ‖x‖:      min={x_anc.norm(dim=-1).min():.3f}  "
          f"max={x_anc.norm(dim=-1).max():.3f}")
    print("A spread near zero means the two rules below CANNOT disagree much — the "
          "cone widths carry no information.")

    _, _, test_loader = get_dataloader(
        root_dir=os.environ['FAST'] + '/datasets/iab_dataset',
        model_name='hypclip', num_images_per_semantic_per_class=2000,
        batch_size=BATCH, degraded=0,
        config={'model_name': 'hypclip', 'clip_name': ckpt['clip_name'], 'num_classes': K},
        num_workers=8)
    test_ds = test_loader.dataset
    g = torch.Generator().manual_seed(0)
    keep = torch.randperm(len(test_ds), generator=g)[:N_IMAGES].tolist()
    loader = DataLoader(Subset(test_ds, keep), batch_size=BATCH, shuffle=False,
                        num_workers=8)
    print(f"\n{len(test_ds)} test images, probing {len(keep)}")

    # exp_map0 is radial, so the space component of a hyperboloid point points along
    # its tangent vector: cosine on x_anc/x_img IS cosine in tangent space.
    anc_dir = F.normalize(x_anc, dim=-1)

    rule = "argmin ξ"
    print(f"Cone rule: {rule}")

    cone_pred, cos_pred, all_labels = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="probe", leave=False):
            x_img, _ = model.encode_image(b['image'].to(device))
            B = x_img.shape[0]
            score = oxy_angle(
                x_anc.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1),
                x_img.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1),
                curv=curv).reshape(B, K)
            cone_pred.append(score.argmin(1).cpu())
            cos_pred.append((F.normalize(x_img, dim=-1) @ anc_dir.T).argmax(1).cpu())
            all_labels.append(b['label'].cpu())

    cone = torch.cat(cone_pred)
    cos = torch.cat(cos_pred)
    y = torch.cat(all_labels)
    agree = (cone == cos).float().mean().item()

    print(f"\n  accuracy, {rule} (the model)  : {(cone == y).float().mean():.4f}")
    print(f"  accuracy, argmax cos (no geometry)  : {(cos == y).float().mean():.4f}")
    print(f"  AGREEMENT between the two rules     : {agree:.4f}")

    disagree = cone != cos
    n = int(disagree.sum())
    print(f"\n  {n} / {len(y)} images decided differently")
    if n:
        # Where they differ, which rule was right? If the cones are only ever wrong,
        # the widths are actively hurting; if they are right, they earn their place.
        cone_right = int((cone[disagree] == y[disagree]).sum())
        print(f"    cones right, cosine wrong : {cone_right}")
        print(f"    cosine right, cones wrong : {int((cos[disagree] == y[disagree]).sum())}")
        cls = torch.bincount(y[disagree], minlength=K)
        top = cls.argsort(descending=True)[:5]
        print("    concentrated on: " +
              ", ".join(f"{names[i]}={int(cls[i])}" for i in top if cls[i] > 0))


if __name__ == '__main__':
    main(sys.argv[1])
