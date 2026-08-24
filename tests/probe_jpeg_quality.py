"""D3 — does JPEG REMOVE our signal or INJECT a false one?

Evaluates an existing checkpoint at JPEG quality 95/85/75/65/30 on the harness
test split. The benchmark suite starts at 65, so 95/85/75 are new territory:

  * accuracy already broken at q95 (visually lossless)  -> JPEG *injects* a
    structure the model reads as a class (see dataset.py:161 — only `real`
    accepts .jpg, so "JPEG artifact => real" is a free feature on clean data);
  * smooth decay q95 -> q65                             -> JPEG *removes* the
    band our fingerprint lives in.

Standalone: touches no repo file. Run it on a GPU node.

    python -m tests.probe_jpeg_quality $WORK/hyp_fine_tuning/checkpoints/attribution_22cls_sweepwin_vitl14.pt
"""
import os
import sys
import types

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from comparison.dataset.ImageAttributionDataset.dataloader import get_dataloader
from comparison.training.metrics.base_metrics_class import calculate_metrics_for_test
from comparison.training.test_hypclip import load_anchors, harness_class_names
from geometry.lorentz import half_aperture, oxy_angle
from models.attribution_clip import AttributionCLIP

QUALITIES = [95, 85, 75, 65, 30]
N_IMAGES = 4000          # same subset for every quality — that is what matters
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
    real_idx = names.index('real')

    # Anchor geometry. A degraded image loses specificity, so its embedding drifts
    # toward the origin — and a shallow point falls into the WIDEST cone. If `real`
    # holds the smallest ‖t‖ it is the default sink for every corrupted image, which
    # would explain recall(real)=1.000 under JPEG without any codec shortcut.
    mr = ckpt.get('min_radius', 0.1)
    psi = half_aperture(x_anc, curv=curv, min_radius=mr)
    order = torch.argsort(psi, descending=True)
    print("\nanchor cones (widest first):")
    for i in order.tolist():
        mark = '  <-- real' if i == real_idx else ''
        print(f"  {names[i]:<16} ‖t‖={x_anc[i].norm():6.2f}  psi={psi[i]:.3f}{mark}")

    # degraded=0: the split, the enumeration and the 2000-cap are the harness's,
    # so these are the same test images as every baseline. The compression is
    # applied below by overriding get_degraded_img, which dataset.__getitem__
    # calls unconditionally in test mode.
    _, _, test_loader = get_dataloader(
        root_dir=os.environ['FAST'] + '/datasets/iab_dataset',
        model_name='hypclip', num_images_per_semantic_per_class=2000,
        batch_size=BATCH, degraded=0,
        config={'model_name': 'hypclip', 'clip_name': ckpt['clip_name'], 'num_classes': K},
        num_workers=8)

    test_ds = test_loader.dataset                      # Subset over the base dataset
    base = test_ds.dataset
    g = torch.Generator().manual_seed(0)
    keep = torch.randperm(len(test_ds), generator=g)[:N_IMAGES].tolist()
    subset = Subset(test_ds, keep)
    print(f"{len(test_ds)} test images, evaluating on {len(subset)}")

    if 'real' not in names or len(names) != 22:
        print(f"WARNING: {len(names)} classes — set IAB_EXCLUDE_GENERATORS=dalle3 for the "
              f"22-class setup the checkpoint was trained on.")

    print(f"\n{'quality':>8} {'acc':>8} {'auc':>8} {'recall(real)':>14} {'recall(other)':>14}"
          f" {'mean |x|':>10} {'pred=real':>10} {'xi(real)':>10} {'xi(true)':>10}")
    for q in QUALITIES:
        base.jpeg_q = q
        base.get_degraded_img = types.MethodType(
            lambda self, img: self.compress(img, quality=self.jpeg_q), base)

        loader = DataLoader(subset, batch_size=BATCH, shuffle=False, num_workers=8)
        logits, labels, sems, norms = [], [], [], []
        with torch.no_grad():
            for b in tqdm(loader, desc=f"q{q}", leave=False):
                x_img, _ = model.encode_image(b['image'].to(device))
                norms.append(x_img.norm(dim=-1).cpu())
                B = x_img.shape[0]
                xi = oxy_angle(x_anc.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1),
                               x_img.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1),
                               curv=curv).reshape(B, K)
                logits.append((-xi).cpu())
                labels.append(b['label'].cpu())
                sems.append(b['semantic_label'].cpu())

        all_logits = torch.cat(logits)
        all_labels = torch.cat(labels)
        auc, acc, ap, _, _, extra = calculate_metrics_for_test(
            all_labels, all_logits, torch.cat(sems), need_softmax=True)
        rec = extra['recall_per_class']
        other = [r for i, r in enumerate(rec) if i != real_idx]
        pred_real = (all_logits.argmax(1) == real_idx).float().mean()
        # logits are -xi, so flip the sign back. Excluding the images that ARE real
        # keeps xi(real) an honest "how close is everything else to the real cone".
        xi = -all_logits
        not_real = all_labels != real_idx
        xi_real = xi[not_real, real_idx].mean()
        xi_true = xi.gather(1, all_labels.unsqueeze(1)).squeeze(1)[not_real].mean()
        print(f"{q:>8} {acc:>8.4f} {auc:>8.4f} {rec[real_idx]:>14.3f} "
              f"{sum(other)/len(other):>14.3f} {torch.cat(norms).mean():>10.2f} "
              f"{pred_real:>10.3f} {xi_real:>10.3f} {xi_true:>10.3f}")


if __name__ == '__main__':
    main(sys.argv[1])
