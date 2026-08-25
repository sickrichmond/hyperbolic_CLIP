"""Does the hyperbolic geometry decide anything, or is it a cosine classifier?

The hypothesis, stated so it can be falsified: with --target_norm 4.0 every anchor
is pushed to the same norm, so every cone gets the same half-aperture ψ. When the
ψ are equal, `argmin_c ξ_c` — the inference rule — degenerates into "the anchor at
the smallest ANGLE", i.e. nearest-class-direction. exp_map0 is radial, so direction
is preserved from the tangent space onto the hyperboloid, and the whole hyperbolic
apparatus would then be an expensive way to write `argmax_c cos(x, a_c)`. That
would also explain why a two-term hinge is enough.

This measures it directly: same images, same model, two decision rules.

    IAB_EXCLUDE_GENERATORS=dalle3 python -m tests.probe_cone_vs_cosine \\
        $WORK/hyp_fine_tuning/checkpoints/attribution_22cls_sweepwin_vitl14.pt

Read `agreement`. >= 0.99 confirms the hypothesis (declare it as a limit: we have
cones, not hierarchy). Clearly below, and the cone widths ARE doing work, which is
a positive result worth claiming. The ψ spread printed first says how much room
there was for a difference at all.

Standalone: touches no repo file. GPU node.
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
from losses.axis_cone_loss import axis_cone_q, sin_psi_from_depth
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

    # The cone rule depends on which loss trained the checkpoint: argmin xi for the
    # entailment-cone loss, argmin q for the axis loss. Reading the wrong one would
    # answer the wrong question without failing.
    loss_kind = ckpt.get('loss', 'cone')
    min_radius = ckpt.get('min_radius', 0.1)
    rule = 'argmin q (axis cone)' if loss_kind == 'axis' else 'argmin ξ'
    # ‖a‖ = 2K/sin psi is maintained by the trainer, so the aperture comes back from the
    # anchors themselves — no second array to keep in the same class order.
    sin_psi = sin_psi_from_depth(x_anc, min_radius) if loss_kind == 'axis' else None
    print(f"Cone rule: {rule}")
    if sin_psi is not None:
        psi_deg = torch.rad2deg(torch.arcsin(sin_psi))
        print(f"  ψ per class: min {psi_deg.min():.1f}°  mean {psi_deg.mean():.1f}°  "
              f"max {psi_deg.max():.1f}°  (a SPREAD here is what lets argmin q differ "
              f"from argmax cos at all)")

    cone_pred, cos_pred, all_labels = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="probe", leave=False):
            x_img, _ = model.encode_image(b['image'].to(device))
            B = x_img.shape[0]
            if loss_kind == 'axis':
                score = axis_cone_q(x_img, x_anc, sin_psi)
            else:
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
