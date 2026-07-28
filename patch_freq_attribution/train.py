"""Train the pixel (image + 3x3 patches) + spectral two-branch hyperbolic attributor.

Adds to patch_attribution.train a second hyperbolic space fed by the centred
log-magnitude FFT, with its own projection head and its own centroid-initialised
anchors. Loss:

    L = L_cone(pixel views) + L_cone(spectrum) + lambda_fuse * CE(fused logits)

The cross-entropy sees the exterior angles DETACHED, so it trains only the two
fusion temperatures — the geometry stays trained by the cone losses alone.
--lambda_fuse 0 leaves both temperatures at their init (a plain sum of logits),
which is the fusion ablation.

Usage: see slurm/slurm_train_22cls_patchfreq.sh
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from geometry.lorentz import exp_map0
from losses.attribution_loss import EntailmentConeLoss
from patch_attribution.train import (
    add_common_args, build_datasets, build_loaders, init_anchors, lift,
    save_checkpoint,
)
from patch_freq_attribution.model import PatchFreqAttributionCLIP, fused_logits
from train_attribution import build_anchors


def parse_args():
    p = add_common_args(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--lambda_fuse", type=float, default=1.0,
                   help="Weight of the cross-entropy on the fused logits. It only "
                        "reaches the two branch temperatures (the angles are "
                        "detached); 0 freezes them at 1.0 = plain sum.")
    p.add_argument("--anchor_init_cache_spec", type=str, default=None,
                   help="Cache for the SPECTRAL class centroids (separate file from "
                        "--anchor_init_cache, which holds the pixel ones).")
    return p.parse_args()


@torch.no_grad()
def run_validation(core, val_loader, x_anc_pix, x_anc_spec, tau, class_names, device):
    """Balanced accuracy under the fused decision rule."""
    core.eval()
    correct = {c: 0 for c in class_names}
    total = {c: 0 for c in class_names}
    for batch in tqdm(val_loader, desc="val", leave=False):
        pixel = batch["pixel_values"].to(device)
        with autocast("cuda"):
            x_views, x_spec = core(pixel)
        pred = fused_logits(x_views.float(), x_spec.float(), x_anc_pix, x_anc_spec,
                            tau, curv=core.curv).argmax(dim=1)
        for p, gt in zip((class_names[i] for i in pred.tolist()), batch["generator"]):
            total[gt] += 1
            correct[gt] += int(p == gt)
    per_class = {c: (correct[c] / total[c]) if total[c] else 0.0 for c in class_names}
    return {
        "overall_acc": sum(correct.values()) / max(sum(total.values()), 1),
        "balanced_acc": sum(per_class.values()) / len(class_names),
        "per_class_acc": per_class,
        "total": sum(total.values()),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_names, _ = build_anchors(args.generators)
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    print(f"Class anchors: image centroids (text-free), {len(class_names)} classes, "
          f"two hyperbolic spaces (pixel + spectrum)")

    train_ds, val_ds = build_datasets(args)
    train_loader, val_loader = build_loaders(args, train_ds, val_ds)

    model = PatchFreqAttributionCLIP(
        clip_name=args.clip_name, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        hyperbolic_dim=args.hyperbolic_dim, curv=args.curv, patch_size=args.patch_size,
    ).to(device)

    # Raw temperatures; exp(0) = 1 → the plain sum of the two logit vectors.
    tau = nn.Parameter(torch.zeros(2, device=device))

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        if ckpt["class_names"] != class_names:
            raise ValueError(f"--init_from was trained on {ckpt['class_names']}")
        model.clip.load_state_dict(ckpt["lora_state"])
        model.projection.load_state_dict(ckpt["projection"])
        model.projection_spec.load_state_dict(ckpt["projection_spec"])
        anchor_pix = nn.Parameter(ckpt["anchor_tangent"].to(device).float())
        anchor_spec = nn.Parameter(ckpt["anchor_tangent_spec"].to(device).float())
        with torch.no_grad():
            tau.copy_(ckpt["fusion_tau"].to(device))
        print(f"Warm start from {args.init_from} "
              f"(epoch {ckpt['epoch']}, val_balanced {100*ckpt['val_balanced']:.1f}%)")
    else:
        print("\n--- pixel anchors ---")
        anchor_pix = init_anchors(model, train_ds, class_names, args, device)
        print("\n--- spectral anchors ---")
        anchor_spec = init_anchors(model, train_ds, class_names, args, device,
                                   feat_fn=model._clip_spectrum,
                                   cache_path=args.anchor_init_cache_spec,
                                   head=model.projection_spec)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    core = model.module if isinstance(model, nn.DataParallel) else model
    core.print_trainable_summary()

    trainable = core.trainable_parameters() + [anchor_pix, anchor_spec, tau]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    opt_steps = (len(train_loader) // args.grad_accum) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(opt_steps, 1), eta_min=1e-6)
    scaler = GradScaler("cuda")
    cone_loss = EntailmentConeLoss(
        curv=args.curv, min_radius=args.min_radius, margin=args.margin,
        lambda_neg=args.lambda_neg, lambda_norm=args.lambda_norm,
        target_norm=args.target_norm,
    )
    print(f"Views/sample=11 (1 global + 9 patches of {args.patch_size}px + 1 spectrum)  "
          f"micro-batch={args.batch_size} x grad_accum={args.grad_accum} "
          f"→ effective batch {args.batch_size * args.grad_accum}  "
          f"λ_fuse={args.lambda_fuse}")

    best_balanced = -1.0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        sums = dict.fromkeys(["loss", "pix", "spec", "fuse", "acc_pix", "acc_spec"], 0.0)
        optimizer.zero_grad()
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for step, batch in enumerate(bar, 1):
            labels = torch.tensor([name_to_idx[g] for g in batch["generator"]],
                                  device=device, dtype=torch.long)
            with autocast("cuda"):
                x_views, x_spec = model(batch["pixel_values"].to(device))
            V = x_views.shape[1]
            x_anc_pix = lift(anchor_pix, args.curv)
            x_anc_spec = lift(anchor_spec, args.curv)

            L_pix, st_pix = cone_loss(x_views.reshape(-1, x_views.shape[-1]),
                                      x_anc_pix, labels.repeat_interleave(V))
            L_spec, st_spec = cone_loss(x_spec, x_anc_spec, labels)
            # Detached angles: the CE calibrates the two temperatures without
            # pulling the representations towards a softmax objective.
            L_fuse = F.cross_entropy(
                fused_logits(x_views.detach().float(), x_spec.detach().float(),
                             x_anc_pix.detach(), x_anc_spec.detach(), tau,
                             curv=args.curv),
                labels)
            loss = L_pix + L_spec + args.lambda_fuse * L_fuse

            scaler.scale(loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            sums["loss"] += loss.item()
            sums["pix"] += L_pix.item()
            sums["spec"] += L_spec.item()
            sums["fuse"] += L_fuse.item()
            sums["acc_pix"] += st_pix["cone_acc"].item()
            sums["acc_spec"] += st_spec["cone_acc"].item()
            if step % 25 == 0:
                bar.set_postfix(loss=f"{sums['loss']/step:.3f}",
                                pix=f"{sums['acc_pix']/step:.3f}",
                                spec=f"{sums['acc_spec']/step:.3f}",
                                τ=f"{tau.exp()[0]:.2f}/{tau.exp()[1]:.2f}")

        n = len(train_loader)
        print(f"\nEpoch {epoch}: loss={sums['loss']/n:.4f}  "
              f"L_pix={sums['pix']/n:.4f}  L_spec={sums['spec']/n:.4f}  "
              f"L_fuse={sums['fuse']/n:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"           view cone_acc pixel={100*sums['acc_pix']/n:.1f}%  "
              f"spectral={100*sums['acc_spec']/n:.1f}%  "
              f"τ_pix={tau.exp()[0]:.3f}  τ_spec={tau.exp()[1]:.3f}")

        core.eval()
        with torch.no_grad():
            x_anc_pix = exp_map0(anchor_pix.float(), curv=args.curv)
            x_anc_spec = exp_map0(anchor_spec.float(), curv=args.curv)
        val = run_validation(core, val_loader, x_anc_pix, x_anc_spec, tau,
                             class_names, device)
        print(f"  val: overall={100*val['overall_acc']:.1f}%  "
              f"balanced={100*val['balanced_acc']:.1f}%  ({val['total']} samples)")
        for c, a in val["per_class_acc"].items():
            print(f"    {c:10s}: {100*a:5.1f}%")

        if val["balanced_acc"] > best_balanced:
            best_balanced = val["balanced_acc"]
            save_checkpoint(out_path, core, anchor_pix, class_names, args, val, epoch,
                            extra={
                                "projection_spec": core.projection_spec.state_dict(),
                                "anchor_tangent_spec": anchor_spec.detach().cpu(),
                                "fusion_tau": tau.detach().cpu(),
                            })
            print(f"  ↳ saved checkpoint (balanced val={100*best_balanced:.1f}%) → {out_path}")

    print(f"\nBest balanced val accuracy: {100*best_balanced:.1f}%  ({out_path})")


if __name__ == "__main__":
    main()
