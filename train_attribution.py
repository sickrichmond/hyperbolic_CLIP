"""Train a hyperbolic image-attribution classifier with the entailment-cone hinge loss.

Supports text and free class anchors, image-only training, validation-based
checkpoint selection, and Poincare diagnostics. The pairwise positive-cosine
anchor penalty is enabled by default with weight 0.2; --lambda_cosine 0 disables it.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoint_io import atomic_torch_save
from data.iab_clip_dataset import IABCLIPDataset
from geometry.lorentz import exp_map0
from models.attribution_clip import AttributionCLIP
from losses.attribution_loss import EntailmentConeLoss
from training.attribution_args import parse_args, validate_args
from training.anchors import AnchorState, build_anchors, make_balanced_sampler, calibrate_image_depth
from training.attribution_diagnostics import (
    run_validation, report_epoch, report_validation, finalize_training,
)


def main():
    args = parse_args()
    validate_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"Seed: {args.seed}")

    class_names, anchor_texts = build_anchors(args.generators, args.anchor_prompts)
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    if args.anchor_init == "image_centroid":
        print(f"Class anchors: image centroids (text-free), {len(class_names)} classes")
        for i, c in enumerate(class_names):
            print(f"  [{i}] {c}")
    else:
        src = args.anchor_prompts or "default templates"
        print(f"Class anchors ({args.anchor_init}, from {src}):")
        for i, (c, t) in enumerate(zip(class_names, anchor_texts)):
            print(f"  [{i}] {c:14s} → \"{t}\"")

    req_cap = args.require_caption
    train_include = val_include = None
    if args.split_manifest:
        import json
        with open(args.split_manifest) as f:
            man = json.load(f)
        train_include = set(man["train"])
        val_include = set(man["val"])
        print(f"Split manifest: {len(train_include)} train + {len(val_include)} val images "
              f"(require_caption={req_cap}; harness val used for model selection)")

    dataset_kwargs = dict(
        root=args.dataset_path, captions_dir=args.captions_dir,
        generators=args.generators, semantics=args.semantics,
        processor_name=args.clip_name, max_per_class=args.max_per_class,
        seed=args.seed,
    )
    print("\n=== Train split ===")
    if train_include is not None:
        train_ds = IABCLIPDataset(
            **dataset_kwargs,
            split="all",
            include_paths=train_include, require_caption=req_cap,
        )
        print("\n=== Val split (harness val) ===")
        val_ds = IABCLIPDataset(
            **dataset_kwargs,
            split="all",
            include_paths=val_include, require_caption=False, include_uncaptioned=True,
        )
    else:
        train_ds = IABCLIPDataset(
            **dataset_kwargs,
            split="train", val_frac=args.val_frac,
            split_scheme=args.split_scheme, test_frac=args.test_frac,
            require_caption=req_cap,
        )
        print("\n=== Val split ===")
        val_ds = IABCLIPDataset(
            **dataset_kwargs,
            split="val", val_frac=args.val_frac,
            include_uncaptioned=True, split_scheme=args.split_scheme, test_frac=args.test_frac,
            require_caption=False,
        )

    train_ds.train_augment = args.train_augment
    train_ds.aug_policy = args.aug_policy
    if args.train_augment:
        what = ("random JPEG / blur / downsample" if args.aug_policy == "corruption"
                else "OmniDFA Table 8 (JPEG 75-95 / resize / hflip / RandAugment / blur)")
        print(f"Train-time augmentation: ON  policy={args.aug_policy}  ({what}) — "
              "results are NOT head-to-head comparable with the baselines.")

    sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = AttributionCLIP(
        clip_name=args.clip_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        hyperbolic_dim=args.hyperbolic_dim,
        curv=args.curv,
        lora_target=args.lora_target,
        init_scale=args.init_scale,
        image_radius=args.fixed_image_radius,
    ).to(device)

    anchors = AnchorState(model, train_ds, class_names, anchor_texts, args, device)

    calibrate_image_depth(model, train_loader, anchors, args, device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    core = model.module if isinstance(model, nn.DataParallel) else model
    core.print_trainable_summary()

    cone_loss = EntailmentConeLoss(
        curv=args.curv, min_radius=args.min_radius,
        margin=args.margin, lambda_neg=args.lambda_neg,
        lambda_norm=args.lambda_norm, target_norm=args.target_norm,
        lambda_cosine=args.lambda_cosine,
        norm_mode=args.norm_mode, neg_samples=args.neg_samples,
    ).to(device)

    # Geometric tensors use their own LR and no weight decay.
    backbone = core.trainable_parameters()
    geometric = anchors.parameters()
    anchor_lr = args.lr if args.anchor_lr is None else args.anchor_lr
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": args.lr,
                       "weight_decay": args.weight_decay})
    if geometric:
        groups.append({"params": geometric, "lr": anchor_lr, "weight_decay": 0.0})
    if not groups:
        raise ValueError("No trainable parameters: use a free --anchor_init mode or "
                         "disable --anchors_only")
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            groups,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.momentum > 0,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    print(f"Optimizer: {args.optimizer}"
          + (f" (momentum {args.momentum}, nesterov)" if args.optimizer == "sgd" else "")
          + f"  lr={args.lr}  weight_decay={args.weight_decay}"
          + (f"  |  {len(geometric)} geometric tensors at lr={anchor_lr} wd=0"
             if geometric else ""))
    steps_per_epoch = len(train_loader)
    if args.lr_schedule == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        print(f"LR schedule: constant at {args.lr} for all "
              f"{args.num_epochs * steps_per_epoch} steps")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs * steps_per_epoch, eta_min=args.lr_min
        )
        print(f"LR schedule: cosine {args.lr} → {args.lr_min}")
    scaler = GradScaler("cuda")
    print("Image-only attribution training.")

    plot_epoch_snapshot = None
    diag_state = None
    snap_every = 0
    if args.diag_plot_dir:
        from training.poincare import plot_epoch_snapshot, _load_horopca
        _load_horopca()
        Path(args.diag_plot_dir).mkdir(parents=True, exist_ok=True)
        snap_every = args.snapshot_every
        print(f"Per-epoch Poincare snapshots → {args.diag_plot_dir}"
              + (f" (+ every {snap_every} steps)" if snap_every else ""))
        if args.plot_all_train:
            print("Final Poincare snapshot will include every clean training image "
                  "(one extra pass after training).")

    stat_csv = None
    stat_csv_keys: list[str] = []
    if args.log_every > 0:
        if not args.diag_plot_dir:
            raise ValueError("--log_every needs --diag_plot_dir to write stats.csv into")
        stat_csv = open(Path(args.diag_plot_dir) / "stats.csv", "w", encoding="utf-8")
        print(f"Step-level stats every {args.log_every} steps → "
              f"{Path(args.diag_plot_dir) / 'stats.csv'}")

    best_balanced = -1.0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_keys = ["loss_img_in_cls", "loss_pos", "loss_neg", "xi_sat",
                 "psi_min_deg", "psi_max_deg",
                 "sep_min_deg", "sep_mean_deg", "sep_overlap",
                 "loss_norm", "loss_cosine", "mean_anc_cos_sim", "max_anc_cos_sim",
                 "cone_acc", "inside_img", "mean_psi_anc", "mean_xi_img_anc",
                 "mean_anc_norm"]
    stat_csv_keys = base_keys

    if snap_every and plot_epoch_snapshot is not None:
        model.eval()
        with torch.no_grad():
            e0, l0 = [], []
            for probe0 in train_loader:
                e0.append(core.encode_image(probe0["pixel_values"].to(device))[0]
                          .float().cpu())
                l0.extend(name_to_idx[g] for g in probe0["generator"])
                if sum(t.shape[0] for t in e0) >= 1500:
                    break
            t_now0 = anchors.tangent()
            xa0 = (exp_map0(t_now0.float(), curv=args.curv) if t_now0 is not None
                   else core.encode_text(anchors.anchor_ids, anchors.anchor_mask)[0])
            diag_state = plot_epoch_snapshot(
                torch.cat(e0).numpy(), l0, xa0, class_names,
                Path(args.diag_plot_dir) / "step_0000000.png",
                curv=args.curv, min_radius=args.min_radius,
                state=None, seed=args.seed, title="init (step 0)")
        if not args.anchors_only:
            model.train()

    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        if args.anchors_only:
            model.eval()
        else:
            model.train()
        sums = {"loss": 0.0, **{k: 0.0 for k in stat_csv_keys}}
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for step, batch in enumerate(bar, 1):
            global_step += 1
            pixel    = batch["pixel_values"].to(device)
            labels   = torch.tensor([name_to_idx[g] for g in batch["generator"]],
                                    device=device, dtype=torch.long)

            with autocast("cuda"):
                x_img = model(pixel)
                t_anc = anchors.tangent()
                if t_anc is None:
                    x_anc, _ = core.encode_text(anchors.anchor_ids, anchors.anchor_mask)
            if t_anc is not None:
                # Hyperbolic lifting requires float32 even during mixed-precision training.
                with autocast("cuda", enabled=False):
                    x_anc = exp_map0(t_anc.float(), curv=args.curv)
            loss, stats = cone_loss(x_img, x_anc, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if backbone:
                nn.utils.clip_grad_norm_(backbone, 1.0)
            if geometric:
                nn.utils.clip_grad_norm_(geometric, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            anchors.project_(args)

            sums["loss"] += loss.item()
            for k in stat_csv_keys:
                sums[k] += stats[k].item()

            if stat_csv is not None and (global_step % args.log_every == 0
                                         or global_step == 1):
                row = {"step": global_step, "epoch": epoch,
                       "lr": scheduler.get_last_lr()[0], "loss": loss.item(),
                       **{k: stats[k].item() for k in stat_csv_keys}}
                if stat_csv.tell() == 0:
                    stat_csv.write(",".join(row) + "\n")
                stat_csv.write(",".join(f"{v:.6g}" for v in row.values()) + "\n")
                stat_csv.flush()      # a killed job keeps everything up to the kill

            if (snap_every and plot_epoch_snapshot is not None
                    and global_step % snap_every == 0):
                with torch.no_grad():
                    diag_state = plot_epoch_snapshot(
                        x_img.detach().float().cpu().numpy(), labels.tolist(),
                        x_anc.detach(), class_names,
                        Path(args.diag_plot_dir) / f"step_{global_step:07d}.png",
                        curv=args.curv, min_radius=args.min_radius,
                        state=diag_state, seed=args.seed,
                        title=f"epoch {epoch} · step {global_step}")

            if step % 25 == 0 or step == steps_per_epoch:
                post = {
                    "loss": f"{sums['loss']/step:.3f}",
                    "ic":   f"{sums['loss_img_in_cls']/step:.3f}",
                    "acc":  f"{sums['cone_acc']/step:.3f}",
                    "ψa":   f"{sums['mean_psi_anc']/step:.3f}",
                }
                bar.set_postfix(**post)

        avg = {k: v / steps_per_epoch for k, v in sums.items()}
        report_epoch(avg, epoch, scheduler.get_last_lr()[0], anchors)

        # Re-encode text anchors with dropout disabled for validation.
        core.eval()
        with torch.no_grad():
            t_anc_val = anchors.tangent()
            if t_anc_val is None:
                x_anc_val, _ = core.encode_text(anchors.anchor_ids, anchors.anchor_mask)
            else:
                x_anc_val = exp_map0(t_anc_val.float(), curv=args.curv)
        val = run_validation(core, val_loader, x_anc_val, class_names, device, args.curv,
                             collect=4000 if plot_epoch_snapshot else 0)
        report_validation(val)

        if plot_epoch_snapshot is not None:
            diag_state = plot_epoch_snapshot(
                val["emb"], val["emb_labels"], x_anc_val, class_names,
                Path(args.diag_plot_dir) / f"epoch_{epoch:02d}.png",
                curv=args.curv, min_radius=args.min_radius,
                state=diag_state, seed=args.seed,
                title=f"epoch {epoch}")

        if val["balanced_acc"] > best_balanced:
            best_balanced = val["balanced_acc"]
            atomic_torch_save(
                {
                    "lora_state":      core.clip.state_dict(),
                    "projection":      core.projection.state_dict(),
                    "clip_name":       args.clip_name,
                    "lora_r":          args.lora_r,
                    "lora_alpha":      args.lora_alpha,
                    "lora_target":     args.lora_target,
                    "hyperbolic_dim":  args.hyperbolic_dim,
                    "init_scale":      args.init_scale,
                    "init_depth":      args.init_depth,
                    "image_radius":    args.fixed_image_radius,
                    "radial_margin":   args.radial_margin,
                    "anchors_only":    args.anchors_only,
                    "curv":            args.curv,
                    "min_radius":      args.min_radius,
                    "class_names":     class_names,
                    "anchor_texts":    anchor_texts,
                    "anchor_init":     args.anchor_init,
                    "anchor_prompts":  args.anchor_prompts,
                    "anchor_norm_range": args.anchor_norm_range,
                    "anchor_tangent":  (t_anc_val.detach().cpu()
                                        if t_anc_val is not None else None),
                    "anchor_drift":    (F.softplus(anchors.anchor_drift).item()
                                        if anchors.anchor_drift is not None else None),
                    "loss":            "cone",
                    "lr_min":          args.lr_min,
                    "freeze_anchors":  args.freeze_anchors,
                    "pos_mode":        "hinge",
                    "neg_samples":     args.neg_samples,
                    "optimizer":       args.optimizer,
                    "momentum":        args.momentum,
                    "lr":              args.lr,
                    "norm_mode":       args.norm_mode,
                    "target_norm":     args.target_norm,
                    "lambda_norm":     args.lambda_norm,
                    "lambda_cosine":   args.lambda_cosine,
                    "theta_max":       150.0,
                    "hierarchy":         "none",
                    "family_names":      [],
                    "family_of":         None,
                    "target_norm_family": 0.0,
                    "lambda_family":     0.0,
                    "fam_tau":           None,
                    "train_augment":   args.train_augment,
                    "aug_policy":      args.aug_policy,
                    "generators":      args.generators,
                    "semantics":       args.semantics,
                    "val_balanced":    val["balanced_acc"],
                    "epoch":           epoch,
                },
                out_path,
            )
            print(f"  ↳ saved checkpoint (balanced val={100*best_balanced:.1f}%) → {out_path}")

    if args.plot_all_train:
        finalize_training(args, model, core, train_ds, class_names,
                          name_to_idx, device, out_path, anchors, plot_epoch_snapshot)

    if stat_csv is not None:
        stat_csv.close()
        print(f"Step-level trace: {Path(args.diag_plot_dir) / 'stats.csv'}")

    print(f"\nBest balanced val accuracy: {100*best_balanced:.1f}%  ({out_path})")

if __name__ == "__main__":
    main()
