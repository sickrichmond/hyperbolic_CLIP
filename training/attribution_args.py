"""CLI options and startup constraints for attribution training."""
import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path",   required=True)
    p.add_argument("--captions_dir",   required=True)
    p.add_argument("--generators",     nargs="+", default=["real", "FLUX"])
    p.add_argument("--semantics",      nargs="+",
                   default=["COCO", "cat", "dog", "wild", "FFHQ", "celebahq",
                            "bedroom", "church", "classroom", "ImageNet-1k"])
    p.add_argument("--output",         default="attribution_checkpoint_FLUX.pt")
    p.add_argument("--clip_name",      default="openai/clip-vit-base-patch32")
    p.add_argument("--lora_r",         type=int,   default=8)
    p.add_argument("--lora_alpha",     type=int,   default=16)
    p.add_argument("--lora_target",    type=str,   default=None,
                   help="PEFT fullmatch regex for adapter modules; default targets q/v "
                        "projections in both encoders.")
    p.add_argument("--hyperbolic_dim", type=int,   default=128)
    p.add_argument("--init_depth",     type=float, default=0.0,
                   help="Rescale the projection head once to this median image tangent norm (0 "
                        "disables).")
    p.add_argument("--fixed_image_radius", type=float, default=0.0,
                   help="Fix image tangent norms throughout training (0 disables); directions "
                        "remain trainable.")
    p.add_argument("--radial_margin", type=float, default=0.0,
                   help="Minimum tangent-radius gap from the deepest bounded anchor to "
                        "fixed-radius images.")
    p.add_argument("--anchors_only", action="store_true", default=False,
                   help="Freeze CLIP, LoRA and the projection head; optimize free "
                        "anchors.")
    p.add_argument("--init_scale",     type=float, default=0.1,
                   help="Initial scale of the projection head's last layer.")
    p.add_argument("--curv",           type=float, default=1.0)
    p.add_argument("--min_radius",     type=float, default=0.1)
    p.add_argument("--margin",         type=float, default=0.1)
    p.add_argument("--lambda_neg",     type=float, default=1.0)
    p.add_argument("--no_captions", action="store_true", default=False,
                   help="Compatibility flag: training is image-only; this has no effect.")
    p.add_argument("--require_caption", action="store_true", default=False,
                   help="Restrict training rows to images with captions; captions are not encoded.")
    p.add_argument("--anchor_init",
                   choices=["text", "image_centroid", "text_free", "random"],
                   default="text",
                   help="Class anchors: encoded text, projected image centroids, learned text "
                        "offsets, or random directions.")
    p.add_argument("--freeze_anchors", action="store_true", default=False,
                   help="Freeze random or image_centroid anchor parameters.")
    p.add_argument("--anchor_norm_range", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"),
                   help="Clamp free anchor tangent norms after each step.")
    p.add_argument("--anchor_prompts", type=str, default=None,
                   help="JSON map from class name to prompt, replacing the default text templates.")
    p.add_argument("--anchor_drift_init", type=float, default=0.1,
                   help="Initial positive scale s for text_free anchors t=t0+s*delta, with "
                        "delta initialized to zero.")
    p.add_argument("--anchor_init_norm", type=float, default=2.0,
                   help="Initial tangent norm for free anchors (0 keeps the raw scale).")
    p.add_argument("--anchor_init_cache", type=str, default=None,
                   help="Cache path for normalized CLIP class means; reuse only with the same "
                        "backbone and training data.")
    p.add_argument("--train_augment", action="store_true", default=False,
                   help="Enable training-image augmentation; validation images remain clean.")
    p.add_argument("--aug_policy", choices=["corruption", "omnidfa"], default="corruption",
                   help="Training augmentation policy: corruption (JPEG/blur/downsample) or "
                        "omnidfa (JPEG/resize/hflip/RandAugment/blur).")
    p.add_argument("--lambda_norm",    type=float, default=0.0,
                   help="Weight of the spatial anchor-norm regularizer (0 disables).")
    p.add_argument("--target_norm",    type=float, default=0.0,
                   help="Target Lorentz spatial-coordinate norm, not tangent-space norm.")
    p.add_argument("--norm_mode", choices=["floor", "bilateral"], default="floor",
                   help="Penalize norms below the target (floor) or deviations on "
                        "both sides (bilateral).")
    p.add_argument("--neg_samples",    type=int,   default=0,
                   help="Random negatives retained per sample (0 uses all); full pairwise "
                        "scores are still computed.")
    p.add_argument("--lambda_cosine", type=float, default=0.2,
                   help="Weight of mean positive cosine similarity over unique anchor pairs "
                        "(default: 0.2; 0 disables).")
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--num_epochs",     type=int,   default=10)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--lr_min",         type=float, default=1e-6,
                   help="Minimum learning rate for the cosine schedule.")
    p.add_argument("--lr_schedule", choices=["cosine", "constant"], default="cosine",
                   help="Cosine annealing to --lr_min or a constant learning rate, stepped "
                        "after each optimizer update.")
    p.add_argument("--anchor_lr",      type=float, default=None,
                   help="Learning rate for free anchor and drift parameters; defaults "
                        "to --lr, with zero weight decay.")
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                   help="Optimizer for model parameters and geometric parameters: AdamW or "
                        "SGD.")
    p.add_argument("--momentum",       type=float, default=0.9,
                   help="SGD momentum; positive values enable Nesterov. Ignored by AdamW.")
    p.add_argument("--val_frac",       type=float, default=0.2)
    p.add_argument("--split_scheme",   choices=["caption", "stratified"], default="caption",
                   help="Without a manifest: split by caption presence or use a stratified "
                        "train/val/test partition.")
    p.add_argument("--test_frac",      type=float, default=0.1,
                   help="Held-out test fraction for --split_scheme stratified.")
    p.add_argument("--split_manifest", type=str, default=None,
                   help="JSON train/val path lists from the comparison harness; overrides "
                        "internal dataset splitting.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--max_per_class",  type=int,   default=None)
    p.add_argument("--num_workers",    type=int,   default=8)
    p.add_argument("--log_every",      type=int,   default=0,
                   help="Write instantaneous statistics every N steps and at step 1 (0 "
                        "disables); requires --diag_plot_dir.")
    p.add_argument("--snapshot_every", type=int,   default=0,
                   help="Plot the current training batch every N steps (0 disables).")
    p.add_argument("--diag_plot_dir",  type=str,   default=None,
                   help="Directory for fresh per-epoch PCA plots of all clean training images.")
    p.add_argument("--plot_all_train", action="store_true", default=False,
                   help="Plot all clean training rows from the selected checkpoint in "
                        "train_all_final.png; adds one full data pass.")
    return p.parse_args(argv)


def validate_args(args):
    """Validate mode combinations before loading data or model weights."""
    if min(args.fixed_image_radius, args.radial_margin) < 0:
        raise ValueError("radius and radial margin must be non-negative")
    if args.fixed_image_radius > 0 and args.init_depth > 0:
        raise ValueError("Use --fixed_image_radius or --init_depth, not both")
    if args.anchors_only and args.anchor_init == "text":
        raise ValueError("--anchors_only requires a free --anchor_init mode")
    if args.freeze_anchors and args.anchor_init not in ("random", "image_centroid"):
        raise ValueError("--freeze_anchors requires random or image_centroid anchors")
    if args.plot_all_train and not args.diag_plot_dir:
        raise ValueError("--plot_all_train requires --diag_plot_dir")
    if args.fixed_image_radius > 0:
        if args.anchor_init not in ("random", "image_centroid"):
            raise ValueError("--fixed_image_radius needs random or image_centroid anchors "
                             "whose depth is bounded")
        if args.anchor_norm_range:
            max_anchor_radius = args.anchor_norm_range[1]
        else:
            raise ValueError("--fixed_image_radius also needs "
                             "--anchor_norm_range to bound anchor depth")
        required = max_anchor_radius + args.radial_margin
        if args.fixed_image_radius < required:
            raise ValueError(
                f"--fixed_image_radius {args.fixed_image_radius:g} must be at least "
                f"{required:.3f}: deepest anchor {max_anchor_radius:.3f} + "
                f"radial margin {args.radial_margin:g}")
