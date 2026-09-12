"""CLI options and startup constraints for attribution training."""
import argparse
import math


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
                        "anchor/aperture and loss parameters.")
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
                   choices=["text", "image_centroid", "text_free", "random", "simplex"],
                   default="text",
                   help="Class anchors: encoded text, projected image centroids, learned text "
                        "offsets, random directions, or a regular simplex.")
    p.add_argument("--freeze_anchors", action="store_true", default=False,
                   help="Freeze random, image_centroid or simplex anchor directions.")
    p.add_argument("--anchor_norm_range", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"),
                   help="Cone loss: clamp free anchor tangent norms after each step. Axis loss "
                        "instead derives depth from aperture.")
    p.add_argument("--anchor_prompts", type=str, default=None,
                   help="JSON map from class name to prompt, replacing the default text templates.")
    p.add_argument("--anchor_drift_init", type=float, default=0.1,
                   help="Initial positive scale s for text_free anchors t=t0+s*delta, with "
                        "delta initialized to zero.")
    p.add_argument("--anchor_init_norm", type=float, default=2.0,
                   help="Initial tangent norm for free anchors (0 keeps the raw scale); axis "
                        "loss subsequently derives depth from aperture.")
    p.add_argument("--anchor_init_cache", type=str, default=None,
                   help="Cache path for normalized CLIP class means; reuse only with the same "
                        "backbone and training data.")
    p.add_argument("--train_augment", action="store_true", default=False,
                   help="Enable training-image augmentation; validation images remain clean.")
    p.add_argument("--aug_policy", choices=["corruption", "omnidfa"], default="corruption",
                   help="Training augmentation policy: corruption (JPEG/blur/downsample) or "
                        "omnidfa (JPEG/resize/hflip/RandAugment/blur).")
    p.add_argument("--lambda_norm",    type=float, default=0.0,
                   help="Cone loss: weight of the spatial anchor-norm regularizer (0 disables).")
    p.add_argument("--target_norm",    type=float, default=0.0,
                   help="Cone loss: target Lorentz spatial-coordinate norm, not tangent-space norm.")
    p.add_argument("--norm_mode", choices=["floor", "bilateral"], default="floor",
                   help="Cone loss: penalize norms below the target (floor) or deviations on "
                        "both sides (bilateral).")
    p.add_argument("--lambda_sep",     type=float, default=0.0,
                   help="Axis loss: weight of the pairwise angular separation penalty.")
    p.add_argument("--separation_margin", type=float, default=0.0,
                   help="Angular gap in degrees added to pairwise aperture sums for separation "
                        "and calibration checks.")
    p.add_argument("--inside_margin", type=float, default=0.0,
                   help="Axis loss: angular margin in degrees inside the cone wall for coverage "
                        "and aperture calibration.")
    p.add_argument("--lambda_cover", type=float, default=0.0,
                   help="Axis loss: weight of per-class coverage violations; updates image/axis "
                        "directions with aperture detached.")
    p.add_argument("--lambda_center", type=float, default=1.0,
                   help="Axis loss: weight of the mean pull toward the correct axis, with "
                        "aperture detached.")
    p.add_argument("--loss", choices=["cone", "axis"], default="cone",
                   help="cone: exterior-angle objective; axis: normalized chord score "
                        "q=(1-cos(theta))/(1-cos(psi)), with q=1 at the wall.")
    p.add_argument("--psi_range", type=float, nargs=2, default=[5.0, 60.0],
                   metavar=("MIN_DEG", "MAX_DEG"),
                   help="Axis loss: half-aperture bounds in degrees; learned apertures start at "
                        "the midpoint via a sigmoid.")
    p.add_argument("--fixed_psi", type=float, default=0.0, metavar="DEG",
                   help="Axis loss: fixed half-aperture in degrees (0 learns it); requires "
                        "--lambda_aperture 0.")
    p.add_argument("--nu", type=float, default=0.05,
                   help="Axis loss: target outside fraction for the aperture objective, "
                        "validation selection and calibration; not a hard coverage guarantee.")
    p.add_argument("--calibrate_psi", action="store_true", default=False,
                   help="Fit training-angle quantiles plus inside margin after selection; apply "
                        "only if range, coverage and separation checks pass. Requires fixed "
                        "simplex anchors.")
    p.add_argument("--lambda_aperture", type=float, default=1.0,
                   help="Axis loss: weight of log-wall-size plus outside-violation cost divided "
                        "by --nu; updates apertures with directions detached.")
    p.add_argument("--neg_samples",    type=int,   default=0,
                   help="Random negatives retained per sample (0 uses all); full pairwise "
                        "scores are still computed.")
    p.add_argument("--lambda_ce",      type=float, default=0.0,
                   help="Axis loss: cross-entropy weight for logits -q/tau; tau is learned.")
    p.add_argument("--ce_tau_init",    type=float, default=1.0,
                   help="Axis loss: initial positive CE temperature, parameterized with softplus.")
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
                   help="Learning rate for free anchor, drift and aperture parameters; defaults "
                        "to --lr, with zero weight decay.")
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                   help="Optimizer for model/loss parameters and geometric parameters: AdamW or "
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
                   help="Plot the current training batch every N steps (0 disables); fit the "
                        "shared projection before training.")
    p.add_argument("--diag_plot_dir",  type=str,   default=None,
                   help="Directory for epoch validation snapshots; requires HoroPCA. See "
                        "training/poincare.py.")
    p.add_argument("--plot_all_train", action="store_true", default=False,
                   help="Plot all clean training rows from the selected checkpoint in "
                        "train_all_final.png; adds one full data pass.")
    return p.parse_args(argv)


def validate_args(args):
    """Validate mode combinations before loading data or model weights."""
    if min(args.fixed_image_radius, args.radial_margin, args.separation_margin,
           args.inside_margin) < 0:
        raise ValueError("radius and angular margins must be non-negative")
    if args.loss == "cone" and (args.lambda_ce != 0 or args.lambda_sep != 0):
        raise ValueError("--lambda_ce and --lambda_sep are only supported by --loss axis")
    if args.fixed_image_radius > 0 and args.init_depth > 0:
        raise ValueError("Use --fixed_image_radius or --init_depth, not both")
    if args.anchors_only and args.anchor_init == "text":
        raise ValueError("--anchors_only requires a free --anchor_init mode")
    if args.freeze_anchors and args.anchor_init not in ("random", "image_centroid", "simplex"):
        raise ValueError("--freeze_anchors requires random, image_centroid or simplex anchors")
    if args.loss == "axis" and not (0 < args.psi_range[0] <= args.psi_range[1] < 90):
        raise ValueError("--psi_range must satisfy 0 < MIN <= MAX < 90 degrees")
    if args.loss == "axis" and not 0 < args.nu < 1:
        raise ValueError("--nu must be in (0, 1) for the axis coverage objective")
    if args.loss == "axis" and args.inside_margin >= args.psi_range[0]:
        raise ValueError("--inside_margin must be smaller than the minimum --psi_range")
    if args.fixed_psi and args.loss != "axis":
        raise ValueError("--fixed_psi is only defined for --loss axis")
    if args.fixed_psi and not (args.psi_range[0] <= args.fixed_psi <= args.psi_range[1]):
        raise ValueError("--fixed_psi must lie inside --psi_range")
    if args.fixed_psi and args.lambda_aperture != 0:
        raise ValueError("A fixed aperture makes --lambda_aperture inert; set it to 0")
    if args.fixed_psi and args.freeze_anchors and args.lambda_sep != 0:
        raise ValueError("Fixed axes/apertures make --lambda_sep inert; set it to 0")
    if args.calibrate_psi and not (args.loss == "axis" and args.fixed_psi
                                   and args.freeze_anchors
                                   and args.anchor_init == "simplex"):
        raise ValueError("--calibrate_psi requires --loss axis, --fixed_psi, "
                         "--freeze_anchors and --anchor_init simplex")
    if args.plot_all_train and not args.diag_plot_dir:
        raise ValueError("--plot_all_train requires --diag_plot_dir")
    if args.fixed_image_radius > 0:
        if args.anchor_init not in ("random", "image_centroid", "simplex"):
            raise ValueError("--fixed_image_radius needs random, simplex or image_centroid anchors "
                             "whose depth is bounded")
        if args.loss == "axis":
            rc = args.curv ** 0.5
            min_psi = math.radians(args.psi_range[0])
            max_anchor_radius = math.asinh(
                rc * 2.0 * args.min_radius / math.sin(min_psi)) / rc
        elif args.anchor_norm_range:
            max_anchor_radius = args.anchor_norm_range[1]
        else:
            raise ValueError("--fixed_image_radius with --loss cone also needs "
                             "--anchor_norm_range to bound anchor depth")
        required = max_anchor_radius + args.radial_margin
        if args.fixed_image_radius < required:
            raise ValueError(
                f"--fixed_image_radius {args.fixed_image_radius:g} must be at least "
                f"{required:.3f}: deepest anchor {max_anchor_radius:.3f} + "
                f"radial margin {args.radial_margin:g}")
