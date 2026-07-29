"""Train the multi-view (image + 3x3 patch grid) hyperbolic attributor.

Same objective as train_attribution.py with --anchor_init image_centroid, except
every sample contributes 10 views instead of 1 and all of them must land inside
the cone of the sample's generator. The cone loss is untouched: the views are
folded into the batch dimension with repeated labels.

Because ten views cost ten forward passes, the micro-batch must be small; keep
the tuned effective batch of 256 with --grad_accum, and cap the epoch length with
--samples_per_epoch so a run fits the 48h walltime.

Usage (see slurm/slurm_train_22cls_patch.sh for the real invocation):
    python -m patch_attribution.train \\
        --dataset_path $FAST/datasets/iab_dataset \\
        --captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --split_manifest $WORK/hyp_fine_tuning/split_manifest_22cls.json \\
        --generators real FLUX ... --output ckpt.pt
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer

from data.iab_clip_dataset import IABCLIPDataset
from geometry.lorentz import exp_map0, half_aperture
from losses.attribution_loss import EntailmentConeLoss
from patch_attribution.model import PatchAttributionCLIP, view_logits
from train_attribution import build_anchors, class_centroids, make_balanced_sampler


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments shared by this trainer and patch_freq_attribution's.

    Deliberately shorter than train_attribution.py's parser: the caption terms
    are incompatible with free anchors, and these runs always use a split
    manifest, so neither set of flags exists here.
    """
    p.add_argument("--dataset_path",   required=True)
    p.add_argument("--captions_dir",   required=True)
    p.add_argument("--split_manifest", required=True,
                   help="JSON from dump_split_manifest.py: strict data parity with "
                        "the baselines (train on their train, validate on their val).")
    p.add_argument("--generators",     nargs="+", required=True)
    p.add_argument("--semantics",      nargs="+",
                   default=["COCO", "cat", "dog", "wild", "FFHQ", "celebahq",
                            "bedroom", "church", "classroom", "ImageNet-1k"])
    p.add_argument("--output",         required=True)
    p.add_argument("--clip_name",      default="openai/clip-vit-large-patch14")
    p.add_argument("--lora_r",         type=int,   default=16)
    p.add_argument("--lora_alpha",     type=int,   default=32)
    p.add_argument("--hyperbolic_dim", type=int,   default=128)
    p.add_argument("--curv",           type=float, default=1.0)
    p.add_argument("--min_radius",     type=float, default=0.5)
    p.add_argument("--margin",         type=float, default=0.3)
    p.add_argument("--lambda_neg",     type=float, default=1.0)
    p.add_argument("--lambda_norm",    type=float, default=0.0)
    p.add_argument("--target_norm",    type=float, default=0.0)
    p.add_argument("--patch_size",     type=int,   default=112,
                   help="Side of the 3x3 grid crops in the 224px input. 112 = "
                        "overlapping quarter-image windows, 75 = disjoint ninths. "
                        "Only used with --patch_source tensor.")
    p.add_argument("--patch_source",   choices=["tensor", "native"], default="tensor",
                   help="Where the 3x3 grid is cut. 'tensor': out of the 224px "
                        "preprocessed image — free, but the crops hold LESS detail "
                        "than the whole-image view. 'native': out of the "
                        "full-resolution image before CLIP's resize — the crops hold "
                        "~2x the detail of the whole view, at ~2-3x dataloader CPU.")
    p.add_argument("--anchor_init", choices=["text", "image_centroid"],
                   default="image_centroid",
                   help="'image_centroid': free anchors initialised at the per-class "
                        "mean image embedding. 'text': the class templates encoded by "
                        "CLIP at every step, as in train_attribution.py.")
    p.add_argument("--anchor_init_norm",  type=float, default=2.0,
                   help="Common tangent radius the centroid anchors start at; only "
                        "their directions carry information, and the raw radius at "
                        "init would degenerate every cone into a halfspace.")
    p.add_argument("--anchor_init_cache", type=str, default=None,
                   help="Cache of the per-class CLIP-space means. LoRA is zero-init, "
                        "so this is run-independent — the ~20 min pre-pass happens "
                        "once. The cache written by train_attribution.py is reusable.")
    p.add_argument("--init_from",      type=str, default=None,
                   help="Warm-start from a checkpoint of this trainer (weights and "
                        "anchors, fresh optimiser). Ten views per sample will not "
                        "converge inside one 48h job.")
    p.add_argument("--batch_size",     type=int,   default=32,
                   help="MICRO batch. Multiplied by ~10 views before it hits the GPU.")
    p.add_argument("--grad_accum",     type=int,   default=8,
                   help="Optimiser steps every N micro-batches; batch_size*grad_accum "
                        "is the effective batch the learning rate was tuned at (256).")
    p.add_argument("--samples_per_epoch", type=int, default=None,
                   help="Draws per epoch from the balanced sampler (default: the whole "
                        "train split). With ten views a full epoch does not fit the "
                        "walltime — shorten the epoch instead of dropping views.")
    p.add_argument("--num_epochs",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--max_per_class",  type=int,   default=None)
    p.add_argument("--num_workers",    type=int,   default=8)
    return p


def parse_args():
    return add_common_args(argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args()


def build_datasets(args) -> tuple[IABCLIPDataset, IABCLIPDataset]:
    """Train on the harness train split, validate on the harness val split."""
    with open(args.split_manifest) as f:
        man = json.load(f)
    print(f"Split manifest: {len(man['train'])} train + {len(man['val'])} val images")
    common = dict(
        root=args.dataset_path, captions_dir=args.captions_dir,
        generators=args.generators, semantics=args.semantics,
        processor_name=args.clip_name, max_per_class=args.max_per_class,
        split="all", seed=args.seed, require_caption=False,
        patch_grid=(args.patch_source == "native"),
    )
    print("\n=== Train split ===")
    train_ds = IABCLIPDataset(include_paths=set(man["train"]), **common)
    print("\n=== Val split (harness val) ===")
    val_ds = IABCLIPDataset(include_paths=set(man["val"]),
                            include_uncaptioned=True, **common)
    return train_ds, val_ds


def build_loaders(args, train_ds, val_ds) -> tuple[DataLoader, DataLoader]:
    sampler = make_balanced_sampler(train_ds)
    if args.samples_per_epoch:
        sampler.num_samples = args.samples_per_epoch
        print(f"Epoch shortened to {args.samples_per_epoch} draws "
              f"(of {len(train_ds)} train images)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


def init_anchors(core, train_ds, class_names, args, device,
                 feat_fn=None, cache_path=None, head=None,
                 init_norm=None) -> nn.Parameter:
    """Free tangent-space anchors initialised at the per-class embedding centroids.

    feat_fn/head/init_norm default to the pixel branch; the spectral branch passes
    its own (its centroids sit ~7x closer together, so it may need a larger radius
    to get cones that are narrow enough to separate).
    """
    mean_clip = class_centroids(core, train_ds, class_names, args, device,
                                feat_fn=feat_fn, cache_path=cache_path)
    head = head or core.projection
    norm = args.anchor_init_norm if init_norm is None else init_norm
    with torch.no_grad():
        t0 = head(mean_clip.to(device))
        if norm > 0:
            t0 = F.normalize(t0, dim=-1) * norm
        psi0 = half_aperture(exp_map0(t0, curv=args.curv),
                             curv=args.curv, min_radius=args.min_radius)
        # How far apart the anchors actually START. Two cones of half-aperture psi
        # whose axes are less than 2*psi apart overlap from the first step, and the
        # negative term then has a trivial escape (push everything away from every
        # anchor) that satisfies 21 hinges out of 22 while staying at chance
        # accuracy — which is exactly what the spectral branch did for four epochs.
        d = F.normalize(t0, dim=-1)
        cos = (d @ d.T)[~torch.eye(len(t0), dtype=torch.bool, device=t0.device)]
        theta = torch.arccos(cos.clamp(-1, 1))
    print(f"Anchor init: ‖t‖={t0.norm(dim=-1).mean():.2f}  ψ={psi0.mean():.3f} rad  "
          f"pairwise angle mean={theta.mean():.3f} min={theta.min():.3f} rad  "
          f"(K={len(class_names)})")
    if theta.mean() < 2 * psi0.mean():
        print(f"  ⚠️  mean anchor separation < 2ψ — the cones overlap at init. "
              f"Raise the init norm (narrower cones) if this branch stalls at chance.")
    return nn.Parameter(t0.detach().float())


def lift(anchor_tangent: torch.Tensor, curv: float) -> torch.Tensor:
    """Tangent parameters → hyperboloid, in fp32: the hyperbolic ops NaN under fp16."""
    with autocast("cuda", enabled=False):
        return exp_map0(anchor_tangent.float(), curv=curv)


class Anchors:
    """The class anchors, either free tangent parameters or the text templates.

    Text anchors are not parameters: they are re-encoded from the templates at
    every step, so they move as the text encoder's LoRA moves (exactly what
    train_attribution.py does).
    """

    def __init__(self, tangent, tokenizer=None, texts=None, device="cuda", curv=1.0):
        self.tangent, self.curv = tangent, curv
        self.params = [tangent] if tangent is not None else []
        self.ids = self.mask = None
        if tangent is None:
            tok = tokenizer(texts, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=77)
            self.ids = tok["input_ids"].to(device)
            self.mask = tok["attention_mask"].to(device)

    @property
    def mode(self) -> str:
        return "image_centroid" if self.tangent is not None else "text"

    def points(self, core) -> torch.Tensor:
        """(K, D) on the hyperboloid, under the CURRENT weights."""
        if self.tangent is not None:
            return lift(self.tangent, self.curv)
        x_anc, _ = core.encode_text(self.ids, self.mask)
        return x_anc


@torch.no_grad()
def run_validation(core, val_loader, x_anc, class_names, device) -> dict:
    """Balanced accuracy under the multi-view decision rule (mean of -xi)."""
    core.eval()
    correct = {c: 0 for c in class_names}
    total = {c: 0 for c in class_names}
    for batch in tqdm(val_loader, desc="val", leave=False):
        with autocast("cuda"):
            x = core.encode_views(batch["pixel_values"].to(device))
        pred = view_logits(x.float(), x_anc, curv=core.curv).argmax(dim=1)
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


def save_checkpoint(path, core, anchors, class_names, args, val, epoch, extra=None):
    torch.save(
        {
            "lora_state":     core.clip.state_dict(),
            "projection":     core.projection.state_dict(),
            "clip_name":      args.clip_name,
            "lora_r":         args.lora_r,
            "lora_alpha":     args.lora_alpha,
            "hyperbolic_dim": args.hyperbolic_dim,
            "curv":           args.curv,
            "min_radius":     args.min_radius,
            "patch_size":     args.patch_size,
            "patch_source":   args.patch_source,
            "class_names":    class_names,
            "anchor_init":    anchors.mode,
            # Centroid mode only: tangent-space anchors in class_names order, which
            # inference lifts. None in text mode, and test_hypclip.load_anchors then
            # falls back to re-encoding the templates.
            "anchor_tangent": (anchors.tangent.detach().cpu()
                               if anchors.tangent is not None else None),
            "generators":     args.generators,
            "semantics":      args.semantics,
            "val_balanced":   val["balanced_acc"],
            "epoch":          epoch,
            **(extra or {}),
        },
        path,
    )


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_names, anchor_texts = build_anchors(args.generators)
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    print(f"Class anchors: {args.anchor_init}, {len(class_names)} classes")

    train_ds, val_ds = build_datasets(args)
    train_loader, val_loader = build_loaders(args, train_ds, val_ds)

    model = PatchAttributionCLIP(
        clip_name=args.clip_name, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        hyperbolic_dim=args.hyperbolic_dim, curv=args.curv, patch_size=args.patch_size,
    ).to(device)

    tangent = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        if ckpt["class_names"] != class_names:
            raise ValueError(f"--init_from was trained on {ckpt['class_names']}")
        model.clip.load_state_dict(ckpt["lora_state"])
        model.projection.load_state_dict(ckpt["projection"])
        if ckpt["anchor_tangent"] is not None:
            tangent = nn.Parameter(ckpt["anchor_tangent"].to(device).float())
        print(f"Warm start from {args.init_from} "
              f"(epoch {ckpt['epoch']}, val_balanced {100*ckpt['val_balanced']:.1f}%)")
    elif args.anchor_init == "image_centroid":
        tangent = init_anchors(model, train_ds, class_names, args, device)
    anchors = Anchors(tangent, CLIPTokenizer.from_pretrained(args.clip_name),
                      anchor_texts, device, args.curv)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    core = model.module if isinstance(model, nn.DataParallel) else model
    core.print_trainable_summary()

    trainable = core.trainable_parameters() + anchors.params
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps_per_epoch * args.num_epochs, 1), eta_min=1e-6)
    scaler = GradScaler("cuda")
    cone_loss = EntailmentConeLoss(
        curv=args.curv, min_radius=args.min_radius, margin=args.margin,
        lambda_neg=args.lambda_neg, lambda_norm=args.lambda_norm,
        target_norm=args.target_norm,
    )
    grid = (f"9 patches of {args.patch_size}px from the 224 tensor"
            if args.patch_source == "tensor" else
            "9 half-size patches from the full-resolution image")
    # The optimizer-step count is the number that actually decides whether a run
    # converges; printing it makes an over-large grad_accum obvious immediately.
    print(f"Views/sample=10 (1 global + {grid})  "
          f"micro-batch={args.batch_size} x grad_accum={args.grad_accum} "
          f"→ effective batch {args.batch_size * args.grad_accum}, "
          f"{steps_per_epoch} optimizer steps/epoch "
          f"({steps_per_epoch * args.num_epochs} total)")

    best_balanced = -1.0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stat_keys = ["loss_img_in_cls", "cone_acc", "inside_img", "mean_psi_anc",
                 "mean_xi_img_anc", "mean_anc_norm"]

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        sums = {"loss": 0.0, **{k: 0.0 for k in stat_keys}}
        optimizer.zero_grad()
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for step, batch in enumerate(bar, 1):
            labels = torch.tensor([name_to_idx[g] for g in batch["generator"]],
                                  device=device, dtype=torch.long)
            with autocast("cuda"):
                x = model(batch["pixel_values"].to(device))            # (B, V, D)
                x_anc = anchors.points(core)
            V = x.shape[1]
            # Every view is an independent sample of its class: fold the view axis
            # into the batch and repeat the labels. The cone loss needs no change.
            loss, stats = cone_loss(x.reshape(-1, x.shape[-1]), x_anc,
                                    labels.repeat_interleave(V))

            scaler.scale(loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            sums["loss"] += loss.item()
            for k in stat_keys:
                sums[k] += stats[k].item()
            if step % 25 == 0:
                bar.set_postfix(loss=f"{sums['loss']/step:.3f}",
                                acc=f"{sums['cone_acc']/step:.3f}",
                                ψa=f"{sums['mean_psi_anc']/step:.3f}")

        avg = {k: v / len(train_loader) for k, v in sums.items()}
        print(f"\nEpoch {epoch}: train loss={avg['loss']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"           view cone_acc={100*avg['cone_acc']:.1f}%  "
              f"inside_img={100*avg['inside_img']:.1f}%  "
              f"ψ_anc={avg['mean_psi_anc']:.3f}  "
              f"ξ_view→anc={avg['mean_xi_img_anc']:.3f}  "
              f"‖t̄_anc‖={avg['mean_anc_norm']:.2f}")

        # eval() first: text anchors must not be encoded with LoRA dropout active.
        core.eval()
        with torch.no_grad():
            x_anc_val = anchors.points(core)
        val = run_validation(core, val_loader, x_anc_val, class_names, device)
        print(f"  val: overall={100*val['overall_acc']:.1f}%  "
              f"balanced={100*val['balanced_acc']:.1f}%  ({val['total']} samples)")
        for c, a in val["per_class_acc"].items():
            print(f"    {c:10s}: {100*a:5.1f}%")

        if val["balanced_acc"] > best_balanced:
            best_balanced = val["balanced_acc"]
            save_checkpoint(out_path, core, anchors, class_names, args, val, epoch)
            print(f"  ↳ saved checkpoint (balanced val={100*best_balanced:.1f}%) → {out_path}")

    print(f"\nBest balanced val accuracy: {100*best_balanced:.1f}%  ({out_path})")


if __name__ == "__main__":
    main()
