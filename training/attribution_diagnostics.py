"""Validation, training reports and full-training plots."""
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from geometry.lorentz import exp_map0
from losses.attribution_loss import predict_class


@torch.no_grad()
def run_validation(model_inner, val_loader, x_anc, class_names, device, curv,
                   collect: int = 0) -> dict:
    """Compute accuracy using argmin exterior angle.

    Positive collect values retain a per-class quota of embeddings for plotting,
    without another forward pass."""
    model_inner.eval()

    per_class_correct = {c: 0 for c in class_names}
    per_class_total   = {c: 0 for c in class_names}
    idx_of = {c: i for i, c in enumerate(class_names)}
    emb, emb_lab = [], []
    quota = max(1, collect // len(class_names)) if collect else 0
    taken = {c: 0 for c in class_names}

    for batch in tqdm(val_loader, desc="val", leave=False):
        pixel = batch["pixel_values"].to(device)
        with autocast("cuda"):
            x_img, _ = model_inner.encode_image(pixel)
        if quota and any(n < quota for n in taken.values()):
            keep = []
            for j, g in enumerate(batch["generator"]):
                if taken[g] < quota:
                    taken[g] += 1
                    keep.append(j)
            if keep:
                emb.append(x_img[keep].float().cpu())
                emb_lab.extend(idx_of[batch["generator"][j]] for j in keep)
        pred_idx = predict_class(x_img, x_anc, curv=curv)
        pred_names = [class_names[i] for i in pred_idx.tolist()]
        for pred, gt in zip(pred_names, batch["generator"]):
            per_class_total[gt] += 1
            per_class_correct[gt] += int(pred == gt)

    total = sum(per_class_total.values())
    correct = sum(per_class_correct.values())
    per_class_acc = {
        c: (per_class_correct[c] / per_class_total[c]) if per_class_total[c] > 0 else 0.0
        for c in class_names
    }
    balanced = sum(per_class_acc.values()) / len(class_names)
    result = {
        "overall_acc":   correct / total if total else 0.0,
        "balanced_acc":  balanced,
        "per_class_acc": per_class_acc,
        "total":         total,
        "emb":           torch.cat(emb).numpy() if emb else None,
        "emb_labels":    emb_lab,
    }
    return result


@torch.no_grad()
def collect_plot_embeddings(model, loader, name_to_idx, device):
    """Collect one embedding for every row in an unshuffled plotting loader."""
    model.eval()
    emb, labels = [], []
    for batch in tqdm(loader, desc="all-train plot", leave=False):
        pixel = batch["pixel_values"].to(device)
        with autocast("cuda"):
            emb.append(model(pixel).float().cpu())
        labels.extend(name_to_idx[g] for g in batch["generator"])
    x = torch.cat(emb).numpy()
    if len(x) != len(loader.dataset):
        raise RuntimeError(f"Full-train plot collected {len(x)} of {len(loader.dataset)} rows")
    return x, labels


def report_epoch(avg, epoch, lr, anchors):
    """Print epoch averages of the per-batch entailment-cone statistics.

    L_cosine is unweighted. cos_max_avg averages per-batch pairwise maxima,
    rather than reporting the largest cosine observed during the epoch.
    """
    line1 = (f"\nEpoch {epoch}: train loss={avg['loss']:.4f}  "
             f"L_img_cls={avg['loss_img_in_cls']:.4f}"
             f"  (pos={avg['loss_pos']:.4f} neg={avg['loss_neg']:.4f})")
    line1 += f"  L_norm={avg['loss_norm']:.4f}"
    line1 += (f"  L_cosine={avg['loss_cosine']:.4f}"
              f"  cos_mean={avg['mean_anc_cos_sim']:.4f}"
              f"  cos_max_avg={avg['max_anc_cos_sim']:.4f}")
    line1 += (f"  min∠={avg['sep_min_deg']:.1f}°"
              f"  mean∠={avg['sep_mean_deg']:.1f}°"
              f"  overlap={100*avg['sep_overlap']:.0f}%"
              f"  2ψ/min∠={2*math.degrees(avg['mean_psi_anc'])/max(avg['sep_min_deg'], 1e-6):.1f}")
    if anchors.anchor_drift is not None:
        line1 += f"  drift_s={F.softplus(anchors.anchor_drift).item():.4f}"
    line1 += f"  lr={lr:.2e}"
    print(line1)
    print(f"           cone_acc={100*avg['cone_acc']:.1f}%  "
          f"inside_img={100*avg['inside_img']:.1f}%  "
          f"ψ_anc={avg['mean_psi_anc']:.3f}"
          f"∈[{avg['psi_min_deg']:.1f},{avg['psi_max_deg']:.1f}]°  "
          f"ξ_img→anc={avg['mean_xi_img_anc']:.3f}  "
          f"ξ_sat={100*avg['xi_sat']:.0f}%  "
          f"‖x_anc‖={avg['mean_anc_norm']:.2f}"
          + (f"  ‖t_anc‖={anchors.anchor_tangent.norm(dim=-1).mean().item():.2f}"
             if anchors.anchor_tangent is not None else ""))


def report_validation(val):
    """Print overall and per-class validation accuracy."""
    val_line = (f"  val: overall={100*val['overall_acc']:.1f}%  "
                f"balanced={100*val['balanced_acc']:.1f}%")
    print(val_line + f"  ({val['total']} samples)")
    for c, a in val["per_class_acc"].items():
        print(f"    {c:14s}: acc={100*a:5.1f}%")


def finalize_training(args, model, core, train_ds, class_names,
                      name_to_idx, device, out_path, anchors, plot_epoch_snapshot):
    """Plot all clean training rows using the selected checkpoint."""
    best_ckpt = torch.load(out_path, map_location=device, weights_only=False)
    core.clip.load_state_dict(best_ckpt["lora_state"])
    core.projection.load_state_dict(best_ckpt["projection"])

    train_ds.train_augment = False
    plot_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    all_emb, all_labels = collect_plot_embeddings(
        model, plot_loader, name_to_idx, device,
    )
    with torch.no_grad():
        t_anc_plot = best_ckpt.get("anchor_tangent")
        x_anc_plot = (exp_map0(t_anc_plot.float().to(device), curv=args.curv)
                      if t_anc_plot is not None
                      else core.encode_text(anchors.anchor_ids, anchors.anchor_mask)[0])
    plot_epoch_snapshot(
        all_emb, all_labels, x_anc_plot, class_names,
        Path(args.diag_plot_dir) / "train_all_final.png",
        curv=args.curv,
        title=(f"best epoch {best_ckpt['epoch']} · all {len(all_labels)} "
               "training images"),
    )
