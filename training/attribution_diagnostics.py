"""Validation, training reports and final aperture/plot diagnostics."""
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoint_io import atomic_torch_save
from geometry.lorentz import exp_map0
from losses.attribution_loss import predict_class
from losses.axis_cone_loss import (
    axis_cone_q, calibrate_axis_apertures, depth_from_sin_psi,
)


@torch.no_grad()


def run_validation(model_inner, val_loader, x_anc, class_names, device, curv,
                   collect: int = 0, predict=None, inside=None) -> dict:
    """Compute accuracy and optional coverage with the supplied decision rule.

    Default prediction is argmin exterior angle. Positive collect values retain
    a per-class quota of embeddings for plotting, without another forward pass."""
    model_inner.eval()

    per_class_correct = {c: 0 for c in class_names}
    per_class_total   = {c: 0 for c in class_names}
    per_class_inside  = {c: 0 for c in class_names} if inside is not None else None
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
        pred_idx = (predict(x_img, x_anc) if predict is not None
                    else predict_class(x_img, x_anc, curv=curv))
        true_idx = torch.tensor([idx_of[g] for g in batch["generator"]],
                                device=device, dtype=torch.long)
        covered = inside(x_img, x_anc, true_idx) if inside is not None else None
        pred_names = [class_names[i] for i in pred_idx.tolist()]
        covered_list = covered.tolist() if covered is not None else None
        for j, (pred, gt) in enumerate(zip(pred_names, batch["generator"])):
            per_class_total[gt] += 1
            per_class_correct[gt] += int(pred == gt)
            if covered_list is not None:
                per_class_inside[gt] += int(covered_list[j])

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
    if per_class_inside is not None:
        per_class_coverage = {
            c: per_class_inside[c] / per_class_total[c] if per_class_total[c] else 0.0
            for c in class_names
        }
        result.update({
            "per_class_coverage": per_class_coverage,
            "balanced_coverage": sum(per_class_coverage.values()) / len(class_names),
            "min_coverage": min(per_class_coverage.values()),
        })
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


def report_epoch(avg, args, epoch, lr, anchors):
    """Print averaged training statistics for the selected loss."""
    if args.loss == "axis":
        print(f"\nEpoch {epoch}: train loss={avg['loss']:.4f}  "
              f"pos={avg['loss_pos']:.4f}  cover={avg['loss_cover']:.4f}  "
              f"ap={avg['loss_ap']:+.4f}  "
              f"neg={avg['loss_neg']:.4f}  sep={avg['loss_sep']:.4f}  "
              f"lr={lr:.2e}")
        if args.lambda_ce > 0:
            print(f"           CE={avg['loss_ce']:.4f}  τ={avg['ce_tau']:.3f}")
        print(f"           acc={100*avg['cone_acc']:.1f}%  "
              f"inside={100*avg['inside_img']:.1f}% (out {100*(1-avg['inside_img']):.1f}%"
              f" vs ν {100*args.nu:.0f}%)  "
              f"viol={avg['viol_mass']:.3f}  "
              f"q_pos={avg['q_pos']:.3f}  "
              f"ψ∈[{avg['psi_min_deg']:.1f},{avg['psi_max_deg']:.1f}]°"
              f" (μ{avg['psi_deg']:.1f}°)  "
              f"min∠={avg['sep_min_deg']:.1f}°  mean∠={avg['sep_mean_deg']:.1f}°  "
              f"overlap={100*avg['sep_overlap']:.1f}%  "
              f"‖a_anc‖={avg['anc_norm']:.2f}"
              + (f"  ‖t_anc‖={anchors.anchor_tangent.norm(dim=-1).mean().item():.2f}"
                 if anchors.anchor_tangent is not None else ""))
    else:
        line1 = (f"\nEpoch {epoch}: train loss={avg['loss']:.4f}  "
                 f"L_img_cls={avg['loss_img_in_cls']:.4f}"
                 f"  (pos={avg['loss_pos']:.4f} neg={avg['loss_neg']:.4f})")
        line1 += f"  L_norm={avg['loss_norm']:.4f}"
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
    """Print overall and per-class validation accuracy and optional coverage."""
    val_line = (f"  val: overall={100*val['overall_acc']:.1f}%  "
                f"balanced={100*val['balanced_acc']:.1f}%")
    if "balanced_coverage" in val:
        val_line += (f"  coverage={100*val['balanced_coverage']:.1f}%  "
                     f"worst={100*val['min_coverage']:.1f}%")
    print(val_line + f"  ({val['total']} samples)")
    for c, a in val["per_class_acc"].items():
        coverage = (f"  cover={100*val['per_class_coverage'][c]:5.1f}%"
                    if "per_class_coverage" in val else "")
        print(f"    {c:14s}: acc={100*a:5.1f}%{coverage}")


def finalize_training(args, model, core, train_ds, val_loader, class_names,
                      name_to_idx, device, out_path, anchors, plot_epoch_snapshot):
    """Calibrate apertures and plot all clean rows using the selected checkpoint.

    Calibration is applied only when every class meets the requested training
    coverage and all candidate cone pairs pass the angular separation check.
    """
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
    plot_sin_psi = best_ckpt.get("anchor_sin_psi")
    if plot_sin_psi is not None:
        plot_sin_psi = plot_sin_psi.to(device)

    if args.calibrate_psi:
        target_coverage = 1.0 - args.nu
        x_train = torch.from_numpy(all_emb)
        y_train = torch.tensor(all_labels)
        required_psi, _ = calibrate_axis_apertures(
            x_train, y_train, x_anc_plot.cpu(), target_coverage,
            inside_margin=math.radians(args.inside_margin),
        )
        calibrated_psi = required_psi.clamp(anchors.psi_lo, anchors.psi_hi)

        dots = (F.normalize(x_train, dim=-1)
                * F.normalize(x_anc_plot.cpu(), dim=-1)[y_train]).sum(-1).clamp(-1, 1)
        padded = torch.arccos(dots) + math.radians(args.inside_margin)
        train_coverage = torch.stack([
            (padded[y_train == k] <= calibrated_psi[k]).float().mean()
            for k in range(len(class_names))
        ])

        axes = F.normalize(x_anc_plot, dim=-1)
        cos = (axes @ axes.T).clamp(-1 + 1e-6, 1 - 1e-6)
        iu = torch.triu_indices(len(class_names), len(class_names), offset=1,
                                device=device)
        sep = torch.arccos(cos[iu[0], iu[1]])
        need = (calibrated_psi.to(device)[iu[0]]
                + calibrated_psi.to(device)[iu[1]]
                + math.radians(args.separation_margin))
        overlap = need > sep + 1e-6
        range_ok = bool((required_psi <= anchors.psi_hi + 1e-7).all().item())
        coverage_ok = bool((train_coverage + 1e-7 >= target_coverage).all().item())
        feasible = range_ok and coverage_ok and not bool(overlap.any().item())

        print(f"\nAperture calibration on {len(all_labels)} best-checkpoint train "
              f"embeddings: target={100*target_coverage:.1f}% + "
              f"{args.inside_margin:g}° inside margin")
        for k, name in enumerate(class_names):
            print(f"  {name:14s} required={math.degrees(required_psi[k].item()):5.1f}°  "
                  f"candidate={math.degrees(calibrated_psi[k].item()):5.1f}°  "
                  f"covered={100*train_coverage[k].item():5.1f}%")
        n_overlap = int(overlap.sum().item())
        print(f"  non-overlap: {len(overlap) - n_overlap}/{len(overlap)} "
              f"pairs feasible")

        report = {
            "applied": feasible,
            "target_coverage": target_coverage,
            "inside_margin_deg": args.inside_margin,
            "required_psi_deg": torch.rad2deg(required_psi).tolist(),
            "candidate_psi_deg": torch.rad2deg(calibrated_psi).tolist(),
            "train_coverage": train_coverage.tolist(),
            "overlapping_pairs": n_overlap,
            "num_train": len(all_labels),
        }
        best_ckpt["aperture_calibration"] = report

        if feasible:
            plot_sin_psi = torch.sin(calibrated_psi).to(device)
            rc = args.curv ** 0.5
            want = torch.asinh(
                rc * depth_from_sin_psi(plot_sin_psi, args.min_radius)) / rc
            calibrated_tangent = (F.normalize(t_anc_plot.float().to(device), dim=-1)
                                  * want.unsqueeze(1))
            x_anc_plot = exp_map0(calibrated_tangent, curv=args.curv)
            best_ckpt["anchor_tangent"] = calibrated_tangent.cpu()
            best_ckpt["anchor_sin_psi"] = plot_sin_psi.cpu()
            val_cal = run_validation(
                core, val_loader, x_anc_plot, class_names, device, args.curv,
                predict=lambda xi_, xa_: axis_cone_q(
                    xi_, xa_, plot_sin_psi).argmin(1),
            )
            best_ckpt["val_balanced_calibrated"] = val_cal["balanced_acc"]
            print(f"  applied: calibrated balanced val="
                  f"{100*val_cal['balanced_acc']:.1f}%")
        else:
            print("  NOT applied: coverage and non-overlap are jointly infeasible; "
                  "the fixed training aperture remains in the checkpoint")
        atomic_torch_save(best_ckpt, out_path)

    if args.plot_all_train:
        plot_epoch_snapshot(
            all_emb, all_labels, x_anc_plot, class_names,
            Path(args.diag_plot_dir) / "train_all_final.png",
            curv=args.curv, min_radius=args.min_radius,
            psi=(torch.arcsin(plot_sin_psi).cpu().numpy()
                 if plot_sin_psi is not None else None),
            state=None, seed=args.seed,
            title=(f"best epoch {best_ckpt['epoch']} · all {len(all_labels)} "
                   "training images"),
        )
