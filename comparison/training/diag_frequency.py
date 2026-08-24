"""
Frequency / degradation DIAGNOSTIC for the hyperbolic-CLIP attributor — NO training.

Answers two questions about why ours collapses under degradation (esp. JPEG):

  (1) High-frequency reliance. We sweep a pure low-pass (Gaussian blur, sigma
      0.5..5) and a JPEG-quality ramp (q 90..30) over the SAME clean test images
      and plot accuracy vs strength. If accuracy falls off a cliff under gentle
      low-pass, the model is leaning on high-frequency generator fingerprints.
      Comparing the blur curve to the JPEG curve separates "removed high freq"
      (blur) from "added 8x8 block artifacts" (JPEG-specific, out-of-distribution).

  (2) Where the errors go. For each condition we use the HiFi family hierarchy
      (dataset._HIFI_HIERARCHY level-3, 6 families) to route every synthetic-image
      error into: -> real (all synthetic signal gone), -> same family (only the
      fine model id is lost, the family cue survives), or -> cross family. Plus the
      recall on real images (do reals stay real?).

It reuses test_hypclip's model/anchor machinery so the decision rule is identical
(logits = -xi, argmax reproduces the cone prediction). The test images/split are
byte-identical to the baseline eval; degradation is injected by monkeypatching
ImageAttributionDataset.get_degraded_img, so the SAME test loader is enumerated
once and re-scanned per condition (cheap).

Usage (CINECA, via SLURM — see slurm/slurm_diag_frequency.sh):
    python -m comparison.training.diag_frequency \\
        --checkpoint $WORK/hyp_fine_tuning/checkpoints/attribution_22cls_base_vitl14.pt \\
        --root_dir   $FAST/datasets/iab_dataset \\
        --log_dir    $WORK/outputs/hypclip_diag_22cls
"""
import os
import io
import json
import argparse
import datetime

import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

from comparison.dataset.ImageAttributionDataset.dataloader import get_dataloader
from comparison.dataset.ImageAttributionDataset import dataset as ds_mod
from comparison.dataset.ImageAttributionDataset.dataset import (
    ImageAttributionDataset, model_class_to_label, _HIFI_HIERARCHY,
)
from comparison.training.metrics.base_metrics_class import calculate_metrics_for_test
from comparison.training.test_hypclip import build_anchor_texts

from models.attribution_clip import AttributionCLIP
from geometry.lorentz import oxy_angle
from transformers import CLIPTokenizer


# ---- controllable degradation, injected via monkeypatch (fork-safe globals) ----
_OP_KIND = 'identity'      # 'identity' | 'blur' | 'jpeg'
_OP_PARAM = 0.0


def _apply_op(image):
    if _OP_KIND == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=_OP_PARAM))
    if _OP_KIND == 'jpeg':
        img = image.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=int(_OP_PARAM))
        buf.seek(0)
        return Image.open(buf)
    return image


def _patched_get_degraded_img(self, image):
    # Ignores self.degraded; the active op comes from the module globals above.
    return _apply_op(image)


# Conditions: (name, kind, param). clean first; blur/jpeg sweeps include the
# benchmark's own points (blur 3/5, jpeg 65/30) as sanity anchors vs the real eval.
CONDITIONS = [
    ('clean',      'identity', 0.0),
    ('blur0.5',    'blur',     0.5),
    ('blur1.0',    'blur',     1.0),
    ('blur1.5',    'blur',     1.5),
    ('blur2.0',    'blur',     2.0),
    ('blur3.0',    'blur',     3.0),   # == benchmark L5
    ('blur5.0',    'blur',     5.0),   # == benchmark L6
    ('jpeg90',     'jpeg',     90),
    ('jpeg75',     'jpeg',     75),
    ('jpeg65',     'jpeg',     65),    # == benchmark L3
    ('jpeg50',     'jpeg',     50),
    ('jpeg30',     'jpeg',     30),    # == benchmark L4
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root_dir', required=True)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--num_images_per_semantic_per_class', '-n', type=int, default=2000)
    p.add_argument('--log_dir', type=str, default='./logs_diag_frequency')
    return p.parse_args()


@torch.no_grad()
def encode_anchors(model, clip_name, device):
    anchor_texts = build_anchor_texts()
    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    tok = tokenizer(anchor_texts, return_tensors='pt', padding='max_length',
                    truncation=True, max_length=77)
    x_anc, _ = model.encode_text(tok['input_ids'].to(device),
                                 tok['attention_mask'].to(device))
    return x_anc


@torch.no_grad()
def run_condition(model, x_anc, curv, loader, device):
    K = x_anc.shape[0]
    all_logits, all_labels, all_sem = [], [], []
    for batch in loader:
        pixel = batch['image'].to(device)
        x_img, _ = model.encode_image(pixel)
        B = x_img.shape[0]
        x_anc_t = x_anc.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1)
        x_img_t = x_img.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        xi = oxy_angle(x_anc_t, x_img_t, curv=curv).reshape(B, K)
        all_logits.append((-xi).cpu())
        all_labels.append(batch['label'].cpu())
        all_sem.append(batch['semantic_label'].cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    sem = torch.cat(all_sem, dim=0)
    return logits, labels, sem


def error_routing(logits, labels):
    """Where predictions land, using the level-3 HiFi family (6 groups)."""
    idx_to_name = {v: k for k, v in model_class_to_label.items()}
    real_label = model_class_to_label['real']
    family = {i: _HIFI_HIERARCHY[idx_to_name[i]][2] for i in idx_to_name}

    pred = logits.argmax(dim=1)
    labels = labels.view(-1)
    fam_pred = torch.tensor([family[int(p)] for p in pred])
    fam_true = torch.tensor([family[int(l)] for l in labels])

    real_mask = labels == real_label
    synth_mask = ~real_mask
    n_synth = int(synth_mask.sum())

    correct = pred == labels
    to_real = pred == real_label

    out = {
        'real_recall': float((correct & real_mask).sum() / max(1, int(real_mask.sum()))),
        'synth_acc': float((correct & synth_mask).sum() / max(1, n_synth)),
    }
    if n_synth:
        wrong_synth = synth_mask & (~correct)
        n_wrong = int(wrong_synth.sum())
        same_fam_wrong = wrong_synth & (fam_pred == fam_true) & (~to_real)
        to_real_wrong = wrong_synth & to_real
        cross_fam_wrong = wrong_synth & (fam_pred != fam_true) & (~to_real)
        # fractions over ALL synthetic images (so acc + the three routes sum to 1)
        out['synth_err_to_real'] = float(to_real_wrong.sum() / n_synth)
        out['synth_err_same_family'] = float(same_fam_wrong.sum() / n_synth)
        out['synth_err_cross_family'] = float(cross_fam_wrong.sum() / n_synth)
        # and the same three normalized over the errors only (routing of mistakes)
        if n_wrong:
            out['err_route_to_real'] = float(to_real_wrong.sum() / n_wrong)
            out['err_route_same_family'] = float(same_fam_wrong.sum() / n_wrong)
            out['err_route_cross_family'] = float(cross_fam_wrong.sum() / n_wrong)
    return out


def main():
    global _OP_KIND, _OP_PARAM
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    clip_name = ckpt['clip_name']
    curv = ckpt.get('curv', 1.0)
    model = AttributionCLIP.from_checkpoint(ckpt).to(device)
    model.clip.load_state_dict(ckpt['lora_state'])
    model.projection.load_state_dict(ckpt['projection'])
    model.eval()

    x_anc = encode_anchors(model, clip_name, device)

    # Inject the controllable degradation, then build the test loader ONCE.
    ImageAttributionDataset.get_degraded_img = _patched_get_degraded_img
    config = {'model_name': 'hypclip', 'clip_name': clip_name,
              'num_classes': int(x_anc.shape[0])}
    _OP_KIND, _OP_PARAM = 'identity', 0.0
    _, _, test_loader = get_dataloader(
        root_dir=args.root_dir,
        model_name='hypclip',
        num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,
        batch_size=args.batch_size,
        degraded=0,
        config=config,
        num_workers=args.num_workers,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    results = []
    for name, kind, param in CONDITIONS:
        _OP_KIND, _OP_PARAM = kind, param      # workers fork with these globals
        logits, labels, sem = run_condition(model, x_anc, curv, test_loader, device)
        auc, acc, ap, conf_matrix, semantic_acc, extra = \
            calculate_metrics_for_test(labels, logits, sem, need_softmax=True)
        route = error_routing(logits, labels)
        row = {'cond': name, 'kind': kind, 'param': param,
               'acc': float(acc), 'auc': float(auc), 'ap': float(ap), **route}
        results.append(row)
        print(f"[{name:9s}] acc={acc:.3f} auc={auc:.3f} "
              f"real_recall={route['real_recall']:.3f} "
              f"synth_acc={route['synth_acc']:.3f} "
              f"->real={route.get('synth_err_to_real', 0):.3f} "
              f"->samefam={route.get('synth_err_same_family', 0):.3f} "
              f"->crossfam={route.get('synth_err_cross_family', 0):.3f}")

    out_json = os.path.join(args.log_dir, 'diag_frequency.json')
    with open(out_json, 'w') as f:
        json.dump({'checkpoint': args.checkpoint,
                   'when': str(datetime.datetime.now()),
                   'n_classes': int(x_anc.shape[0]),
                   'results': results}, f, indent=2)

    # Human-readable summary tables.
    out_txt = os.path.join(args.log_dir, 'diag_frequency.txt')
    with open(out_txt, 'w') as f:
        f.write("=== Diagnostic 1: accuracy vs degradation strength ===\n")
        f.write(f"{'cond':10s} {'acc':>6s} {'auc':>6s} {'real_rec':>9s} {'synth_acc':>10s}\n")
        for r in results:
            f.write(f"{r['cond']:10s} {r['acc']:6.3f} {r['auc']:6.3f} "
                    f"{r['real_recall']:9.3f} {r['synth_acc']:10.3f}\n")
        f.write("\n=== Diagnostic 2: where synthetic-image errors go "
                "(fractions over all synthetic images) ===\n")
        f.write(f"{'cond':10s} {'->real':>8s} {'->samefam':>10s} {'->crossfam':>11s}\n")
        for r in results:
            f.write(f"{r['cond']:10s} {r.get('synth_err_to_real', 0):8.3f} "
                    f"{r.get('synth_err_same_family', 0):10.3f} "
                    f"{r.get('synth_err_cross_family', 0):11.3f}\n")
    print(f"\nsaved → {out_json}\nsaved → {out_txt}")


if __name__ == '__main__':
    main()
