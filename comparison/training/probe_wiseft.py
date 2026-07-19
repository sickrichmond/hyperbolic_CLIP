"""
WiSE-FT probe for the hyperbolic-CLIP attributor — NO training.

Hypothesis under test (from diag_frequency.py): ours tolerates a pure low-pass
well (blur sigma 1.5 costs 4 points) but collapses under near-lossless JPEG
(q=90 halves accuracy). Since CLIP is pre-trained on web images that are largely
JPEG already, its FROZEN features ought to be JPEG-robust — so the fragility was
probably introduced by the LoRA fine-tuning latching onto the pristine-PNG
statistics of the training set (a shortcut).

WiSE-FT (Wortsman et al., 2022) tests exactly this by interpolating in weight
space between the zero-shot and the fine-tuned model. With LoRA the interpolation
is exact and free: the adapter is additive, W(alpha) = W_base + alpha * scaling * B@A,
so scaling every LoraLayer's `scaling` by alpha walks the straight path from the
frozen CLIP (alpha=0) to the fine-tuned model (alpha=1). No retraining, no merge.

CAVEAT (read before interpreting): the hyperbolic `projection` head was trained
jointly with the adapter at alpha=1, so alpha=0 is NOT a valid zero-shot model —
the head expects LoRA-adapted features. Absolute accuracy is therefore expected
to fall as alpha decreases, for reasons unrelated to robustness. The meaningful
signal is the RETENTION ratio acc(degraded)/acc(clean) at each alpha: if retention
RISES as alpha falls, the shortcut hypothesis is supported, and a less aggressive
adapter (lower rank / fewer target modules / explicit WiSE-FT) becomes a real,
augmentation-free architectural fix.

Usage (CINECA, via SLURM — see slurm/slurm_probe_wiseft.sh):
    python -m comparison.training.probe_wiseft \\
        --checkpoint $WORK/hyp_fine_tuning/checkpoints/attribution_22cls_base_vitl14.pt \\
        --root_dir   $FAST/datasets/iab_dataset \\
        --log_dir    $WORK/outputs/hypclip_wiseft_22cls
"""
import os
import json
import argparse
import datetime

import torch
from torch.utils.data import DataLoader, Subset

from comparison.dataset.ImageAttributionDataset.dataloader import get_dataloader
from comparison.dataset.ImageAttributionDataset.dataset import ImageAttributionDataset
from comparison.training.metrics.base_metrics_class import calculate_metrics_for_test
from comparison.training.test_hypclip import build_anchor_texts
# Reuse the exact degradation injection from the frequency diagnostic so the
# conditions here are identical to the ones measured there.
from comparison.training import diag_frequency as diag

from models.attribution_clip import AttributionCLIP
from geometry.lorentz import oxy_angle
from transformers import CLIPTokenizer


# (name, kind, param) — clean plus the JPEG ramp that exposed the collapse,
# with one blur point as the "low-pass is fine" control.
CONDITIONS = [
    ('clean',   'identity', 0.0),
    ('jpeg90',  'jpeg',     90),
    ('jpeg65',  'jpeg',     65),
    ('jpeg30',  'jpeg',     30),
    ('blur2.0', 'blur',     2.0),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root_dir', required=True)
    p.add_argument('--alphas', type=float, nargs='+',
                   default=[1.0, 0.9, 0.75, 0.5, 0.25, 0.0],
                   help='LoRA interpolation coefficients (1.0 = fine-tuned, 0.0 = frozen CLIP)')
    p.add_argument('--scope', choices=['both', 'vision', 'text'], default='both',
                   help='which encoder(s) to interpolate; "both" is true WiSE-FT')
    p.add_argument('--max_samples', type=int, default=8000,
                   help='deterministic subsample of the test split (0 = full split). '
                        'A probe only needs relative accuracy, and this keeps the '
                        'alpha x condition grid affordable.')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--num_images_per_semantic_per_class', '-n', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log_dir', type=str, default='./logs_probe_wiseft')
    return p.parse_args()


def collect_lora_layers(model, scope):
    """(module, original_scaling) for every PEFT LoraLayer in the chosen scope.

    Duck-typed on the `scaling` dict so it survives peft version differences.
    """
    out = []
    for name, mod in model.named_modules():
        scaling = getattr(mod, 'scaling', None)
        if not isinstance(scaling, dict) or not scaling:
            continue
        if scope == 'vision' and 'vision_model' not in name:
            continue
        if scope == 'text' and 'text_model' not in name:
            continue
        out.append((mod, dict(scaling)))
    return out


def set_alpha(layers, alpha):
    """W(alpha) = W_base + alpha * scaling * B@A, for every adapter."""
    for mod, original in layers:
        for key, val in original.items():
            mod.scaling[key] = val * alpha


@torch.no_grad()
def encode_anchors(model, tokenizer, device):
    """Re-encoded at every alpha: the text encoder carries LoRA too, so the class
    anchors move along the interpolation path exactly like the image features."""
    tok = tokenizer(build_anchor_texts(), return_tensors='pt',
                    padding='max_length', truncation=True, max_length=77)
    x_anc, _ = model.encode_text(tok['input_ids'].to(device),
                                 tok['attention_mask'].to(device))
    return x_anc


@torch.no_grad()
def evaluate(model, x_anc, curv, loader, device):
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
    auc, acc, ap, _, _, _ = calculate_metrics_for_test(labels, logits, sem,
                                                       need_softmax=True)
    return float(acc), float(auc)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    clip_name = ckpt['clip_name']
    curv = ckpt.get('curv', 1.0)
    model = AttributionCLIP(
        clip_name=clip_name,
        lora_r=ckpt.get('lora_r', 8),
        lora_alpha=ckpt.get('lora_alpha', 16),
        hyperbolic_dim=ckpt.get('hyperbolic_dim', 128),
        curv=curv,
    ).to(device)
    model.clip.load_state_dict(ckpt['lora_state'])
    model.projection.load_state_dict(ckpt['projection'])
    model.eval()

    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    layers = collect_lora_layers(model, args.scope)
    if not layers:
        raise RuntimeError(f"no LoRA layers found for scope={args.scope}; "
                           "cannot interpolate")
    print(f"interpolating {len(layers)} LoRA layers (scope={args.scope})")

    # Degradation injection + test split, built ONCE and re-scanned per condition.
    ImageAttributionDataset.get_degraded_img = diag._patched_get_degraded_img
    config = {'model_name': 'hypclip', 'clip_name': clip_name}
    diag._OP_KIND, diag._OP_PARAM = 'identity', 0.0
    _, _, test_loader = get_dataloader(
        root_dir=args.root_dir,
        model_name='hypclip',
        num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,
        batch_size=args.batch_size,
        degraded=0,
        config=config,
        num_workers=args.num_workers,
    )

    test_ds = test_loader.dataset
    if args.max_samples and args.max_samples < len(test_ds):
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(test_ds), generator=g)[:args.max_samples].tolist()
        test_ds = Subset(test_ds, idx)
        print(f"subsampled test split: {len(test_ds)} / {len(test_loader.dataset)}")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    os.makedirs(args.log_dir, exist_ok=True)
    results = []
    for alpha in args.alphas:
        set_alpha(layers, alpha)
        x_anc = encode_anchors(model, tokenizer, device)   # anchors move with alpha
        row = {'alpha': alpha}
        for name, kind, param in CONDITIONS:
            diag._OP_KIND, diag._OP_PARAM = kind, param    # workers fork with these
            acc, auc = evaluate(model, x_anc, curv, loader, device)
            row[name] = acc
            row[f'{name}_auc'] = auc
        clean = row['clean'] or 1e-9
        for name, _, _ in CONDITIONS:
            if name != 'clean':
                row[f'ret_{name}'] = row[name] / clean     # retention vs own clean
        results.append(row)
        print(f"[alpha={alpha:.2f}] " +
              "  ".join(f"{n}={row[n]:.3f}" for n, _, _ in CONDITIONS) +
              f"  | ret_jpeg90={row['ret_jpeg90']:.3f} ret_jpeg65={row['ret_jpeg65']:.3f}")

    set_alpha(layers, 1.0)   # restore the fine-tuned model

    with open(os.path.join(args.log_dir, 'probe_wiseft.json'), 'w') as f:
        json.dump({'checkpoint': args.checkpoint, 'scope': args.scope,
                   'when': str(datetime.datetime.now()), 'results': results},
                  f, indent=2)

    txt = os.path.join(args.log_dir, 'probe_wiseft.txt')
    names = [n for n, _, _ in CONDITIONS]
    with open(txt, 'w') as f:
        f.write(f"WiSE-FT probe (scope={args.scope}) — absolute accuracy\n")
        f.write(f"{'alpha':>6s} " + " ".join(f"{n:>9s}" for n in names) + "\n")
        for r in results:
            f.write(f"{r['alpha']:6.2f} " +
                    " ".join(f"{r[n]:9.3f}" for n in names) + "\n")
        f.write("\nRetention vs own clean (THE signal: rising = shortcut hypothesis "
                "supported)\n")
        deg = [n for n in names if n != 'clean']
        f.write(f"{'alpha':>6s} " + " ".join(f"{n:>9s}" for n in deg) + "\n")
        for r in results:
            f.write(f"{r['alpha']:6.2f} " +
                    " ".join(f"{r['ret_' + n]:9.3f}" for n in deg) + "\n")
    print(f"\nsaved → {txt}")


if __name__ == '__main__':
    main()
