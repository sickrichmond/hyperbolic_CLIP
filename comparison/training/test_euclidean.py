"""
Fair evaluation of the EUCLIDEAN ablation on ImageAttributionBench, under a
protocol byte-identical to the comparison baselines and to test_hypclip.py.

Same machinery as the hyperbolic runner — `get_dataloader(model_name='hypclip')`
and `calculate_metrics_for_test` — because the `hypclip` adapter is model-agnostic:
it only turns the harness image into a CLIP-preprocessed `pixel_values` tensor,
which is exactly what the Euclidean model consumes too. Same test images, same
split, same degradations, same metrics, same output file format, so the euclidean
column drops straight into the tables next to the hyperbolic ones.

The only model-specific part is the logits. On the sphere the model's decision
rule is argmax_c <x_img, x_anc_c> (losses.euclidean_attribution_loss), so:

    logits[i, c] = logit_scale * <x_img_i, x_anc_c>

with the checkpoint's LEARNED logit_scale. argmax reproduces the model's own
prediction exactly, and the scale matters for AUC/AP because softmax normalises
across classes — dropping it would report a differently-calibrated model.

Usage:
    python -m comparison.training.test_euclidean \\
        --checkpoint $WORK/hyp_fine_tuning/checkpoints/attribution_22cls_euclidean_d128_vitl14.pt \\
        --root_dir   $FAST/datasets/iab_dataset \\
        --level_start 0 --level_end 7 \\
        --log_dir    $WORK/outputs/euclidean_22cls
"""
import argparse
import datetime
import os

import torch
from tqdm import tqdm
from transformers import CLIPTokenizer

from comparison.dataset.ImageAttributionDataset.dataloader import get_dataloader
from comparison.dataset.ImageAttributionDataset.semantic_split import get_semantic
from comparison.training.metrics.base_metrics_class import calculate_metrics_for_test
from comparison.training.test_hypclip import build_anchor_texts, harness_class_names
from data.degradations import LEVEL_LABELS
from models.euclidean_attribution_clip import EuclideanAttributionCLIP

MAX_LOGIT_SCALE = 100.0     # same cap as losses.euclidean_attribution_loss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='euclidean .pt checkpoint')
    p.add_argument('--root_dir', required=True, help='root of the IAB dataset')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--num_images_per_semantic_per_class', '-n', type=int, default=2000)
    p.add_argument('--level_start', type=int, default=0)
    p.add_argument('--level_end', type=int, default=1, help='exclusive (0..7 covers all)')
    p.add_argument('--use_semantic_split', action='store_true', default=False)
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--log_dir', type=str, default='./logs_test_euclidean')
    return p.parse_args()


def load_anchors(ckpt, model, device):
    """Unit-sphere class anchors in the HARNESS label order → column c == label c.

    Same permutation contract as test_hypclip.load_anchors, and the same reason
    for re-encoding the CHECKPOINT's own prompts rather than the templates: a run
    trained with --anchor_prompts would otherwise be scored against sentences it
    never saw, silently.
    """
    ckpt_names = list(ckpt['class_names'])
    harness_names = harness_class_names()
    missing = [n for n in harness_names if n not in ckpt_names]
    if missing:
        raise ValueError(f"checkpoint has no anchor for class(es) {missing}; "
                         f"it was trained on {ckpt_names}")
    perm = [ckpt_names.index(n) for n in harness_names]

    ckpt_texts = ckpt.get('anchor_texts')
    if ckpt_texts:
        anchor_texts = [ckpt_texts[i] for i in perm]
        src = 'checkpoint prompts'
    else:
        anchor_texts = build_anchor_texts()
        src = 'default templates'
    tokenizer = CLIPTokenizer.from_pretrained(ckpt['clip_name'])
    tok = tokenizer(anchor_texts, return_tensors='pt', padding='max_length',
                    truncation=True, max_length=77)
    print(f"Anchors: {len(anchor_texts)} text anchors from {src}")
    print(f"  [0] {harness_names[0]} → \"{anchor_texts[0]}\"")
    return model.encode_text(tok['input_ids'].to(device),
                             tok['attention_mask'].to(device))


@torch.no_grad()
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get('geometry') != 'euclidean':
        raise ValueError(f"{args.checkpoint} is not a euclidean checkpoint "
                         f"(geometry={ckpt.get('geometry')!r}); use test_hypclip.py")
    clip_name = ckpt['clip_name']

    model = EuclideanAttributionCLIP(
        clip_name=clip_name,
        lora_r=ckpt.get('lora_r', 8),
        lora_alpha=ckpt.get('lora_alpha', 16),
        embed_dim=ckpt.get('embed_dim', 128),
    ).to(device)
    model.clip.load_state_dict(ckpt['lora_state'])
    model.projection.load_state_dict(ckpt['projection'])
    model.logit_scale.data = ckpt['logit_scale'].to(device)
    model.eval()

    x_anc = load_anchors(ckpt, model, device)                          # (K, D)
    K = x_anc.shape[0]
    scale = torch.clamp(model.logit_scale.exp(), max=MAX_LOGIT_SCALE)
    print(f"embed_dim={ckpt.get('embed_dim')}  logit_scale={scale.item():.2f}  "
          f"(val_balanced at save time: {ckpt.get('val_balanced')})")

    os.makedirs(args.log_dir, exist_ok=True)
    config = {'model_name': 'hypclip', 'clip_name': clip_name, 'num_classes': K}

    train_semantics = test_semantics = None
    if args.use_semantic_split:
        train_semantics, test_semantics = get_semantic(args.task_id)

    for degraded in range(args.level_start, args.level_end):
        deg_label = LEVEL_LABELS.get(degraded, str(degraded))
        print(f"\n=== degraded={degraded} ({deg_label}) ===")

        _, _, test_loader = get_dataloader(
            root_dir=args.root_dir,
            model_name='hypclip',
            num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,
            batch_size=args.batch_size,
            degraded=degraded,
            config=config,
            num_workers=args.num_workers,
            use_semantic_split=args.use_semantic_split,
            train_semantics=train_semantics,
            test_semantics=test_semantics,
        )

        all_logits, all_labels, all_sem = [], [], []
        for batch in tqdm(test_loader, total=len(test_loader), desc=f"deg{degraded}"):
            x_img = model.encode_image(batch['image'].to(device))       # (B, D)
            all_logits.append((scale * (x_img @ x_anc.t())).cpu())      # (B, K)
            all_labels.append(batch['label'].cpu())
            all_sem.append(batch['semantic_label'].cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        sem = torch.cat(all_sem, dim=0)

        auc, acc, ap, conf_matrix, semantic_acc, extra = \
            calculate_metrics_for_test(labels, logits, sem, need_softmax=True)

        out_path = os.path.join(args.log_dir, f"test_results_degraded_{degraded}.txt")
        with open(out_path, 'w') as f:
            f.write(f"Test metrics for degraded={degraded} ({deg_label}) "
                    f"({datetime.datetime.now()}):\n")
            f.write(f"acc: {acc}\n")
            f.write(f"auc: {auc}\n")
            f.write(f"ap: {ap}\n")
            for k, v in extra.items():
                f.write(f"{k}: {v}\n")
            f.write(f"semantic_acc: {semantic_acc}\n")
            f.write(f"conf_matrix:\n{conf_matrix}\n")
        print(f"  acc={acc:.4f}  auc={auc:.4f}  ap={ap:.4f}  "
              f"P_macro={extra['precision_macro']:.4f}  R_macro={extra['recall_macro']:.4f}")
        print(f"  saved → {out_path}")


if __name__ == '__main__':
    main()
