"""Evaluate a multi-view attributor under the baselines' exact protocol.

Same machinery as comparison/training/test_hypclip.py — `get_dataloader` for the
enumeration/split/degradations and `calculate_metrics_for_test` for the metrics —
so the test images are byte-identical to resnet50/dct/hifi_net/... and the output
files drop straight into the comparison tables. Views are cut from the
CLIP-preprocessed tensor the `hypclip` adapter already returns, so no new dataset
adapter is needed.

`run_eval` is parameterised by a builder, and patch_freq_attribution/eval.py
reuses it with its own two-branch model.

Usage:
    python -m patch_attribution.eval \\
        --checkpoint $WORK/hyp_fine_tuning/checkpoints/attribution_22cls_patch.pt \\
        --root_dir $FAST/datasets/iab_dataset --level_start 0 --level_end 7 \\
        --log_dir $WORK/outputs/hypclip_patch
"""
import argparse
import datetime
import os

import torch
from tqdm import tqdm

from comparison.dataset.ImageAttributionDataset.dataloader import get_dataloader
from comparison.dataset.ImageAttributionDataset.semantic_split import get_semantic
from comparison.training.metrics.base_metrics_class import calculate_metrics_for_test
from comparison.training.test_hypclip import load_anchors
from data.degradations import LEVEL_LABELS
from patch_attribution.model import PatchAttributionCLIP, view_logits


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root_dir', required=True, help='root of the IAB dataset')
    p.add_argument('--batch_size', type=int, default=16,
                   help='multiplied by the number of views before it hits the GPU')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--num_images_per_semantic_per_class', '-n', type=int, default=2000)
    p.add_argument('--level_start', type=int, default=0)
    p.add_argument('--level_end', type=int, default=7, help='exclusive')
    p.add_argument('--use_semantic_split', action='store_true', default=False)
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--log_dir', type=str, default='./logs_test_patch')
    return p.parse_args()


def build_patch_model(ckpt, device):
    """→ (model, logits_fn) for the image+patches model."""
    curv = ckpt.get('curv', 1.0)
    model = PatchAttributionCLIP(
        clip_name=ckpt['clip_name'],
        lora_r=ckpt.get('lora_r', 16),
        lora_alpha=ckpt.get('lora_alpha', 32),
        hyperbolic_dim=ckpt.get('hyperbolic_dim', 128),
        curv=curv,
        patch_size=ckpt.get('patch_size', 112),
    ).to(device)
    model.clip.load_state_dict(ckpt['lora_state'])
    model.projection.load_state_dict(ckpt['projection'])
    model.eval()
    # load_anchors reorders the checkpoint's class order ('real' first) into the
    # harness label order ('real' last) — a wrong permutation still yields a full,
    # meaningless metrics table, so never build the anchors by hand here.
    x_anc = load_anchors(ckpt, model, curv, device)
    return model, lambda pixel: view_logits(model.encode_views(pixel).float(), x_anc, curv)


@torch.no_grad()
def run_eval(args, build=build_patch_model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, logits_fn = build(ckpt, device)

    os.makedirs(args.log_dir, exist_ok=True)
    # The view source is read from the CHECKPOINT, never from a flag: evaluating a
    # native-grid model on 224-tensor views (or vice versa) silently produces a full
    # set of wrong metrics. 'hypclip' returns the plain 224 tensor and the model cuts
    # its own grid; 'hypclip_patch' returns the 10 full-resolution views.
    model_name = ('hypclip_patch' if ckpt.get('patch_source') == 'native'
                  else 'hypclip')
    print(f"Patch source: {ckpt.get('patch_source', 'tensor')} → dataset '{model_name}'")
    config = {'model_name': model_name, 'clip_name': ckpt['clip_name']}

    train_semantics = test_semantics = None
    if args.use_semantic_split:
        train_semantics, test_semantics = get_semantic(args.task_id)

    for degraded in range(args.level_start, args.level_end):
        deg_label = LEVEL_LABELS.get(degraded, str(degraded))
        print(f"\n=== degraded={degraded} ({deg_label}) ===")

        _, _, test_loader = get_dataloader(
            root_dir=args.root_dir,
            model_name=model_name,
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
            all_logits.append(logits_fn(batch['image'].to(device)).cpu())
            all_labels.append(batch['label'].cpu())
            all_sem.append(batch['semantic_label'].cpu())

        auc, acc, ap, conf_matrix, semantic_acc, extra = calculate_metrics_for_test(
            torch.cat(all_labels), torch.cat(all_logits), torch.cat(all_sem),
            need_softmax=True)

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
    run_eval(parse_args())
