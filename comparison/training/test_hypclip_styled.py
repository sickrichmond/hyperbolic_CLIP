"""Evaluate IAB cone checkpoints on ROOT/style/generator/semantic/images.

Every image is evaluated, without a train/test split, sample cap or additional
corruptions. Predictions compete over ALL checkpoint classes. Balanced accuracy
averages recalls only over generators present in each style.

Usage: python -m comparison.training.test_hypclip_styled --help
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def discover_styles(root, class_names):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Styled dataset root is not a directory: {root}")
    styles = []
    for style in sorted(p for p in root.iterdir() if p.is_dir()):
        generators = {p.name for p in style.iterdir() if p.is_dir()}
        unknown = generators - set(class_names)
        if unknown:
            raise ValueError(f"{style}: generators absent from checkpoint: {sorted(unknown)}")
        if not generators:
            raise ValueError(f"Style has no generator folders: {style}")
        styles.append((style, [name for name in class_names if name in generators]))
    if not styles:
        raise ValueError(f"No style folders found in {root}")
    return styles


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root_dir', required=True, help='parent of the style folders')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('batch_size must be positive and num_workers must be non-negative')
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch

    from comparison.training.test_hypclip import (
        AttributionCLIP, calculate_metrics_for_test, evaluate_loader, load_anchors,
    )
    from data.iab_dataset import IABDataset, SEMANTIC_TO_SUPER, _IMAGE_EXTS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('dataset', 'iab') != 'iab':
        raise ValueError('Styled IAB evaluation requires an IAB checkpoint')
    class_names = list(ckpt['class_names'])
    if len(set(class_names)) != len(class_names):
        raise ValueError('Checkpoint class_names contains duplicates')
    styles = discover_styles(args.root_dir, class_names)
    curv = ckpt.get('curv', 1.0)
    model = AttributionCLIP.from_checkpoint(ckpt).to(device)
    model.clip.load_state_dict(ckpt['lora_state'])
    model.projection.load_state_dict(ckpt['projection'])
    model.eval()
    output = Path(args.log_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = []

    with torch.no_grad():
        anchors = load_anchors(ckpt, model, curv, device, class_names=class_names)
        for style, generators in styles:
            dataset = IABDataset(str(style), generators, list(SEMANTIC_TO_SUPER),
                                 processor_name=ckpt['clip_name'])
            counts = Counter(generator for _, generator in dataset.samples)
            empty = [name for name in generators if not counts[name]]
            if empty:
                raise ValueError(f'{style}: no images found for {empty}')
            # Catch unexpected semantic layouts instead of silently dropping images.
            files = {p for name in generators for p in (style / name).rglob('*')
                     if p.is_file() and p.suffix.lower() in _IMAGE_EXTS}
            loaded = {p for p, _ in dataset.samples}
            if files != loaded:
                raise ValueError(f'{style}: images outside the supported IAB semantic layout: '
                                 f'{sorted(str(p) for p in files - loaded)[:5]}')
            print(f'\nStyle: {style.name} | images: {len(dataset)} | counts: {dict(counts)}')
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=False, drop_last=False)
            logits, labels, _ = evaluate_loader(
                model, loader, anchors, class_names, device, curv)
            auc, acc, ap, confusion, _, extra = calculate_metrics_for_test(
                labels, logits, need_softmax=True)
            result = {
                'checkpoint': str(Path(args.checkpoint).resolve()),
                'data_root': str(style.resolve()), 'style': style.name,
                'class_names': class_names, 'evaluated_generators': generators,
                'num_images': len(dataset), 'decision_rule': 'argmin exterior angle',
                'preprocessing': 'checkpoint CLIP processor; no extra corruptions',
                'accuracy': acc, 'auc': auc, 'ap': ap, **extra,
                'confusion_matrix': confusion.tolist(),
            }
            style_output = output / style.name
            style_output.mkdir(parents=True, exist_ok=True)
            path = style_output / 'metrics.json'
            path.write_text(json.dumps(result, indent=2) + '\n')
            summary.append({'style': style.name, 'num_images': len(dataset),
                            'accuracy': acc, 'balanced_accuracy': extra['balanced_accuracy']})
            with (output / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            print(f"  accuracy={100 * acc:.2f}% | "
                  f"balanced accuracy={100 * extra['balanced_accuracy']:.2f}%")
            print(f'  saved → {path}')

    print('\nStyle                        Balanced accuracy (%)')
    for row in summary:
        print(f"{row['style']:<28} {100 * row['balanced_accuracy']:>8.2f}")


if __name__ == '__main__':
    main()
