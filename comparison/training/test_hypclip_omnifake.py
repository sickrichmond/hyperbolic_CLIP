"""Evaluate an IAB HypCLIP checkpoint on OmniFake's ten shared test classes."""

import argparse
import json
from pathlib import Path

from data.omnifake import OMNIFAKE_TO_IAB, discover_test_samples


class OmniFakeTestDataset:
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from data.image_io import open_image_retry

        path, label = self.samples[index]
        image = open_image_retry(path)
        return {'pixel_values': self.processor(images=image, return_tensors='pt')['pixel_values'][0],
                'generator': label}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root_dir', type=Path, required=True, help='Parent of OmniFake val/')
    parser.add_argument('--log_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('batch_size must be positive and num_workers must be non-negative')

    samples = discover_test_samples(args.root_dir)

    import torch
    from transformers import CLIPImageProcessor

    from comparison.training.test_hypclip import (
        AttributionCLIP, calculate_metrics_for_test, evaluate_loader, load_anchors,
    )

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('dataset', 'iab') != 'iab':
        raise ValueError('OmniFake evaluation requires an IAB checkpoint')
    class_names = list(ckpt['class_names'])
    if len(set(class_names)) != len(class_names):
        raise ValueError('Checkpoint class_names contains duplicates')
    missing = (set(OMNIFAKE_TO_IAB.values()) | {'real'}) - set(class_names)
    if missing:
        raise ValueError(f'Checkpoint is missing IAB labels: {sorted(missing)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    curv = ckpt.get('curv', 1.0)
    model = AttributionCLIP.from_checkpoint(ckpt).to(device)
    model.clip.load_state_dict(ckpt['lora_state'])
    model.projection.load_state_dict(ckpt['projection'])
    model.eval()
    processor = CLIPImageProcessor.from_pretrained(ckpt['clip_name'])
    loader = torch.utils.data.DataLoader(
        OmniFakeTestDataset(samples, processor), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, drop_last=False)

    print(f'Scoring {len(samples)} OmniFake test images against {len(class_names)} IAB labels')
    with torch.no_grad():
        anchors = load_anchors(ckpt, model, curv, device, class_names=class_names)
        logits, labels, _ = evaluate_loader(model, loader, anchors, class_names, device, curv)
    auc, accuracy, ap, confusion, _, extra = calculate_metrics_for_test(labels, logits)
    results = {
        'checkpoint': str(args.checkpoint.resolve()),
        'data_root': str(args.root_dir.resolve()),
        'dataset': 'OmniFake',
        'split': 'val',
        'preprocessing': 'checkpoint CLIP processor',
        'decision_rule': 'argmin exterior angle',
        'num_images': len(samples),
        'omnifake_to_iab': OMNIFAKE_TO_IAB,
        'class_names': class_names,
        'accuracy': accuracy,
        'balanced_accuracy': extra['balanced_accuracy'],
        'auc': float(auc),
        'ap': float(ap),
        'support_per_class': extra['support_per_class'],
        'recall_per_class': extra['recall_per_class'],
        'confusion_matrix': confusion.tolist(),
    }
    args.log_dir.mkdir(parents=True, exist_ok=True)
    path = args.log_dir / 'metrics.json'
    with path.open('x') as stream:
        json.dump(results, stream, indent=2)
        stream.write('\n')
    print(f"Accuracy {accuracy:.2%} | balanced accuracy {extra['balanced_accuracy']:.2%}")
    for folder, label in OMNIFAKE_TO_IAB.items():
        index = class_names.index(label)
        print(f"{folder:<28} {extra['recall_per_class'][index]:.2%}")
    print(f'Metrics saved to {path}')


if __name__ == '__main__':
    main()
