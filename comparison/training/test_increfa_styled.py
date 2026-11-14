"""Evaluate an external, static IAB IncreFA checkpoint on styled IAB or OmniFake.

Run with the hyp_fine_tuning environment. IncreFA's native trainer supplies the model,
family setup, label mapping and image transform. Scores are fractions in JSON/CSV.
"""
import argparse
from copy import deepcopy
import csv
from io import BytesIO
import json
import os
from pathlib import Path
from runpy import run_path
import sys
import time

from comparison.training.test_hypclip_styled import discover_styles


SEMANTIC_DIRS = {'COCO', 'ImageNet-1k', 'AnimalFace/cat', 'AnimalFace/dog',
                 'AnimalFace/wild', 'HumanFace/FFHQ', 'HumanFace/celebahq',
                 'Scene/bedroom', 'Scene/church', 'Scene/classroom'}


def checkpoint_classes(checkpoint):
    config = checkpoint['config']
    order = checkpoint['results']['class_order']
    if 'manifest_path' not in config['data'] or 'split_manifest_path' in config['data']:
        raise ValueError('Expected an IAB folder-manifest checkpoint')
    if len(order) != 1 or checkpoint['task_id'] != 0:
        raise ValueError('Expected one completed all-class task')
    names = order[0]
    if (not names or names[0] != 'real' or len(names) != len(set(names)) or
            len(names) != config['data']['total_classes'] or
            len(names) != checkpoint['num_learned_classes']):
        raise ValueError('Invalid or incomplete checkpoint class order')
    return list(names)


def restore_model(checkpoint):
    from train import IncreTrainer
    from models import load_model

    names = checkpoint_classes(checkpoint)
    # Reuse native evaluation setup without constructing a training run/logger.
    trainer = IncreTrainer.__new__(IncreTrainer)
    trainer.config = deepcopy(checkpoint['config'])
    trainer.model = load_model(config=trainer.config)
    trainer.setseed()
    trainer.dataset = trainer.load_dataset()
    expected = [[trainer.dataset.idx_to_label_str[i] for i in task]
                for task in trainer.dataset.order]
    if expected != [names]:
        raise ValueError('Checkpoint class order differs from the native IAB manifest')
    trainer.configure_predefined_families()
    if checkpoint['class_to_family'] != trainer.model.class_to_family:
        raise ValueError('Native family mapping differs from the checkpoint')
    missing, unexpected = trainer.model.load_state_dict(checkpoint['model'], strict=False)
    if unexpected or any(not key.startswith('backbone.') for key in missing):
        raise ValueError(f'Invalid model state: missing={missing}, unexpected={unexpected}')
    trainer.model.orth_space.update_indicator(len(names))
    trainer.model.eval()
    return trainer.model, trainer.dataset.transform, names


def generator_logits(model, images, count):
    logits, _ = model(images, y_gt=None, store_samples=False)
    if logits.ndim != 2 or logits.shape[1] < count:
        raise ValueError('Prediction head does not cover all checkpoint classes')
    return logits[:, :count]


def collect_samples(style, names):
    samples = []
    for label, name in enumerate(names):
        directory = style / name
        if not directory.is_dir():
            continue
        files = sorted(p for p in directory.rglob('*')
                       if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
        if not files:
            raise ValueError(f'No images in generator folder: {directory}')
        for path in files:
            if path.parent.relative_to(directory).as_posix() not in SEMANTIC_DIRS:
                raise ValueError(f'Image outside the supported IAB semantic layout: {path}')
            samples.append((path, label))
    return samples


def omnifake_queries(root, names):
    # Load the shared inventory without occupying IncreFA's native data namespace.
    inventory = run_path(str(Path(__file__).resolve().parents[2] / 'data/omnifake.py'))
    mapping = inventory['OMNIFAKE_TO_IAB']
    indices = {name: i for i, name in enumerate(names)}
    missing = set(mapping.values()) - indices.keys()
    if missing:
        raise ValueError(f'Checkpoint is missing OmniFake labels: {sorted(missing)}')
    return [(path, indices[name]) for path, name in inventory['discover_test_samples'](root)], mapping


class StyledDataset:
    def __init__(self, samples, transform, names, *, jpeg99=True):
        self.samples = samples
        self.transform = transform
        self.names = names
        self.jpeg99 = jpeg99

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, label = self.samples[index]
        for attempt in range(8):
            try:
                with Image.open(path) as source:
                    image = source.convert('RGB')
                break
            except OSError:
                if attempt == 7:
                    raise
                time.sleep(0.5 * (attempt + 1))
        # External IncreFA owns the data.* namespace, so encode JPEG here.
        if self.jpeg99:
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=99)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                image = compressed.convert('RGB')
        if self.names[label] == 'grok3':
            width, height = image.size
            if width > 100 and height > 50:
                image = image.crop((0, 0, width - 100, height - 50))
        return self.transform(image), label


def summarize(matrix, names):
    support = [sum(row) for row in matrix]
    total = sum(support)
    if not total:
        raise ValueError('Evaluation has no images')
    recalls = [matrix[i][i] / count for i, count in enumerate(support) if count]
    return {
        'num_images': total,
        'accuracy': sum(matrix[i][i] for i in range(len(names))) / total,
        'balanced_accuracy': sum(recalls) / len(recalls),
        'per_class_recall': {name: matrix[i][i] / support[i] if support[i] else None
                             for i, name in enumerate(names)},
        'per_class_support': dict(zip(names, support)),
        'confusion_matrix': matrix,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--increfa_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--omnifake', action='store_true', help='Use ROOT/val with no added JPEG compression')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('batch_size must be positive and num_workers must be non-negative')
    for name in ('increfa_dir', 'checkpoint', 'root_dir', 'log_dir'):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if not (args.increfa_dir / 'train.py').is_file():
        parser.error('increfa_dir must contain the supplied native train.py')
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from torch.utils.data import DataLoader

    # Native imports and relative config/CLIP-weight paths belong to IncreFA.
    os.chdir(args.increfa_dir)
    sys.path.insert(0, str(args.increfa_dir))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    names = checkpoint_classes(checkpoint)
    if args.omnifake:
        from comparison.training.metrics.base_metrics_class import calculate_metrics_for_test

        samples, mapping = omnifake_queries(args.root_dir, names)
        supported = set(mapping.values())
        evaluations = [(args.root_dir, [name for name in names if name in supported])]
    else:
        evaluations = discover_styles(args.root_dir, names)
    model, transform, names = restore_model(checkpoint)
    device = checkpoint['config']['device']
    del checkpoint
    args.log_dir.mkdir(parents=True, exist_ok=True)
    print(f'Checkpoint: {args.checkpoint}\nClasses: {names}\nResults: {args.log_dir}', flush=True)
    summary = []
    with torch.no_grad():
        for root, generators in evaluations:
            if not args.omnifake:
                samples = collect_samples(root, names)
            dataset = StyledDataset(samples, transform, names, jpeg99=not args.omnifake)
            loader = DataLoader(dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True,
                                shuffle=False, drop_last=False)
            count = len(names)
            matrix = torch.zeros(count * count, dtype=torch.long)
            all_logits, all_labels = [], []
            for images, labels in loader:
                logits = generator_logits(model, images.to(device), count).cpu()
                if args.omnifake:
                    all_logits.append(logits.float())
                    all_labels.append(labels)
                else:
                    predictions = logits.argmax(dim=1)
                    matrix += torch.bincount(labels * count + predictions, minlength=count * count)
            if args.omnifake:
                auc, _, ap, confusion, _, metrics = calculate_metrics_for_test(
                    torch.cat(all_labels), torch.cat(all_logits))
                matrix = confusion.tolist()
                condition = {
                    'dataset': 'OmniFake', 'split': 'val', 'omnifake_to_iab': mapping,
                    'evaluation_condition': 'clean', 'preprocessing': 'native IncreFA IAB preprocessing',
                    'auc': float(auc), 'ap': float(ap),
                    'support_per_class': metrics['support_per_class'],
                    'recall_per_class': metrics['recall_per_class'],
                }
                output = args.log_dir
                summary_keys = ('model_name', 'dataset', 'num_images', 'accuracy', 'balanced_accuracy')
            else:
                matrix = matrix.reshape(count, count).tolist()
                condition = {
                    'style': root.name, 'evaluation_condition': 'jpeg99', 'jpeg_quality': 99,
                    'preprocessing': 'JPEG99 before native IncreFA IAB preprocessing',
                }
                output = args.log_dir / root.name
                summary_keys = ('model_name', 'style', 'num_images', 'jpeg_quality',
                                'accuracy', 'balanced_accuracy')
            report = {
                'model_name': 'increfa', 'checkpoint': str(args.checkpoint),
                'data_root': str(root), 'class_names': names, 'evaluated_generators': generators,
                'decision_rule': 'argmax native generator logits',
                **condition, **summarize(matrix, names),
            }
            output.mkdir(parents=True, exist_ok=True)
            (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
            summary.append({key: report[key] for key in summary_keys})
            with (args.log_dir / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            print(f'{root.name:<28} accuracy={100 * report["accuracy"]:.2f}% '
                  f'balanced_accuracy={100 * report["balanced_accuracy"]:.2f}%', flush=True)


if __name__ == '__main__':
    main()
