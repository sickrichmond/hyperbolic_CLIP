"""Evaluate IAB baselines on every image in ROOT/STYLE/GENERATOR/SEMANTIC.

Reuse native dataset adapters and Trainer.test. Predictions compete over the
full IAB class map; balanced accuracy averages supported generators only.
No split, sampling cap or additional corruption is applied.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path

from comparison.training.test_hypclip_styled import discover_styles

METHODS = ('resnet50', 'dct', 'hifi_net', 'defl', 'dna', 'repmix', 'patch', 'ucf',
           'lida', 'defake', 'pose')


def load_checkpoint_map(path):
    path = Path(path)
    entries = json.loads(path.read_text())
    if not isinstance(entries, dict) or not entries:
        raise ValueError('Checkpoint map must be a nonempty JSON object: {method: path}')
    unknown = set(entries) - set(METHODS)
    if unknown:
        raise ValueError(f'Unknown baselines in checkpoint map: {sorted(unknown)}')
    resolved = {}
    for method, filename in entries.items():
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f'Missing checkpoint path for {method}')
        checkpoint = Path(filename).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f'{method}: checkpoint not found: {checkpoint}')
        resolved[method] = str(checkpoint.resolve())
    return resolved


def prepare_config(checkpoint, method, fallback, class_names):
    """Legacy IAB checkpoints lack metadata: require the active harness ordering."""
    config = deepcopy(checkpoint.get('training_config', fallback))
    if not isinstance(config, dict) or config.get('model_name') != method:
        raise ValueError(f'Configuration must describe baseline {method}')
    for metadata in (checkpoint, config):
        if metadata.get('dataset', 'iab') != 'iab':
            raise ValueError('Styled IAB evaluation requires an IAB checkpoint/config')
        if metadata.get('model_name', method) != method:
            raise ValueError('Checkpoint and requested baseline differ')
        for key in ('class_names', 'output_class_names'):
            if key in metadata and list(metadata[key]) != class_names:
                raise ValueError(f'{key} differs from the active IAB class order; '
                                 'check IAB_EXCLUDE_GENERATORS')
        if metadata.get('num_classes', len(class_names)) != len(class_names):
            raise ValueError('Checkpoint/config class count differs from the active IAB map')
        if metadata.get('output_label_space', 'generators') != 'generators':
            raise ValueError('Evaluation requires generator attribution weights')
        if method == 'dna' and metadata.get('train_stage', 2) != 2:
            raise ValueError('DNA evaluation requires a stage-two checkpoint')
    if not checkpoint.get('model_state_dict'):
        raise ValueError('Checkpoint has no model_state_dict')
    if method == 'dna':
        head = checkpoint['model_state_dict'].get('backbone.classification_head.1.weight')
        if head is not None and head.shape[0] != len(class_names):
            raise ValueError('DNA checkpoint head is not a stage-two IAB attribution head')
    if method == 'ucf' and config.get('backbone_config', {}).get('num_classes') != 2:
        raise ValueError('UCF requires its native two-class detection head')
    if config.get('specific_task_number', len(class_names)) != len(class_names):
        raise ValueError('Attribution head class count differs from the active IAB map')
    config.update(dataset='iab', num_classes=len(class_names), specific_task_number=len(class_names),
                  class_names=class_names, real_class_index=class_names.index('real'),
                  load_param=False, ddp=False)
    return config


def collect_samples(style, class_names, semantic_paths, semantic_labels):
    """Enumerate complete IAB semantic directories, without filename/cap filters."""
    style = Path(style)
    samples = []
    for label, name in enumerate(class_names):
        directory = style / name
        if not directory.is_dir():
            continue
        expected = {p for p in directory.rglob('*') if p.is_file() and
                    p.suffix.lower() in ('.png', '.jpg', '.jpeg')}
        if not expected:
            raise ValueError(f'No images in generator folder: {directory}')
        loaded = set()
        for semantic, relative in semantic_paths.items():
            folder = directory / relative
            if not folder.is_dir():
                continue
            for image in sorted(folder.iterdir()):
                if image in expected:
                    samples.append((str(image), label, semantic_labels[semantic], semantic))
                    loaded.add(image)
        if expected != loaded:
            raise ValueError(f'{directory}: images outside the supported IAB semantic layout: '
                             f'{sorted(str(p) for p in expected - loaded)[:5]}')
    return samples


def build_styled_loader(style, config, batch_size, num_workers, clip_preprocess=None):
    from torch.utils.data import DataLoader
    from comparison.dataset.ImageAttributionDataset import DATASET
    from comparison.dataset.ImageAttributionDataset.dataset import (
        semantic_label_map, semantic_to_relpath,
    )

    # Native adapters construct their transforms/hierarchy; replace enumeration
    # so the ordinary IAB filename filter and train/test split cannot drop rows.
    dataset = DATASET[config['model_name']](
        root_dir='', config=config, degraded=0, clip_preprocess=clip_preprocess)
    dataset.root_dir = str(style)
    dataset.samples = collect_samples(style, config['class_names'],
                                      semantic_to_relpath, semantic_label_map)
    if config['model_name'] == 'repmix':
        dataset.N = len(dataset.samples)
    dataset.set_test()
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, drop_last=False,
                      collate_fn=getattr(dataset, 'collate_fn', None))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument('--checkpoint')
    checkpoints.add_argument('--checkpoints', help='JSON object mapping baseline names to checkpoints')
    parser.add_argument('--config', help='Legacy checkpoint YAML; otherwise use the method default')
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('batch_size must be positive and num_workers must be non-negative')
    if args.checkpoints:
        mapping = load_checkpoint_map(args.checkpoints)
        if args.method not in mapping:
            parser.error(f'Checkpoint map has no entry for {args.method}')
        args.checkpoint = mapping[args.method]
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    import yaml
    from comparison.dataset.ImageAttributionDataset.dataset import model_class_to_label
    from comparison.training.attributors import ATTRIBUTOR
    from comparison.training.logger import create_logger
    from comparison.training.test import init_seed
    from comparison.training.trainer.trainer import Trainer

    class_names = list(model_class_to_label)
    styles = discover_styles(args.root_dir, class_names)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config_name = 'dna_default' if args.method == 'dna' else args.method
    config_path = Path(args.config) if args.config else (
        Path(__file__).parent / 'config/model' / f'{config_name}.yaml')
    config = prepare_config(checkpoint, args.method,
                            yaml.safe_load(config_path.read_text()), class_names)
    init_seed(config.get('manualSeed', 42), torch.cuda.is_available())
    if args.method not in ATTRIBUTOR.data:
        raise RuntimeError(f'{args.method} is unavailable; check the optional dependency import warnings')
    output = Path(args.log_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(output / 'test.log'))
    model = ATTRIBUTOR[args.method](config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    trainer = Trainer(config, model, None, None, logger, log_dir=str(output), test_mode=True)
    clip_preprocess = model.semantic_extractor.clip_preprocess if args.method == 'defl' else None
    summary = []
    for style, generators in styles:
        loader = build_styled_loader(style, config, args.batch_size, args.num_workers,
                                     clip_preprocess=clip_preprocess)
        counts = Counter(class_names[row[1]] for row in loader.dataset.samples)
        print(f'\nBaseline: {args.method} | Style: {style.name} | images: {len(loader.dataset)} '
              f'| counts: {dict(counts)}')
        metrics = trainer.test(loader)
        if metrics['conf_matrix'].shape != (len(class_names), len(class_names)):
            raise ValueError('Prediction head does not match the full IAB label space')
        result = {
            'model_name': args.method, 'checkpoint': str(Path(args.checkpoint).resolve()),
            'data_root': str(style.resolve()), 'style': style.name,
            'class_names': class_names, 'evaluated_generators': generators,
            'num_images': len(loader.dataset), 'preprocessing': 'native adapter; no extra corruptions',
            'config': config,
            **{key: value for key, value in metrics.items() if key not in ('acc', 'conf_matrix')},
            'accuracy': metrics['acc'], 'confusion_matrix': metrics['conf_matrix'].tolist(),
        }
        style_output = output / style.name
        style_output.mkdir(parents=True, exist_ok=True)
        path = style_output / 'metrics.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        summary.append({'model': args.method, 'style': style.name,
                        'num_images': len(loader.dataset), 'accuracy': metrics['acc'],
                        'balanced_accuracy': metrics['balanced_accuracy']})
        with (output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"  balanced accuracy={100 * metrics['balanced_accuracy']:.2f}% | saved → {path}")


if __name__ == '__main__':
    main()
