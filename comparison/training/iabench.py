"""Bind baseline training to an existing shared IABench manifest."""
import json
from pathlib import Path

import torch

from data.iabench_dataset import IABenchDataset, manifest_digest
from comparison.dataset.ImageAttributionDataset.dataset import hifi_label_mapping


def prepare_iabench(root, split_manifest, config, batch_size):
    if not split_manifest:
        raise ValueError("IABench training requires --split_manifest (an existing file)")
    manifest = json.loads(Path(split_manifest).read_text())
    source = IABenchDataset(root, processor_name=None)
    train_source = source.split_view(manifest, 'train')
    val_source = source.split_view(manifest, 'val')
    names = list(manifest['class_names'])
    real = [i for i, name in enumerate(names) if name.casefold() == 'real']
    if real != [0]:
        raise ValueError("IABench manifest must contain exactly one real class, first")
    method = config['model_name']
    if method == 'ucf' and (batch_size < 2 or batch_size % 2):
        raise ValueError("UCF requires an even batch size of at least two")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if method == 'ucf' and len(train_source) < batch_size:
        raise ValueError("UCF training needs at least one complete batch")
    config.update(dataset='iabench', class_names=names, real_class_index=real[0],
                  num_classes=len(names), specific_task_number=len(names), start_epoch=1)
    metadata = {
        'dataset': 'iabench', 'model_name': method, 'class_names': names,
        'dataset_digest': source.dataset_digest, 'manifest_digest': manifest_digest(manifest),
        'split_config': {key: manifest[key] for key in ('seed', 'val_frac', 'test_frac', 'max_per_class')},
        'split_counts': {split: len(manifest[split]) for split in ('train', 'val', 'test')},
        'output_label_space': 'generators', 'output_class_names': names,
    }
    if method in ('hifi_net', 'defl'):
        config['hierarchy_mapping'] = hifi_label_mapping(names, dataset='iabench')
        config['hierarchy_sizes'] = [2, 4, 7, len(names)]
        metadata.update(hierarchy_mapping=config['hierarchy_mapping'],
                        hierarchy_sizes=config['hierarchy_sizes'])
    if method == 'dna':
        stage = config['train_stage']
        if stage not in (1, 2):
            raise ValueError("DNA train_stage must be 1 or 2")
        metadata['train_stage'] = stage
        if stage == 1:
            config['num_classes'] = config['class_num'] = 170
            metadata['output_label_space'] = 'transformations'
            metadata['output_class_names'] = [f'transform_{i}' for i in range(170)]
    metadata['num_classes'] = config['num_classes']
    # Paths and epoch limits may change on resume; the actual training policy may not.
    metadata['training_config'] = {
        key: value for key, value in config.items()
        if key not in ('log_dir', 'pretrained_path', 'nEpochs', 'start_epoch')}
    metadata['batch_size'] = batch_size
    config['checkpoint_metadata'] = metadata
    return train_source, val_source


def validate_checkpoint_metadata(checkpoint, expected):
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Incompatible checkpoint {key}: expected {value!r}, got {checkpoint.get(key)!r}")


def validate_dna_stage_one(path, metadata):
    if not path:
        raise ValueError("IABench DNA stage two requires an explicit --pretrained_path stage-one checkpoint")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    expected = {key: metadata[key] for key in (
        'dataset', 'model_name', 'class_names', 'dataset_digest', 'manifest_digest', 'split_config')}
    expected.update(train_stage=1, num_classes=170, output_label_space='transformations',
                    output_class_names=[f'transform_{i}' for i in range(170)])
    validate_checkpoint_metadata(checkpoint, expected)
    if not checkpoint.get('model_state_dict'):
        raise ValueError("DNA stage-one checkpoint has no model weights")


def get_iabench_dataloaders(train_source, val_source, config, batch_size, num_workers,
                           clip_preprocess=None):
    from torch.utils.data import DataLoader
    from comparison.dataset.ImageAttributionDataset import DATASET

    adapter = DATASET[config['model_name']]
    views = []
    for source in (train_source, val_source):
        view = adapter(root_dir='', config=config, iabench_source=source,
                       clip_preprocess=clip_preprocess)
        view.set_train() if source.split == 'train' else view.set_val()
        views.append(view)
    generator = torch.Generator().manual_seed(config.get('manualSeed', 42))
    common = {'batch_size': batch_size, 'num_workers': num_workers,
              'collate_fn': getattr(views[0], 'collate_fn', None)}
    train = DataLoader(views[0], shuffle=True, generator=generator,
                       drop_last=config['model_name'] == 'ucf', **common)
    val = DataLoader(views[1], shuffle=False, **common)
    return train, val
