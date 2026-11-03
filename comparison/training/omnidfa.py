"""Closed-set protocol, training membership and batch construction for OmniDFA."""
from copy import copy, deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler, Subset


PROTOCOL = {
    'upstream_commit': '35b9052e83e05436682095818693493f79da9458',
    'attribution_protocol': 'closed_set_train_prototypes_v1',
    'prototype_source': 'clean_train',
    'preprocessing': 'paper_augment_two_view_deterministic_eval_v1',
    'augmentation_policy': 'omnidfa_paper',
    'output_label_space': 'generators',
}


def configure_omnidfa(config, dataset, batch_size, num_epochs):
    if batch_size < 18 or batch_size % 9:
        raise ValueError('OmniDFA batch_size must be a multiple of 9, at least 18 (8 fake : 1 real)')
    if num_epochs < 1:
        raise ValueError('OmniDFA needs a positive epoch budget')
    if config.get('precision') not in ('float32', 'bfloat16'):
        raise ValueError('OmniDFA precision must be float32 or bfloat16')
    if config.get('ddp'):
        raise ValueError('OmniDFA uses single-process DataParallel; do not enable the harness DDP option')
    config.update(PROTOCOL, dataset=dataset, omnidfa_num_epochs=num_epochs)
    if dataset == 'iab':
        from comparison.dataset.ImageAttributionDataset.dataset import model_class_to_label
        config['class_names'] = list(model_class_to_label)
        config['num_classes'] = len(model_class_to_label)
        real = [i for i, name in enumerate(config['class_names']) if name.casefold() == 'real']
        if len(real) != 1:
            raise ValueError('OmniDFA needs exactly one real class')
        config['real_class_index'] = real[0]


def dataset_rows(dataset):
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(base))
    return base, [base.samples[i] for i in indices]


def membership_digest(dataset):
    base, rows = dataset_rows(dataset)
    if base.iabench_source is None:
        rows = [(str(Path(row[0]).relative_to(base.root_dir)), *row[1:]) for row in rows]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


class OmniDFABatchSampler(Sampler):
    """Visit every fake row, pad the final batch, and cycle shuffled real rows."""
    def __init__(self, labels, real_index, batch_size, seed=42):
        if batch_size < 18 or batch_size % 9:
            raise ValueError('OmniDFA batch_size must be a multiple of 9, at least 18')
        self.real = [i for i, label in enumerate(labels) if label == real_index]
        self.fake = [i for i, label in enumerate(labels) if label != real_index]
        if not self.real or not self.fake:
            raise ValueError('OmniDFA training requires both real and fake rows')
        self.real_size = batch_size // 9
        self.fake_size = 8 * self.real_size
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return math.ceil(len(self.fake) / self.fake_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def shuffled_cycle(indices, generator):
        while True:
            for i in torch.randperm(len(indices), generator=generator).tolist():
                yield indices[i]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        fake = [self.fake[i] for i in torch.randperm(len(self.fake), generator=generator).tolist()]
        real = self.shuffled_cycle(self.real, generator)
        padding = self.shuffled_cycle(self.fake, generator)
        for start in range(0, len(fake), self.fake_size):
            batch = fake[start:start + self.fake_size]
            batch += [next(padding) for _ in range(self.fake_size - len(batch))]
            yield batch + [next(real) for _ in range(self.real_size)]


def prepare_omnidfa_loaders(train_loader, config, batch_size):
    dataset = train_loader.dataset
    base, rows = dataset_rows(dataset)
    if base.mode != 'train' or (base.iabench_source is not None and base.iabench_source.split != 'train'):
        raise ValueError('OmniDFA prototypes and optimization require training membership')
    counts = torch.bincount(torch.tensor([row[1] for row in rows]), minlength=config['num_classes'])
    if len(counts) != config['num_classes'] or (counts == 0).any():
        raise ValueError('Every OmniDFA class must have training images')
    sampler = OmniDFABatchSampler([row[1] for row in rows], config['real_class_index'],
                                  batch_size, config.get('manualSeed', 42))
    clean = copy(base)
    clean.set_val()  # Change transforms only; retain exact training row membership.
    clean_dataset = Subset(clean, list(dataset.indices)) if isinstance(dataset, Subset) else clean
    common = dict(num_workers=train_loader.num_workers, collate_fn=train_loader.collate_fn,
                  pin_memory=train_loader.pin_memory)
    training = DataLoader(dataset, batch_sampler=sampler, **common)
    support = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **common)
    config['steps_per_epoch'] = len(training)
    metadata = deepcopy(config.get('checkpoint_metadata', {}))
    if config['dataset'] == 'iab':
        metadata.update(dataset='iab', dataset_digest=membership_digest(base),
                        split_config={key: config[key] for key in (
                            'manualSeed', 'use_semantic_split', 'task_id', 'num_images_per_semantic_per_class')})
    metadata.update(PROTOCOL, model_name='omnidfa', class_names=config['class_names'],
                    num_classes=config['num_classes'], output_class_names=config['class_names'],
                    batch_size=batch_size, support_digest=membership_digest(dataset),
                    support_counts=counts.tolist())
    metadata['training_config'] = {key: deepcopy(value) for key, value in config.items()
                                   if key not in ('checkpoint_metadata', 'log_dir', 'pretrained_path',
                                                  'nEpochs', 'start_epoch')}
    config['checkpoint_metadata'] = metadata
    return training, support


def validate_omnidfa_checkpoint(checkpoint):
    expected = dict(PROTOCOL)
    # Older training policies remain evaluable; resume still compares current metadata.
    if (checkpoint.get('preprocessing') == 'upstream_train_deterministic_eval_v1'
            and checkpoint.get('augmentation_policy') == 'omnidfa_upstream'):
        expected.update(preprocessing='upstream_train_deterministic_eval_v1',
                        augmentation_policy='omnidfa_upstream')
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f'Incompatible OmniDFA checkpoint {key}')
    names = checkpoint.get('class_names')
    config = checkpoint.get('training_config')
    if (not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names) or not isinstance(config, dict)):
        raise ValueError('OmniDFA checkpoint requires ordered class names and training_config')
    if checkpoint.get('model_name') != 'omnidfa' or checkpoint.get('dataset') not in ('iab', 'iabench'):
        raise ValueError('OmniDFA checkpoint method or dataset is incompatible')
    real = [i for i, name in enumerate(names) if name.casefold() == 'real']
    if len(real) != 1 or config.get('real_class_index') != real[0]:
        raise ValueError('OmniDFA checkpoint real class mapping is incompatible')
    for key in ('class_names', 'num_classes', 'model_name', 'dataset', *PROTOCOL):
        if config.get(key) != checkpoint.get(key):
            raise ValueError(f'Incompatible OmniDFA training_config {key}')
    if checkpoint.get('num_classes') != len(names) or checkpoint.get('output_class_names') != names:
        raise ValueError('OmniDFA checkpoint output order is incompatible')
    state = checkpoint.get('model_state_dict', {})
    bank, counts = state.get('prototypes'), state.get('prototype_counts')
    if (not isinstance(bank, torch.Tensor) or not bank.is_floating_point() or bank.shape != (len(names), 128)
            or not torch.isfinite(bank).all()
            or not torch.allclose(bank.norm(dim=1), torch.ones(len(names)), atol=1e-4)
            or not isinstance(counts, torch.Tensor) or counts.shape != (len(names),)
            or counts.dtype != torch.long or (counts <= 0).any()
            or counts.tolist() != checkpoint.get('support_counts')
            or not isinstance(checkpoint.get('support_digest'), str)):
        raise ValueError('OmniDFA checkpoint requires a complete training prototype bank and membership')
    return deepcopy(config)


def prepare_omnidfa_iab_test(checkpoint, args):
    from comparison.dataset.ImageAttributionDataset.dataset import ImageAttributionDataset, model_class_to_label
    config = validate_omnidfa_checkpoint(checkpoint)
    expected = dict(manualSeed=config['manualSeed'], use_semantic_split=args.use_semantic_split,
                    task_id=args.task_id, num_images_per_semantic_per_class=args.num_images_per_semantic_per_class)
    if (checkpoint.get('dataset') != 'iab' or list(model_class_to_label) != checkpoint['class_names']
            or checkpoint.get('split_config') != expected):
        raise ValueError('IAB OmniDFA checkpoint class mapping or split settings do not match evaluation')
    source = ImageAttributionDataset(args.root_dir, args.num_images_per_semantic_per_class)
    if membership_digest(source) != checkpoint.get('dataset_digest'):
        raise ValueError('IAB OmniDFA dataset membership does not match its checkpoint')
    return config
