"""Closed-set protocol, training membership and batch construction for OmniDFA."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler, Subset


CROSS_ENTROPY_PROTOCOL = 'closed_set_cross_entropy_v1'
PROTOTYPE_PROTOCOL = 'closed_set_train_prototypes_v1'
PROTOCOL = {
    'upstream_commit': '35b9052e83e05436682095818693493f79da9458',
    'attribution_protocol': CROSS_ENTROPY_PROTOCOL,
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


def prepare_omnidfa_train_loader(train_loader, config, batch_size):
    dataset = train_loader.dataset
    base, rows = dataset_rows(dataset)
    if base.mode != 'train' or (base.iabench_source is not None and base.iabench_source.split != 'train'):
        raise ValueError('OmniDFA optimization requires training membership')
    labels = [row[1] for row in rows]
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=config['num_classes'])
    if len(counts) != config['num_classes'] or (counts == 0).any():
        raise ValueError('Every OmniDFA class must have training images')
    sampler = OmniDFABatchSampler(labels, config['real_class_index'],
                                  batch_size, config.get('manualSeed', 42))
    training = DataLoader(dataset, batch_sampler=sampler, num_workers=train_loader.num_workers,
                          collate_fn=train_loader.collate_fn, pin_memory=train_loader.pin_memory)
    config['steps_per_epoch'] = len(training)
    metadata = deepcopy(config.get('checkpoint_metadata', {}))
    if config['dataset'] == 'iab':
        metadata.update(dataset='iab', dataset_digest=membership_digest(base),
                        split_config={key: config[key] for key in (
                            'manualSeed', 'use_semantic_split', 'task_id', 'num_images_per_semantic_per_class')})
    metadata.update(PROTOCOL, model_name='omnidfa', class_names=config['class_names'],
                    num_classes=config['num_classes'], output_class_names=config['class_names'],
                    batch_size=batch_size, train_digest=membership_digest(dataset),
                    train_counts=counts.tolist())
    metadata['training_config'] = {key: deepcopy(value) for key, value in config.items()
                                   if key not in ('checkpoint_metadata', 'log_dir', 'pretrained_path',
                                                  'nEpochs', 'start_epoch')}
    config['checkpoint_metadata'] = metadata
    return training


def validate_omnidfa_checkpoint(checkpoint, *, for_training=False):
    expected = dict(PROTOCOL)
    legacy = checkpoint.get('attribution_protocol') == PROTOTYPE_PROTOCOL
    if legacy:
        if for_training:
            raise ValueError('OmniDFA prototype checkpoints are supported for evaluation only. '
                             'Start a fresh TwinNeXt + CE run or resume a cross-entropy checkpoint.')
        expected.update(attribution_protocol=PROTOTYPE_PROTOCOL, prototype_source='clean_train')
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
    for key in ('class_names', 'num_classes', 'model_name', 'dataset', *expected):
        if config.get(key) != checkpoint.get(key):
            raise ValueError(f'Incompatible OmniDFA training_config {key}')
    if checkpoint.get('num_classes') != len(names) or checkpoint.get('output_class_names') != names:
        raise ValueError('OmniDFA checkpoint output order is incompatible')
    state = checkpoint.get('model_state_dict')
    if not isinstance(state, dict):
        raise ValueError('OmniDFA checkpoint requires model weights')
    if legacy:
        _validate_prototype_bank(checkpoint, state, len(names))
    else:
        for key, shape in (('backbone.mlp.fc2.weight', (len(names), 512)),
                           ('backbone.mlp.fc2.bias', (len(names),))):
            value = state.get(key)
            if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                    or value.shape != shape or not torch.isfinite(value).all()):
                raise ValueError(f'Invalid OmniDFA classification head: {key}')
        counts = checkpoint.get('train_counts')
        if (not isinstance(counts, list) or len(counts) != len(names)
                or any(type(count) is not int or count <= 0 for count in counts)
                or not isinstance(checkpoint.get('train_digest'), str)):
            raise ValueError('OmniDFA checkpoint requires training membership and class counts')
    return deepcopy(config)


def _validate_prototype_bank(checkpoint, state, num_classes):
    bank, counts = state.get('prototypes'), state.get('prototype_counts')
    if (not isinstance(bank, torch.Tensor) or not bank.is_floating_point() or bank.shape != (num_classes, 128)
            or not torch.isfinite(bank).all()
            or not torch.allclose(bank.norm(dim=1), bank.new_ones(num_classes), atol=1e-4)
            or not isinstance(counts, torch.Tensor) or counts.shape != (num_classes,)
            or counts.dtype != torch.long or (counts <= 0).any()
            or counts.tolist() != checkpoint.get('support_counts')
            or not isinstance(checkpoint.get('support_digest'), str)):
        raise ValueError('OmniDFA checkpoint requires a complete training prototype bank and membership')


def omnidfa_report_provenance(config):
    provenance = {key: config[key] for key in ('attribution_protocol', 'preprocessing')}
    if config['attribution_protocol'] == PROTOTYPE_PROTOCOL:
        provenance['prototype_source'] = config['prototype_source']
    else:
        provenance['method_label'] = 'TwinNeXt + CE'
    return dict(provenance, train_augment=True, aug_policy=config['augmentation_policy'])


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
