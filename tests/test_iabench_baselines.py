"""Baseline split, adapter, head and resume checks without downloads or GPUs.

Run: python -m tests.test_iabench_baselines
"""
from contextlib import ExitStack
from copy import deepcopy
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
import torch
from torch import nn
from torchvision import transforms
import yaml
import numpy as np

from data.export_iabench import export_images
from data.iabench_dataset import IABenchDataset
from comparison.dataset.ImageAttributionDataset import DATASET
from comparison.dataset.ImageAttributionDataset.dataset import _IABENCH_HIERARCHY, hifi_label_mapping
from comparison.training.attributors import ATTRIBUTOR
from comparison.training.iabench import (
    prepare_iabench, get_iabench_dataloaders, validate_dna_stage_one)
from comparison.training import train
from comparison.training.trainer.trainer import Trainer

METHODS = ('resnet50', 'dct', 'hifi_net', 'defl', 'dna', 'repmix', 'patch', 'ucf')
CONFIGS = Path('comparison/training/config/model')


def config_for(method, pretrain=False):
    name = 'dna_pretrain' if pretrain else 'dna_default' if method == 'dna' else method
    return yaml.safe_load((CONFIGS / f'{name}.yaml').read_text())


def make_data(root, names):
    stream = BytesIO()
    PILImage.new('RGB', (260, 270), (40, 80, 100)).save(stream, format='PNG')
    labels = [name for name in names for _ in range(10)]
    data = Dataset.from_dict({
        'image': [{'bytes': stream.getvalue(), 'path': None}] * len(labels),
        'label': labels, 'file_name': [f'{i}.png' for i in range(len(labels))]},
        features=Features({'image': Image(), 'label': Value('string'), 'file_name': Value('string')}))
    data.save_to_disk(str(root / 'data'), num_shards=2)
    source = IABenchDataset(str(root), processor_name=None)
    manifest = source.make_split_manifest(generators=names)
    path = root / 'splits.json'
    path.write_text(json.dumps(manifest))
    return path, manifest


class TinyAttributor(nn.Module):
    """Only replace model computation; main, adapters and Trainer remain real."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(3, config['num_classes'])
        if config['model_name'] == 'defl':
            self.semantic_extractor = nn.Identity()
            self.semantic_extractor.clip_preprocess = transforms.ToTensor()

    def forward(self, batch, inference=False):
        images = batch.get('image', batch.get('x'))
        features = images.mean(dim=(-2, -1))
        if self.config['model_name'] == 'repmix' and not inference:
            features = features.reshape(-1, self.config['mixup_samples'], 3).mean(dim=1)
        return {'logits': self.fc(features)}

    def compute_losses(self, batch, pred):
        label = batch['label']
        if label.ndim == 2:
            label = label[:, 0]
        return {'overall': nn.functional.cross_entropy(pred['logits'], label)}

    def compute_metrics(self, batch, pred):
        label = batch['label']
        if label.ndim == 2:
            label = label[:, 0]
        from comparison.training.metrics.base_metrics_class import calculate_metrics_for_train
        auc, acc, ap = calculate_metrics_for_train(label.detach(), pred['logits'].detach())
        return {'auc': auc, 'acc': acc, 'ap': ap}


class TinyReconstruction(nn.Module):
    def forward(self, forgery, content):
        return forgery.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1).expand(-1, 3, 256, 256)


class BaselineIABenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / 'source'
        cls.names = ['Real', *sorted(_IABENCH_HIERARCHY)]
        cls.path, cls.manifest = make_data(cls.root, cls.names)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def prepare(self, method, root=None, pretrain=False, batch_size=2):
        config = config_for(method, pretrain)
        sources = prepare_iabench(str(root or self.root), str(self.path), config, batch_size)
        return config, sources

    def test_arrow_export_membership_and_rejection(self):
        exported = Path(self.tmp.name) / 'export'
        export_images(str(self.root), exported)
        with patch.dict(os.environ, IAB_EXCLUDE_GENERATORS=','.join(self.names)):
            config, (arrow, val) = self.prepare('resnet50')
            _, (images, image_val) = self.prepare('resnet50', exported)
        self.assertEqual(config['num_classes'], 29)
        self.assertEqual(config['class_names'], self.names)
        self.assertEqual(arrow.indices, self.manifest['train'])
        self.assertEqual(val.indices, self.manifest['val'])
        self.assertEqual(arrow.indices, images.indices)
        self.assertEqual(val.indices, image_val.indices)
        self.assertIs(arrow.data, val.data)
        self.assertIs(arrow.source_labels, val.source_labels)
        self.assertEqual(arrow._read_image(arrow.indices[0]).tobytes(),
                         images._read_image(images.indices[0]).tobytes())
        for field in ('dataset_digest', 'class_names', 'train'):
            damaged = deepcopy(self.manifest)
            if field == 'dataset_digest':
                damaged[field] = 'wrong'
            elif field == 'class_names':
                damaged[field].reverse()
            else:
                damaged[field][0] = damaged['test'][0]
            path = Path(self.tmp.name) / 'bad.json'
            path.write_text(json.dumps(damaged))
            with self.assertRaises(ValueError):
                prepare_iabench(str(self.root), str(path), config_for('resnet50'), 2)
        with self.assertRaises(ValueError):
            prepare_iabench(str(self.root), None, config_for('resnet50'), 2)
        with self.assertRaises(FileNotFoundError):
            prepare_iabench(str(self.root), '/missing/manifest', config_for('resnet50'), 2)

    def test_all_adapter_batches_and_training_partners(self):
        for method in METHODS:
            with self.subTest(method=method):
                config, sources = self.prepare(method)
                if method == 'dna':
                    config.update(resize_size=[64, 64], multi_size=[[16, 16]] * 2)
                with patch('clip.load', side_effect=AssertionError('second CLIP model loaded')):
                    loaders = get_iabench_dataloaders(*sources, config, 2, 0, transforms.ToTensor())
                    train_loader, val_loader = loaders
                    self.assertEqual([s[0] for s in train_loader.dataset.samples], self.manifest['train'])
                    self.assertEqual([s[0] for s in val_loader.dataset.samples], self.manifest['val'])
                    batch = next(iter(train_loader))
                    val = next(iter(val_loader))
                self.assertNotIn('semantic_label', batch)
                self.assertNotIn('semantic_label', val)
                self.assertEqual(val['label'].flatten().tolist(), [0, 1])
                if method == 'repmix':
                    self.assertEqual(tuple(batch['x'].shape), (4, 3, 224, 224))
                    self.assertEqual(tuple(val['x'].shape), (2, 3, 224, 224))
                    self.assertEqual(tuple(batch['label'].shape), (2, 2))
                    self.assertEqual(val['y_det'].flatten().tolist(), [0, 1])
                    read_rows = []
                    read = sources[0]._read_image
                    def tracked(row):
                        read_rows.append(row)
                        return read(row)
                    with patch.object(sources[0], '_read_image', side_effect=tracked), \
                            patch('numpy.random.choice', return_value=np.array([len(sources[0]) - 1])):
                        train_loader.dataset[0]
                    self.assertEqual(read_rows, [self.manifest['train'][0], self.manifest['train'][-1]])
                    self.assertTrue(set(read_rows) <= set(self.manifest['train']))
                else:
                    size = 224 if method == 'resnet50' else 64 if method == 'dna' else 256
                    self.assertEqual(tuple(batch['image'].shape), (2, 3, size, size))
                if method == 'hifi_net':
                    self.assertEqual(len(batch['hierachi_label']), 4)
                    self.assertEqual([v[0].item() for v in val['hierachi_label']], [1, 2, 5, 0])
                if method == 'defl':
                    self.assertIn('clip_image', batch)
                    self.assertEqual(val['method_label'][0].item(), 5)
                if method == 'dna':
                    self.assertEqual(len(batch['crops']), 2)
                    self.assertEqual(train_loader.dataset.class_num, 170)
                if method == 'ucf':
                    self.assertEqual(val['label_det'].tolist(), [0, 1])
                    self.assertTrue(torch.equal(val['label_spe'], val['label']))
        config, sources = self.prepare('resnet50')
        dataset = DATASET['resnet50']('', iabench_source=sources[0])
        read = sources[0]._read_image
        row = sources[0].indices[0]
        with patch.object(sources[0], '_read_image', side_effect=[OSError('busy'), OSError('busy'), read(row)]) as reader:
            self.assertEqual(dataset._open_image(row, backoff=0).size, (260, 270))
        self.assertEqual([call.args for call in reader.call_args_list], [(row,)] * 3)

    def test_ucf_batches_and_taxonomy(self):
        config, sources = self.prepare('ucf', batch_size=32)
        loader, _ = get_iabench_dataloaders(*sources, config, 32, 0)
        self.assertEqual(len(loader.dataset), 232)
        self.assertEqual(len(loader), 7)
        self.assertTrue(loader.drop_last)
        self.assertEqual(len(set(loader.sampler)), 232)
        for size in (1, 3, 256):
            with self.assertRaises(ValueError):
                self.prepare('ucf', batch_size=size)
        mapping = hifi_label_mapping(self.names, dataset='iabench')
        by_name = dict(zip(self.names, mapping))
        for name in ('E4S', 'R3GAN', 'StyleGAN-XL', 'stylegan3'):
            self.assertEqual(by_name[name][:3], (0, 1, 6))
        self.assertEqual(by_name['cogview4'][:3], (0, 1, 3))
        with self.assertRaisesRegex(ValueError, 'Unmapped'):
            hifi_label_mapping(['real', 'unknown'], dataset='iabench')

    def test_native_classifier_dimensions_with_mocked_backbones(self):
        from comparison.training.attributors.attributor_ucf import UCFAttributor
        from comparison.training.attributors.attributor_patch import PatchAttributor
        for method in METHODS:
            with self.subTest(method=method), ExitStack() as stack:
                config, sources = self.prepare(method)
                if method == 'resnet50':
                    stack.enter_context(patch('torchvision.models.resnet50', return_value=nn.Identity()))
                elif method == 'hifi_net':
                    stack.enter_context(patch('comparison.training.attributors.attributor_hifi_net.get_seg_model', return_value=nn.Identity()))
                elif method == 'defl':
                    stack.enter_context(patch('comparison.training.attributors.attributor_defl.DEFLNetwork', return_value=nn.Identity()))
                    stack.enter_context(patch('comparison.training.attributors.attributor_defl.SemanticFeatureExtractor', return_value=nn.Identity()))
                elif method == 'dna':
                    config['load_param'] = False
                    config.update(resize_size=[64, 64], multi_size=[[16, 16]] * 2)
                elif method == 'repmix':
                    stack.enter_context(patch('comparison.training.attributors.attributor_repmix.ResnetMixup', return_value=nn.Identity()))
                elif method == 'patch':
                    stack.enter_context(patch('comparison.training.attributors.attributor_patch.networks.define_patch_D',
                                             side_effect=lambda *args, num_class: nn.Linear(3, num_class)))
                    stack.enter_context(patch.object(PatchAttributor, 'build_loss', return_value=None))
                elif method == 'ucf':
                    stack.enter_context(patch.object(UCFAttributor, 'build_backbone', return_value=nn.Identity()))
                    stack.enter_context(patch('comparison.training.attributors.attributor_ucf.Conditional_UNet', return_value=TinyReconstruction()))
                model = ATTRIBUTOR[method](config)
                head = {'resnet50': lambda: model.fc, 'dct': lambda: model.feature_extractor.fc,
                        'defl': lambda: model.nn_classifier.fc3, 'dna': lambda: model.backbone.classification_head[-1],
                        'hifi_net': lambda: model.SegNet.branch_cls_level_4.fc,
                        'repmix': lambda: model.attribution, 'patch': lambda: model.net_D,
                        'ucf': lambda: model.head_spe.mlp[-1]}[method]()
                self.assertEqual(head.out_features, 29)
                if method == 'hifi_net':
                    self.assertEqual(model.SegNet.branch_cls_level_3.fc.out_features, 7)
                    self.assertEqual(model.SegNet.parent_idx_2.tolist(), [0, 0, 1, 0])
                    self.assertEqual(model.SegNet.parent_idx_3.tolist(), [0, 1, 1, 1, 1, 2, 1])
                    for row in config['hierarchy_mapping']:
                        for level in (2, 3, 4):
                            self.assertEqual(getattr(model.SegNet, f'parent_idx_{level}')[row[level - 1]].item(), row[level - 2])
                if method == 'ucf':
                    self.assertEqual(model.head_sha.mlp[-1].out_features, 2)
                    self.assertEqual(config['backbone_config']['num_classes'], 2)
                if method == 'repmix':
                    self.assertEqual(model.detection.out_features, 2)
                    with torch.no_grad():
                        model.attribution.weight.zero_()
                        model.attribution.bias.fill_(1)
                        model.detection.weight.zero_()
                        model.detection.bias.copy_(torch.tensor([50., -50.]))
                        gated, _ = model.classifier(torch.zeros(2, config['d_embed']))
                    self.assertGreater(gated[0, 0].item(), .99)
                    self.assertLess(gated[0, 1:].abs().max().item(), .01)
                # Exercise native forward/loss/validation with inexpensive backbone features.
                if method == 'dna':
                    config.update(resize_size=[64, 64], multi_size=[[16, 16]] * 2)
                _, val_loader = get_iabench_dataloaders(*sources, config, 2, 0, transforms.ToTensor())
                batch = next(iter(val_loader))
                if method in ('resnet50', 'defl'):
                    stack.enter_context(patch.object(model, 'extract_features', return_value=torch.zeros(2, 2048)))
                elif method == 'hifi_net':
                    stack.enter_context(patch.object(model, 'extract_features', return_value=None))
                    def classify(features, images):
                        return tuple(getattr(model.SegNet, f'branch_cls_level_{i}').fc(torch.zeros(2, 18))
                                     for i in range(1, 5))
                    stack.enter_context(patch.object(model, 'classifier', side_effect=classify))
                elif method == 'dna':
                    def features(images):
                        flat = torch.zeros(len(images), 512)
                        return model.backbone.classification_head(flat), nn.functional.normalize(model.head(flat), dim=1)
                    stack.enter_context(patch.object(model, 'extract_features', side_effect=features))
                elif method == 'repmix':
                    train_loader, _ = get_iabench_dataloaders(*sources, config, 2, 0)
                    batch = next(iter(train_loader))
                    stack.enter_context(patch.object(model, 'extract_features', return_value=torch.zeros(2, config['d_embed'])))
                elif method == 'patch':
                    model.criterionCE = nn.CrossEntropyLoss()
                    model.softmax = nn.Softmax(dim=1)
                    stack.enter_context(patch.object(model, 'extract_features',
                        side_effect=lambda batch: model.net_D(batch['image'].mean(dim=(-2, -1)))[:, :, None, None]))
                elif method == 'ucf':
                    stack.enter_context(patch.object(model, 'extract_features', return_value={
                        'forgery': torch.zeros(2, 512, 4, 4), 'content': torch.zeros(2, 512, 4, 4)}))
                optimizer = torch.optim.SGD(model.parameters(), lr=.001)
                trainer = Trainer(config, model, optimizer, None, logging.getLogger('native-heads'),
                                  log_dir=self.tmp.name)
                losses, pred = trainer.train_step(batch)
                self.assertTrue(torch.isfinite(losses['overall']))
                expected_shape = (2, 29, 1, 1) if method == 'patch' else (2, 29)
                self.assertEqual(tuple(pred['logits'].shape), expected_shape)
                self.assertTrue(any(p.grad is not None for p in head.parameters()))
                val_pred = trainer.inference(batch)
                self.assertEqual(tuple(val_pred['logits'].shape), expected_shape)
                self.assertIn('acc', model.compute_metrics(batch, val_pred))

    def test_dna_stage_spaces_and_metadata(self):
        config, sources = self.prepare('dna', pretrain=True)
        config.update(resize_size=[64, 64], multi_size=[[16, 16]] * 2)
        train_loader, val_loader = get_iabench_dataloaders(*sources, config, 2, 0)
        self.assertEqual(config['num_classes'], 170)
        self.assertEqual(train_loader.dataset.class_num, 170)
        with patch('comparison.dataset.ImageAttributionDataset.dataset_dna.random.randint', return_value=169):
            for loader in (train_loader, val_loader):
                self.assertEqual(next(iter(loader))['label'].tolist(), [169, 169])
        from comparison.training.attributors.attributor_dna import DNAAttributor
        model = DNAAttributor(config)
        self.assertEqual(model.backbone.classification_head[-1].out_features, 170)
        stage_one = dict(config['checkpoint_metadata'], model_state_dict=model.state_dict())
        ckpt = Path(self.tmp.name) / 'stage_one.pth'
        torch.save(stage_one, ckpt)
        stage_two, _ = self.prepare('dna')
        validate_dna_stage_one(str(ckpt), stage_two['checkpoint_metadata'])
        model.load_parameters(str(ckpt))
        for key, value in (('train_stage', 2), ('manifest_digest', 'wrong'), ('class_names', ['wrong'])):
            wrong = dict(stage_one, **{key: value})
            torch.save(wrong, ckpt)
            with self.assertRaisesRegex(ValueError, key):
                validate_dna_stage_one(str(ckpt), stage_two['checkpoint_metadata'])
        with self.assertRaises(ValueError):
            validate_dna_stage_one(None, stage_two['checkpoint_metadata'])

    def test_training_validation_and_resume_never_read_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'small'
            manifest_path, manifest = make_data(root, ['Real', 'R3GAN'])
            original_read = IABenchDataset._read_image
            reads = []
            def tracked(source, row):
                self.assertIn(row, manifest['train'] if source.split == 'train' else manifest['val'])
                reads.append((source.split, row))
                return original_read(source, row)
            with patch.object(IABenchDataset, '_read_image', tracked), \
                    patch.object(Trainer, 'test', side_effect=AssertionError('test evaluation')), \
                    patch.object(train, 'get_dataloader', side_effect=AssertionError('IAB split')), \
                    patch.object(train, 'create_logger', return_value=logging.getLogger('baseline-fixture')):
                for method in METHODS:
                    with self.subTest(method=method):
                        config = config_for(method, pretrain=method == 'dna')
                        if method == 'dna':
                            config.update(resize_size=[64, 64], multi_size=[[16, 16]] * 2)
                        config_path = Path(directory) / f'{method}.yaml'
                        config_path.write_text(yaml.safe_dump(config))
                        log = Path(directory) / method
                        argv = ['--config', str(config_path), '--dataset', 'iabench',
                                '--root_dir', str(root), '--split_manifest', str(manifest_path),
                                '--batch_size', '4', '--num_workers', '0', '--n_epoch', '1', '--log_dir', str(log)]
                        with patch.dict(ATTRIBUTOR.data, {method: TinyAttributor}):
                            train.main(argv)
                            last = next(log.rglob('ckpt_last.pth'))
                            saved = torch.load(last, weights_only=False)
                            self.assertEqual(saved['epoch'], 1)
                            self.assertEqual(saved['class_names'], ['Real', 'R3GAN'])
                            self.assertTrue((last.parent / 'ckpt_best.pth').is_file())
                            self.assertFalse(list(log.rglob('ckpt_epoch_*.pth')))
                            train.main(argv + ['--resume_checkpoint', str(last), '--n_epoch', '2'])
                            resumed = torch.load(last, weights_only=False)
                            self.assertEqual(resumed['epoch'], 2)
                            self.assertEqual(len(list(log.rglob('ckpt_last.pth'))), 1)
                            self.assertGreaterEqual(resumed['best_metrics']['val_metric'], saved['best_metrics']['val_metric'])
                            if resumed['scheduler_state_dict']:
                                self.assertEqual(resumed['scheduler_state_dict']['last_epoch'], 2)
                            steps = [v['step'].item() for v in resumed['optimizer_state_dict']['state'].values()]
                            self.assertEqual(steps, [8., 8.])
                            if method == 'dna':
                                stage_two_config = config_for('dna')
                                stage_two_config.update(resize_size=[64, 64], multi_size=[[16, 16]] * 2)
                                config_path.write_text(yaml.safe_dump(stage_two_config))
                                stage_two_argv = argv + ['--pretrained_path', str(last), '--log_dir', str(log / 'stage_two')]
                                train.main(stage_two_argv)
                                stage_two_last = next((log / 'stage_two').rglob('ckpt_last.pth'))
                                stage_two = torch.load(stage_two_last, weights_only=False)
                                self.assertEqual(stage_two['num_classes'], 2)
                                self.assertEqual(stage_two['output_label_space'], 'generators')
                                self.assertEqual(stage_two['train_stage'], 2)
                                train.main(stage_two_argv + ['--resume_checkpoint', str(stage_two_last), '--n_epoch', '2',
                                                             '--pretrained_path', '/no/stage_one/needed'])
                                self.assertEqual(torch.load(stage_two_last, weights_only=False)['epoch'], 2)
                                config_path.write_text(yaml.safe_dump(config))
                            resumed['manifest_digest'] = 'wrong'
                            torch.save(resumed, last)
                            with patch.dict(ATTRIBUTOR.data, {method: lambda c: self.fail('model constructed before validation')}):
                                with self.assertRaisesRegex(ValueError, 'manifest_digest'):
                                    train.main(argv + ['--resume_checkpoint', str(last)])
            self.assertEqual({split for split, _ in reads}, {'train', 'val'})
            self.assertFalse({row for _, row in reads} & set(manifest['test']))


if __name__ == '__main__':
    unittest.main()
