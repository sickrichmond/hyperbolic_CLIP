"""Test-only baseline evaluation and metric parity, without GPUs or model downloads."""
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
import numpy as np
from PIL import Image as PILImage
import torch
from torch import nn
from torchvision import transforms

from data.degradations import apply_degradation
from data.export_iabench import export_images
from data.iabench_dataset import IABenchDataset, manifest_digest
from comparison.dataset.ImageAttributionDataset.dataset import ImageAttributionDataset
from comparison.training import test as evaluator
from comparison.training.attributors import ATTRIBUTOR
from comparison.training.iabench import prepare_iabench, prepare_iabench_test, get_iabench_test_loader
from comparison.training.trainer.trainer import Trainer
from comparison.training.metrics.base_metrics_class import calculate_metrics_for_test
from tests.test_iabench_baselines import METHODS, config_for

NATIVE_METRICS = {method: ATTRIBUTOR[method].compute_metrics for method in METHODS}
REPMIX_CLASSIFIER = ATTRIBUTOR['repmix'].classifier


class EvaluationFixture(nn.Module):
    """Small features with the native metric functions and RepMix detection gate."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(3, config['num_classes'])
        self.softmax = self.act = nn.Softmax(dim=1)
        if config['model_name'] == 'repmix':
            self.attribution = self.fc
            self.detection = nn.Linear(3, 2)
        if config['model_name'] == 'defl':
            self.semantic_extractor = nn.Identity()
            self.semantic_extractor.clip_preprocess = transforms.ToTensor()

    def forward(self, batch, inference=False):
        assert inference, 'evaluation must use inference mode'
        assert not self.training
        pixels = batch.get('image', batch.get('x'))
        features = pixels.mean(dim=(-2, -1))
        method = self.config['model_name']
        if method == 'repmix':
            assert batch['label'].shape[1] == 1, 'no mixing at test time'
            logits, _ = REPMIX_CLASSIFIER(self, features)
        else:
            logits = self.fc(features)
        if method == 'patch':
            logits = logits[:, :, None, None].expand(-1, -1, 3, 2)
        return {'logits': logits}

    def compute_metrics(self, batch, pred, test=False):
        return NATIVE_METRICS[self.config['model_name']](self, batch, pred, test=test)


class BaselineEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / 'source'
        cls.names = ['Real', 'R3GAN', 'cogview4']
        y, x = np.mgrid[:270, :260]
        pixels = np.stack(((x * 17 + y * 3) % 256, (x * 7 + y * 13) % 256,
                           ((x // 3 + y // 4) % 2) * 255), axis=-1).astype(np.uint8)
        cls.image = PILImage.fromarray(pixels)
        stream = BytesIO()
        cls.image.save(stream, format='PNG')
        labels = [name for name in cls.names for _ in range(10)]
        dataset = Dataset.from_dict({
            'image': [{'bytes': stream.getvalue(), 'path': None}] * len(labels),
            'label': labels, 'file_name': [f'{i}.png' for i in range(len(labels))]},
            features=Features({'image': Image(), 'label': Value('string'), 'file_name': Value('string')}))
        dataset.save_to_disk(str(cls.root / 'data'), num_shards=2)
        source = IABenchDataset(str(cls.root), processor_name=None)
        cls.manifest = source.make_split_manifest(generators=cls.names)
        cls.manifest_path = cls.root / 'splits.json'
        cls.manifest_path.write_text(json.dumps(cls.manifest))
        cls.exported = Path(cls.tmp.name) / 'export'
        export_images(str(cls.root), cls.exported)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def checkpoint(self, method):
        config = config_for(method)
        if method == 'dna':
            config.update(resize_size=[64, 64], multi_size=[[16, 16]] * 2)
        prepare_iabench(str(self.root), str(self.manifest_path), config, 8)
        model = EvaluationFixture(config)
        return dict(config['checkpoint_metadata'], model_state_dict=model.state_dict())

    def test_exact_membership_and_degradation_pixels(self):
        checkpoint = self.checkpoint('resnet50')
        with patch.dict(os.environ, IAB_EXCLUDE_GENERATORS=','.join(self.names)):
            sources = [prepare_iabench_test(str(root), str(self.manifest_path), checkpoint)[1]
                       for root in (self.root, self.exported)]
        self.assertEqual(sources[0].indices, self.manifest['test'])
        self.assertEqual(sources[1].indices, self.manifest['test'])
        for source in sources:
            dataset = ImageAttributionDataset('', iabench_source=source)
            dataset.set_test()
            outputs = []
            for level in range(7):
                dataset.degraded = level
                actual = dataset[0]['image']
                expected = apply_degradation(self.image, level)
                self.assertEqual(actual.tobytes(), expected.tobytes())
                outputs.append(actual.tobytes())
                self.assertEqual(dataset.samples[0][0], self.manifest['test'][0])
            self.assertEqual(len(set(outputs)), 7)

    def test_all_baselines_and_all_levels_read_only_test(self):
        original_read = IABenchDataset._read_image
        original_split = IABenchDataset.split_view
        original_loader = get_iabench_test_loader
        reads, memberships, batches = [], [], []
        def read(source, row):
            self.assertEqual(source.split, 'test')
            self.assertIn(row, self.manifest['test'])
            reads.append(row)
            return original_read(source, row)
        def split(source, manifest, name):
            self.assertEqual(name, 'test', 'must never construct a train/validation view')
            return original_split(source, manifest, name)
        def loader(*args, **kwargs):
            result = original_loader(*args, **kwargs)
            self.assertFalse(result.drop_last)
            memberships.append([sample[0] for sample in result.dataset.samples])
            return result
        original_forward = EvaluationFixture.forward
        def forward(model, batch, inference=False):
            batches.append(len(batch['label']))
            return original_forward(model, batch, inference)
        for method in METHODS:
            with self.subTest(method=method):
                checkpoint = self.checkpoint(method)
                path = Path(self.tmp.name) / f'{method}.pth'
                torch.save(checkpoint, path)
                reads.clear(); memberships.clear(); batches.clear()
                for root in (self.root, self.exported):
                    logdir = Path(self.tmp.name) / f'results_{method}_{root.name}'
                    with patch.dict(ATTRIBUTOR.data, {method: EvaluationFixture}), \
                            patch.object(IABenchDataset, '_read_image', read), \
                            patch.object(IABenchDataset, 'split_view', split), \
                            patch('comparison.training.iabench.get_iabench_test_loader', side_effect=loader), \
                            patch.object(EvaluationFixture, 'forward', forward), \
                            patch.object(evaluator, 'get_dataloader', side_effect=AssertionError('legacy split')), \
                            patch('clip.load', side_effect=AssertionError('second CLIP load')), \
                            patch.object(evaluator, 'create_logger', return_value=logging.getLogger('evaluation-fixture')):
                        evaluator.main(['--dataset', 'iabench', '--root_dir', str(root),
                            '--split_manifest', str(self.manifest_path), '--resume_checkpoint', str(path),
                            '--batch_size', '2', '--num_workers', '0', '--level_start', '0', '--level_end', '7',
                            '--log_dir', str(logdir)])
                    self.assertEqual(len(list(logdir.glob('test_results_degraded_*.txt'))), 7)
                    for level in range(7):
                        text = (logdir / f'test_results_degraded_{level}.txt').read_text()
                        for metric in ('acc', 'balanced_accuracy', 'auc', 'ap', 'precision_macro',
                                       'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted',
                                       'f1_weighted', 'precision_per_class', 'recall_per_class',
                                       'f1_per_class', 'support_per_class', 'conf_matrix'):
                            self.assertIn(f'{metric}:', text)
                        self.assertIn('support_per_class: [1, 1, 1]', text)
                        self.assertIn(f'class_names: {json.dumps(self.names)}', text)
                        self.assertIn(f'manifest_digest: {manifest_digest(self.manifest)}', text)
                        self.assertIn('split: test', text)
                        self.assertNotIn('semantic_acc:', text)
                        if root == self.exported:
                            arrow_text = (Path(self.tmp.name) / f'results_{method}_source/test_results_degraded_{level}.txt').read_text()
                            self.assertEqual(text.splitlines()[1:], arrow_text.splitlines()[1:])
                self.assertEqual(reads, self.manifest['test'] * 14)
                self.assertEqual(memberships, [self.manifest['test']] * 14)
                self.assertEqual(batches, [2, 1] * 14)

    def test_checkpoint_and_manifest_rejections_before_model_loading(self):
        checkpoint = self.checkpoint('hifi_net')
        path = Path(self.tmp.name) / 'invalid.pth'
        invalid_manifest = Path(self.tmp.name) / 'invalid.json'
        for key, value in (('dataset', 'iab'), ('class_names', list(reversed(self.names))),
                           ('dataset_digest', 'wrong'), ('manifest_digest', 'wrong'),
                           ('output_label_space', 'transformations'), ('num_classes', 170),
                           ('hierarchy_sizes', [2, 4, 6, 3]), ('model_name', 'unknown'),
                           ('training_config', None)):
            with self.subTest(key=key):
                torch.save(dict(checkpoint, **{key: value}), path)
                with patch.dict(ATTRIBUTOR.data, {'hifi_net': lambda c: self.fail('model constructed')}):
                    with self.assertRaises(ValueError):
                        evaluator.main(['--dataset', 'iabench', '--root_dir', str(self.root),
                            '--split_manifest', str(self.manifest_path), '--resume_checkpoint', str(path)])
        damaged = deepcopy(self.manifest)
        damaged['test'][0] = damaged['train'][0]
        invalid_manifest.write_text(json.dumps(damaged))
        damaged_checkpoint = dict(checkpoint, manifest_digest=manifest_digest(damaged))
        with self.assertRaisesRegex(ValueError, 'membership'):
            prepare_iabench_test(str(self.root), str(invalid_manifest), damaged_checkpoint)
        config = config_for('dna', pretrain=True)
        prepare_iabench(str(self.root), str(self.manifest_path), config, 32)
        with self.assertRaisesRegex(ValueError, 'num_classes|output_label_space|train_stage'):
            prepare_iabench_test(str(self.root), str(self.manifest_path), config['checkpoint_metadata'])
        with self.assertRaises(ValueError):
            prepare_iabench_test(str(self.root), None, checkpoint)
        ucf = self.checkpoint('ucf')
        ucf['training_config']['backbone_config']['num_classes'] = 3
        with self.assertRaisesRegex(ValueError, 'two-class detection'):
            prepare_iabench_test(str(self.root), str(self.manifest_path), ucf)
        config, source = prepare_iabench_test(str(self.root), str(self.manifest_path), self.checkpoint('dna'))
        self.assertFalse(config['load_param'])
        wrong_source = source.split_view(self.manifest, 'val')
        with self.assertRaises(ValueError):
            get_iabench_test_loader(wrong_source, config, 2, 0)
        good = self.checkpoint('resnet50')
        good['model_state_dict']['fc.weight'] = torch.zeros(4, 3)
        torch.save(good, path)
        with patch.dict(ATTRIBUTOR.data, {'resnet50': EvaluationFixture}), \
                patch.object(evaluator, 'create_logger', return_value=logging.getLogger('strict-weights')):
            with self.assertRaisesRegex(RuntimeError, 'size mismatch'):
                evaluator.main(['--dataset', 'iabench', '--root_dir', str(self.root),
                    '--split_manifest', str(self.manifest_path), '--resume_checkpoint', str(path),
                    '--log_dir', str(Path(self.tmp.name) / 'strict-results')])

    def test_patch_reduction_matches_native_metrics_with_optional_semantics(self):
        from comparison.training.attributors.attributor_patch import PatchAttributor
        logits = torch.randn(7, 3, 4, 5, generator=torch.Generator().manual_seed(5))
        labels = torch.tensor([0, 1, 2, 1, 0, 1, 2])
        model = EvaluationFixture({'model_name': 'patch', 'num_classes': 3})
        for semantics in (None, torch.arange(7) % 2):
            reference = PatchAttributor.compute_metrics(model, {'label': labels, 'semantic_label': semantics},
                                                        {'logits': logits}, test=True)
            rows = [{'image': logits[i], 'label': labels[i],
                     **({'semantic_label': semantics[i]} if semantics is not None else {})} for i in range(7)]
            trainer = Trainer(model.config, model, None, None, logging.getLogger('patch-parity'),
                              test_mode=True, log_dir=self.tmp.name)
            with patch.object(model, 'forward', side_effect=lambda batch, inference: {'logits': batch['image']}):
                with patch('comparison.training.trainer.trainer.calculate_metrics_for_test',
                           wraps=calculate_metrics_for_test) as metrics:
                    actual = trainer.test(torch.utils.data.DataLoader(rows, batch_size=3))
                    self.assertEqual(tuple(metrics.call_args.args[1].shape), (7, 3))
            for key in reference:
                if key == 'semantic_acc':
                    self.assertEqual(reference[key], actual[key])
                else:
                    np.testing.assert_allclose(reference[key], actual[key], atol=1e-7)

    def test_iab_defaults_and_cli_constraints(self):
        args = evaluator.parser.parse_args(['--resume_checkpoint', 'file.pth'])
        self.assertEqual((args.dataset, args.batch_size, args.num_workers, args.level_start, args.level_end),
                         ('iab', 32, 2, 0, 1))
        for options in (['--level_end', '8'], ['--batch_size', '0'], ['--num_workers', '-1'],
                        ['--dataset', 'iabench', '--use_semantic_split']):
            with self.assertRaises(ValueError):
                evaluator.main(['--resume_checkpoint', 'file.pth', *options])

    def test_repmix_gate_is_independent_of_the_iab_real_label(self):
        from comparison.training.attributors.attributor_repmix import model_class_to_label
        model = EvaluationFixture({'model_name': 'repmix', 'num_classes': 3, 'real_class_index': 0})
        with torch.no_grad():
            model.attribution.weight.zero_()
            model.attribution.bias.fill_(1)
            model.detection.weight.zero_()
            model.detection.bias.copy_(torch.tensor([50., -50.]))
        with patch.dict(model_class_to_label, {'other': 0}, clear=True):
            gated, _ = REPMIX_CLASSIFIER(model, torch.zeros(1, 3))
        self.assertGreater(gated[0, 0].item(), .99)
        self.assertLess(gated[0, 1:].abs().max().item(), .01)


if __name__ == '__main__':
    unittest.main()
