"""Closed-set OmniDFA checks, using real adapters and a tiny downloadable-free backbone.

Run: python -m tests.test_omnidfa
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
import yaml

from comparison.training import train, test, test_baselines_styled
from comparison.training.attributors.attributor_omnidfa import OmniDFAAttributor
from comparison.training.omnidfa import (
    CROSS_ENTROPY_PROTOCOL, PROTOCOL, PROTOTYPE_PROTOCOL, OmniDFABatchSampler,
    configure_omnidfa, prepare_omnidfa_train_loader,
    validate_omnidfa_checkpoint,
)
from comparison.training.iabench import prepare_iabench, get_iabench_dataloaders, prepare_iabench_test
from comparison.training.trainer.omnidfa_trainer import OmniDFATrainer
from comparison.dataset.ImageAttributionDataset.dataset import model_class_to_label
from comparison.dataset.ImageAttributionDataset.dataset_omnidfa import OmniDFADataset
from data.degradations import apply_degradation, omnidfa_augment
from data.export_iabench import export_images
from data import omnifake
from tests.test_iabench_baselines import make_data


class TinyTwin(nn.Module):
    def __init__(self, pretrained=True, out_features=128):
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(6, 512))
        self.mlp.add_module('act', nn.GELU())
        self.mlp.add_module('fc2', nn.Linear(512, out_features))

    def forward_features(self, local, global_image):
        return torch.cat((local.mean((2, 3)), global_image.mean((2, 3))), dim=1)


def configuration():
    return yaml.safe_load(Path('comparison/training/config/model/omnidfa.yaml').read_text())


def legacy_checkpoint(checkpoint, upstream_augmentation=False):
    legacy = deepcopy(checkpoint)
    protocol = dict(attribution_protocol=PROTOTYPE_PROTOCOL, prototype_source='clean_train')
    if upstream_augmentation:
        protocol.update(preprocessing='upstream_train_deterministic_eval_v1',
                        augmentation_policy='omnidfa_upstream')
    legacy.update(protocol)
    legacy['training_config'].update(protocol)
    legacy['support_digest'] = legacy.pop('train_digest')
    legacy['support_counts'] = legacy.pop('train_counts')
    state = {f'backbone.{key}': value for key, value in TinyTwin().state_dict().items()}
    state.update(prototypes=F.normalize(torch.randn(legacy['num_classes'], 128), dim=1),
                 prototype_counts=torch.tensor(legacy['support_counts']))
    state['center_loss.center_real'] = torch.randn(128)
    state['center_loss.cosine_threshold'] = torch.tensor(0.5)
    legacy['model_state_dict'] = state
    return legacy


class OmniDFATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_legacy_classifier_loads_old_state_and_scores_cosine(self):
        checkpoint = legacy_checkpoint(dict(num_classes=3, training_config={}, train_digest='train',
                                            train_counts=[3, 4, 5]))
        config = dict(checkpoint['training_config'], num_classes=3)
        with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', TinyTwin):
            model = OmniDFAAttributor(config)
        state = checkpoint['model_state_dict']
        model.load_state_dict(state, strict=True)
        batch = dict(image=torch.rand(2, 3, 32, 32), global_image=torch.rand(2, 3, 32, 32))
        features = model.backbone.forward_features(batch['image'], batch['global_image'])
        embeddings = F.normalize(model.backbone.mlp(features), dim=1)
        torch.testing.assert_close(model(batch, inference=True)['logits'],
                                   embeddings @ state['prototypes'].T)

    def test_sampler_coverage_ratio_and_epoch_reproducibility(self):
        labels = [0] * 35 + [1] * 3
        sampler = OmniDFABatchSampler(labels, 1, 18, seed=4)
        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        self.assertTrue(set(range(35)).issubset({i for batch in batches for i in batch[:16]}))
        for batch in batches:
            self.assertEqual(len(batch), 18)
            self.assertTrue(all(labels[i] == 0 for i in batch[:16]))
            self.assertTrue(all(labels[i] == 1 for i in batch[16:]))
        self.assertEqual(batches, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(batches, list(sampler))
        for size in (0, 9, 32):
            with self.assertRaises(ValueError):
                OmniDFABatchSampler(labels, 1, size)

    def test_cross_entropy_updates_both_convnext_branches_and_head(self):
        model = OmniDFAAttributor(dict(num_classes=3, pretrained=False))
        backbone = model.backbone
        params = [next(backbone.backbone1.parameters()), next(backbone.backbone2.parameters()),
                  backbone.mlp.fc2.weight]
        self.assertIsNot(params[0], params[1])
        before = [param.detach().clone() for param in params]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = dict(image=torch.rand(2, 3, 32, 32), global_image=torch.rand(2, 3, 32, 32),
                     label=torch.tensor([0, 2]))
        logits = model(batch)['logits']
        self.assertEqual(logits.shape, (2, 3))
        loss = model.compute_losses(batch, {'logits': logits})['overall']
        torch.testing.assert_close(loss, F.cross_entropy(logits, batch['label']))
        loss.backward()
        optimizer.step()
        for param, original in zip(params, before):
            self.assertTrue(torch.isfinite(param).all())
            self.assertFalse(torch.equal(param, original))
        self.assertNotIn('prototypes', model.state_dict())

    def test_shared_augmentation_precedes_both_views_and_handles_small_images(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(model_class_to_label, {'R3GAN': 0, 'real': 1}, clear=True):
            root = Path(directory)
            for name in model_class_to_label:
                (root / name / 'AnimalFace/cat').mkdir(parents=True)
            image = Image.fromarray(torch.randint(0, 256, (270, 260, 3), dtype=torch.uint8).numpy())
            image.save(root / 'R3GAN/AnimalFace/cat/img_p0_i0.png')
            image.save(root / 'real/AnimalFace/cat/real.jpg')
            view = OmniDFADataset(root)
            self.assertIs(view.augment, omnidfa_augment)
            view.set_test()
            with patch.object(view, 'augment', side_effect=AssertionError('evaluation augmented')):
                for level in range(7):
                    view.degraded = level
                    result = view[0]
                    degraded = apply_degradation(image, level)
                    expected_local = T.ToTensor()(T.CenterCrop(224)(degraded))
                    expected_global = T.ToTensor()(T.CenterCrop(224)(T.Resize(224)(degraded)))
                    torch.testing.assert_close(result['image'], expected_local)
                    torch.testing.assert_close(result['global_image'], expected_global)
            view.set_train()
            color = (55, 100, 150)
            tiny = Image.new('RGB', (48, 32), color)
            # Apply one shared augmentation even to JPEG input; branch crops cannot pad.
            with patch.object(view, 'augment', return_value=tiny) as augment:
                result = view[1]
                augment.assert_called_once()
            expected = torch.tensor(color).float().view(3, 1, 1).expand(3, 224, 224) / 255
            torch.testing.assert_close(result['image'], expected)
            torch.testing.assert_close(result['global_image'], expected)

    def test_balanced_validation_metric(self):
        with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', TinyTwin):
            model = OmniDFAAttributor({'num_classes': 2, 'real_class_index': 1})
        metrics = model.compute_metrics({'label': torch.tensor([0, 0, 0, 1])},
                                        {'logits': torch.tensor([[1., 0.]] * 4)})
        self.assertEqual(metrics['acc'], 0.75)
        self.assertEqual(metrics['balanced_accuracy'], 0.5)

    def test_omnifake_evaluation_uses_saved_iab_head_and_both_views(self):
        names = list(model_class_to_label)
        config = dict(configuration(), **PROTOCOL, dataset='iab', class_names=names,
                      num_classes=len(names), real_class_index=names.index('real'))
        checkpoint = dict(PROTOCOL, model_name='omnidfa', dataset='iab', class_names=names,
                          num_classes=len(names), output_class_names=names, training_config=config,
                          train_counts=[1] * len(names), train_digest='train')

        def offline_backbone(pretrained, out_features):
            self.assertFalse(pretrained)
            return TinyTwin(pretrained, out_features)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(omnifake, 'IMAGES_PER_GENERATOR', 1), \
                patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', offline_backbone):
            root = Path(directory)
            for folder in omnifake.OMNIFAKE_TO_IAB:
                path = root / 'data' / 'val' / folder / 'image.png'
                path.parent.mkdir(parents=True)
                Image.new('RGB', (360, 480), (80, 40, 120)).save(path)
            model = OmniDFAAttributor(dict(config, pretrained=False))
            checkpoint['model_state_dict'] = model.state_dict()
            for index, weights in enumerate((checkpoint, legacy_checkpoint(checkpoint))):
                file = root / 'model.pth'
                torch.save(weights, file)
                output = root / f'results_{index}'
                test_baselines_styled.main(['--omnifake', '--method', 'omnidfa',
                    '--checkpoint', str(file), '--root_dir', str(root / 'data'),
                    '--log_dir', str(output), '--batch_size', '4', '--num_workers', '0'])
                result = json.loads((output / 'metrics.json').read_text())
                self.assertEqual(result['attribution_protocol'], weights['attribution_protocol'])
                self.assertEqual(result['class_names'], names)
                self.assertEqual(result['num_images'], 10)
                self.assertEqual(len(result['confusion_matrix']), len(names))

    def test_iabench_views_preserve_training_membership_and_export_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = make_data(root, ['Real', 'R3GAN', 'stylegan3'])
            config = configuration()
            configure_omnidfa(config, 'iabench', 18, 2)
            sources = prepare_iabench(root, manifest, config, 18)
            train_loader, val_loader = get_iabench_dataloaders(*sources, config, 18, 0)
            training = prepare_omnidfa_train_loader(train_loader, config, 18)
            self.assertIs(training.dataset, train_loader.dataset)
            self.assertEqual(training.dataset.mode, 'train')
            self.assertEqual(training.dataset.iabench_source.split, 'train')
            self.assertEqual(config['checkpoint_metadata']['train_counts'], [8, 8, 8])
            sample = val_loader.dataset[0]
            again = val_loader.dataset[0]
            self.assertEqual(sample['image'].shape, (3, 224, 224))
            torch.testing.assert_close(sample['image'], again['image'])
            torch.testing.assert_close(sample['global_image'], again['global_image'])
            self.assertTrue(0 <= sample['image'].min() <= sample['image'].max() <= 1)
            exported = root / 'export'
            export_images(root, exported)
            export_config = configuration()
            configure_omnidfa(export_config, 'iabench', 18, 2)
            _, export_val = prepare_iabench(exported, manifest, export_config, 18)
            export_view = OmniDFADataset('', iabench_source=export_val)
            export_view.set_val()
            self.assertEqual(export_view.samples, val_loader.dataset.samples)
            torch.testing.assert_close(export_view[0]['image'], sample['image'])
            torch.testing.assert_close(export_view[0]['global_image'], sample['global_image'])
            with self.assertRaises(ValueError):
                prepare_omnidfa_train_loader(val_loader, config, 18)

    def run_training(self, root, logs, manifest=None, resume=None, test_after=False):
        argv = ['--config', 'comparison/training/config/model/omnidfa.yaml',
                '--root_dir', str(root), '--n_epoch', '2', '--batch_size', '18',
                '--num_workers', '0', '--log_dir', str(logs)]
        if not test_after:
            argv += ['--do_test']
        if manifest is not None:
            argv += ['--dataset', 'iabench', '--split_manifest', str(manifest)]
        if resume is not None:
            argv += ['--resume_checkpoint', str(resume)]
        with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', TinyTwin):
            train.main(argv)

    def test_iabench_training_resume_and_evaluation_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root = work / 'data'
            manifest, splits = make_data(root, ['Real', 'R3GAN', 'stylegan3'])
            accessed = []
            # Track optimization and scoring at the shared read boundary.
            open_image = OmniDFADataset._open_image

            def tracked(dataset, row, *args, **kwargs):
                accessed.append((dataset.iabench_source.split, dataset.mode, row))
                return open_image(dataset, row, *args, **kwargs)

            save = OmniDFATrainer.save_checkpoint

            def stop_after_last(trainer, *args, **kwargs):
                save(trainer, *args, **kwargs)
                if kwargs.get('filename') == 'ckpt_last.pth':
                    raise RuntimeError('simulated interruption')

            with patch.object(OmniDFADataset, '_open_image', tracked):
                with patch.object(OmniDFATrainer, 'save_checkpoint', stop_after_last):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        self.run_training(root, work / 'logs', manifest)
                checkpoint_path = next((work / 'logs').rglob('ckpt_last.pth'))
                first = torch.load(checkpoint_path, weights_only=False)
                self.assertEqual(first['epoch'], 1)
                self.run_training(root, work / 'ignored', manifest, checkpoint_path)
            self.assertEqual({(split, mode) for split, mode, _ in accessed},
                             {('train', 'train'), ('val', 'val')})
            self.assertTrue(all(row in splits[split] for split, _, row in accessed))
            self.assertEqual(sum(split == 'train' for split, _, _ in accessed), 2 * 18)
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(checkpoint['scheduler_state_dict']['last_epoch'], 2)
            self.assertEqual(next(iter(checkpoint['optimizer_state_dict']['state'].values()))['step'].item(), 2)
            self.assertEqual(len((checkpoint_path.parent / 'steps.csv').read_text().splitlines()), 3)
            self.assertEqual(checkpoint['train_counts'], [8, 8, 8])
            self.assertEqual(checkpoint['attribution_protocol'], CROSS_ENTROPY_PROTOCOL)
            self.assertNotIn('prototypes', checkpoint['model_state_dict'])
            self.assertFalse((work / 'ignored').exists())
            prepare_iabench_test(root, manifest, checkpoint)
            legacy = legacy_checkpoint(checkpoint, upstream_augmentation=True)
            legacy_path = checkpoint_path.parent / 'legacy.pth'
            torch.save(legacy, legacy_path)
            with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', side_effect=AssertionError('backbone loaded')):
                with self.assertRaisesRegex(ValueError, 'prototype checkpoints.*evaluation only'):
                    train.main(['--config', 'comparison/training/config/model/omnidfa.yaml',
                                '--dataset', 'iabench', '--batch_size', '18',
                                '--resume_checkpoint', str(legacy_path), '--log_dir', str(work / 'ignored')])
            damaged = deepcopy(legacy)
            damaged['model_state_dict']['prototype_counts'][0] = 0
            with self.assertRaises(ValueError):
                prepare_iabench_test(root, manifest, damaged)
            damaged = deepcopy(checkpoint)
            damaged['class_names'].reverse()
            with self.assertRaises(ValueError):
                validate_omnidfa_checkpoint(damaged)
            damaged = deepcopy(checkpoint)
            damaged['manifest_digest'] = 'different'
            with self.assertRaisesRegex(ValueError, 'manifest_digest'):
                prepare_iabench_test(root, manifest, damaged)
            accessed.clear()
            argv = ['--dataset', 'iabench', '--split_manifest', str(manifest), '--root_dir', str(root),
                    '--resume_checkpoint', str(checkpoint_path), '--num_workers', '0',
                    '--batch_size', '2', '--level_end', '7', '--log_dir', str(work / 'test')]
            with patch.object(OmniDFADataset, '_open_image', tracked), \
                    patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', TinyTwin):
                test.main(argv)
            self.assertEqual(len(accessed), 7 * len(splits['test']))
            self.assertEqual({(split, mode) for split, mode, _ in accessed}, {('test', 'test')})
            for level in range(7):
                result = (work / 'test' / f'test_results_degraded_{level}.txt').read_text()
                self.assertIn('balanced_accuracy:', result)
                self.assertIn(CROSS_ENTROPY_PROTOCOL, result)
                self.assertIn('method_label: TwinNeXt + CE', result)
                self.assertIn('manifest_digest:', result)
                self.assertIn('aug_policy: omnidfa_paper', result)
            argv[argv.index('--resume_checkpoint') + 1] = str(legacy_path)
            argv[argv.index('--level_end') + 1] = '1'
            argv[argv.index('--log_dir') + 1] = str(work / 'legacy_test')
            with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', TinyTwin):
                test.main(argv)
            result = (work / 'legacy_test' / 'test_results_degraded_0.txt').read_text()
            self.assertIn(PROTOTYPE_PROTOCOL, result)
            self.assertIn('prototype_source: clean_train', result)
            self.assertIn('aug_policy: omnidfa_upstream', result)

    def test_iab_training_and_standalone_evaluation_with_real_last(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(model_class_to_label, {'R3GAN': 0, 'real': 1}, clear=True):
            work = Path(directory)
            for name in model_class_to_label:
                folder = work / 'images' / name / 'AnimalFace' / 'cat'
                folder.mkdir(parents=True)
                for i in range(10):
                    Image.new('RGB', (260, 270), (i * 10, 40, 80)).save(folder / f'img_p{i}_i0.png')
            self.run_training(work / 'images', work / 'logs', test_after=True)
            path = next((work / 'logs').rglob('ckpt_best.pth'))
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint['class_names'], ['R3GAN', 'real'])
            self.assertEqual(checkpoint['training_config']['real_class_index'], 1)
            self.assertEqual(checkpoint['train_counts'], [8, 8])
            for level in range(7):
                result = (path.parent / f'test_results_degraded_{level}.txt').read_text()
                self.assertIn('method_label: TwinNeXt + CE', result)
                self.assertIn(CROSS_ENTROPY_PROTOCOL, result)
            argv = ['--config', 'comparison/training/config/model/omnidfa.yaml',
                    '--resume_checkpoint', str(path), '--root_dir', str(work / 'images'),
                    '--num_workers', '0', '--batch_size', '3', '--log_dir', str(work / 'test')]
            with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', TinyTwin):
                test.main(argv)
            result = next((work / 'test').rglob('test_results_degraded_0.txt')).read_text()
            self.assertIn(CROSS_ENTROPY_PROTOCOL, result)
            bad = deepcopy(checkpoint)
            del bad['model_state_dict']['backbone.mlp.fc2.weight']
            with self.assertRaisesRegex(ValueError, 'classification head'):
                validate_omnidfa_checkpoint(bad)
            legacy_path = path.parent / 'legacy.pth'
            torch.save(legacy_checkpoint(checkpoint), legacy_path)
            argv[argv.index('--resume_checkpoint') + 1] = str(legacy_path)
            argv[argv.index('--log_dir') + 1] = str(work / 'legacy_test')
            with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', TinyTwin):
                test.main(argv)
            result = next((work / 'legacy_test').rglob('test_results_degraded_0.txt')).read_text()
            self.assertIn(PROTOTYPE_PROTOCOL, result)
            argv[argv.index('--resume_checkpoint') + 1] = str(path)
            image = next((work / 'images').rglob('*.png'))
            image.unlink()
            with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', side_effect=AssertionError('backbone loaded')):
                with self.assertRaisesRegex(ValueError, 'membership'):
                    test.main(argv)


if __name__ == '__main__':
    unittest.main()
