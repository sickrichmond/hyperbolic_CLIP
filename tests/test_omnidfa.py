"""Closed-set OmniDFA checks, using real adapters and a tiny downloadable-free backbone.

Run: python -m tests.test_omnidfa
"""
from copy import deepcopy
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

from comparison.training import train, test
from comparison.training.attributors.attributor_omnidfa import OmniDFAAttributor
from comparison.training.backbones.omnidfa import supervised_contrastive_loss, SphereCenterLoss, TwinNeXt
from comparison.training.omnidfa import (
    OmniDFABatchSampler, configure_omnidfa, prepare_omnidfa_loaders,
    validate_omnidfa_checkpoint,
)
from comparison.training.iabench import prepare_iabench, get_iabench_dataloaders, prepare_iabench_test
from comparison.training.trainer.omnidfa_trainer import OmniDFATrainer
from comparison.dataset.ImageAttributionDataset.dataset import model_class_to_label
from comparison.dataset.ImageAttributionDataset.dataset_omnidfa import OmniDFADataset
from data.degradations import apply_degradation, omnidfa_augment
from data.export_iabench import export_images
from tests.test_iabench_baselines import make_data


class TinyTwin(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.fc = nn.Linear(6, 128)

    def forward(self, local, global_image):
        return self.fc(torch.cat((local.mean((2, 3)), global_image.mean((2, 3))), dim=1))


def configuration():
    return yaml.safe_load(Path('comparison/training/config/model/omnidfa.yaml').read_text())


class OmniDFATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_contrastive_matches_upstream_and_has_gradients(self):
        torch.manual_seed(7)
        features = torch.randn(6, 128, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1, 2, 3])
        normalized = F.normalize(features, dim=1)
        similarities = normalized @ normalized.T / 0.07
        eye = torch.eye(6)
        same = labels[:, None].eq(labels[None, :]).float() - eye
        log_probability = similarities - ((similarities.exp() * (1 - eye)).sum(1, keepdim=True) + 1e-7).log()
        expected = (-(same * log_probability).sum(1) / (same.sum(1) + 1e-7)).mean()
        loss = supervised_contrastive_loss(features, labels)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum(), 0)
        singleton_loss = supervised_contrastive_loss(features.detach(), torch.arange(6))
        self.assertEqual(singleton_loss.item(), 0)

    def test_real_center_and_prototype_math(self):
        center = SphereCenterLoss()
        real = torch.randn(4, 128, requires_grad=True)
        expected = (1 - F.normalize(real, dim=1) @ F.normalize(center.center_real, dim=0)).mean()
        loss = center(real)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(center.center_real.grad)
        self.assertLess(center.cosine_threshold, 1)
        with self.assertRaises(ValueError):
            center(real[:0])
        with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', TinyTwin):
            model = OmniDFAAttributor({'num_classes': 2, 'real_class_index': 1})
        embeddings = F.normalize(torch.randn(5, 128), dim=1)
        labels = torch.tensor([1, 0, 1, 0, 1])
        sums = torch.zeros(2, 128).index_add_(0, labels, embeddings)
        model.set_prototypes(sums, torch.tensor([2, 3]))
        torch.testing.assert_close(model.prototypes[1], F.normalize(embeddings[labels == 1].mean(0), dim=0))
        torch.testing.assert_close(model.classifier(embeddings), embeddings @ model.prototypes.T)
        with self.assertRaises(ValueError):
            model.set_prototypes(sums, torch.tensor([0, 3]))

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

    def test_native_architecture_without_downloads(self):
        model = TwinNeXt(pretrained=False).eval()
        self.assertIsNot(next(model.backbone1.parameters()), next(model.backbone2.parameters()))
        with torch.no_grad():
            self.assertEqual(model(torch.zeros(1, 3, 32, 32), torch.ones(1, 3, 32, 32)).shape, (1, 128))

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

    def test_iabench_views_preserve_training_membership_and_export_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = make_data(root, ['Real', 'R3GAN', 'stylegan3'])
            config = configuration()
            configure_omnidfa(config, 'iabench', 18, 2)
            sources = prepare_iabench(root, manifest, config, 18)
            train_loader, val_loader = get_iabench_dataloaders(*sources, config, 18, 0)
            training, support = prepare_omnidfa_loaders(train_loader, config, 18)
            self.assertEqual(support.dataset.samples, train_loader.dataset.samples)
            self.assertEqual(support.dataset.mode, 'val')
            self.assertEqual(training.dataset.mode, 'train')
            self.assertEqual(support.dataset.iabench_source.split, 'train')
            self.assertEqual(config['checkpoint_metadata']['support_counts'], [8, 8, 8])
            sample = support.dataset[0]
            again = support.dataset[0]
            self.assertEqual(sample['image'].shape, (3, 224, 224))
            torch.testing.assert_close(sample['image'], again['image'])
            torch.testing.assert_close(sample['global_image'], again['global_image'])
            self.assertTrue(0 <= sample['image'].min() <= sample['image'].max() <= 1)
            exported = root / 'export'
            export_images(root, exported)
            export_config = configuration()
            configure_omnidfa(export_config, 'iabench', 18, 2)
            export_train, _ = prepare_iabench(exported, manifest, export_config, 18)
            export_view = OmniDFADataset('', iabench_source=export_train)
            export_view.set_val()
            self.assertEqual(export_view.samples, support.dataset.samples)
            torch.testing.assert_close(export_view[0]['image'], sample['image'])
            torch.testing.assert_close(export_view[0]['global_image'], sample['global_image'])
            with self.assertRaises(ValueError):
                prepare_omnidfa_loaders(val_loader, config, 18)

    def run_training(self, root, logs, manifest=None, resume=None):
        argv = ['--config', 'comparison/training/config/model/omnidfa.yaml',
                '--root_dir', str(root), '--n_epoch', '2', '--batch_size', '18',
                '--num_workers', '0', '--log_dir', str(logs), '--do_test']
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
            # Track optimization, prototype collection and scoring at the shared read boundary.
            open_image = OmniDFADataset._open_image

            def tracked(dataset, row, *args, **kwargs):
                accessed.append((dataset.iabench_source.split, row))
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
            self.assertEqual({split for split, _ in accessed}, {'train', 'val'})
            self.assertTrue(all(row in splits[split] for split, row in accessed))
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(checkpoint['scheduler_state_dict']['last_epoch'], 2)
            self.assertEqual(next(iter(checkpoint['optimizer_state_dict']['state'].values()))['step'].item(), 2)
            self.assertEqual(len((checkpoint_path.parent / 'steps.csv').read_text().splitlines()), 3)
            self.assertEqual(checkpoint['model_state_dict']['prototype_counts'].tolist(), [8, 8, 8])
            self.assertFalse((work / 'ignored').exists())
            prepare_iabench_test(root, manifest, checkpoint)
            legacy = deepcopy(checkpoint)
            legacy.update(preprocessing='upstream_train_deterministic_eval_v1',
                          augmentation_policy='omnidfa_upstream')
            legacy['training_config'].update(preprocessing=legacy['preprocessing'],
                                              augmentation_policy=legacy['augmentation_policy'])
            prepare_iabench_test(root, manifest, legacy)
            legacy_path = checkpoint_path.parent / 'legacy.pth'
            torch.save(legacy, legacy_path)
            with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', side_effect=AssertionError('backbone loaded')):
                with self.assertRaisesRegex(ValueError, 'training_config|preprocessing|augmentation_policy'):
                    self.run_training(root, work / 'ignored', manifest, legacy_path)
            damaged = deepcopy(checkpoint)
            damaged['model_state_dict']['prototype_counts'][0] = 0
            with self.assertRaises(ValueError):
                prepare_iabench_test(root, manifest, damaged)
            damaged = deepcopy(checkpoint)
            damaged['class_names'].reverse()
            with self.assertRaises(ValueError):
                validate_omnidfa_checkpoint(damaged)
            accessed.clear()
            argv = ['--dataset', 'iabench', '--split_manifest', str(manifest), '--root_dir', str(root),
                    '--resume_checkpoint', str(checkpoint_path), '--num_workers', '0',
                    '--batch_size', '2', '--level_end', '7', '--log_dir', str(work / 'test')]
            with patch.object(OmniDFADataset, '_open_image', tracked), \
                    patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', TinyTwin):
                test.main(argv)
            self.assertEqual(len(accessed), 7 * len(splits['test']))
            self.assertEqual({split for split, _ in accessed}, {'test'})
            for level in range(7):
                result = (work / 'test' / f'test_results_degraded_{level}.txt').read_text()
                self.assertIn('balanced_accuracy:', result)
                self.assertIn('closed_set_train_prototypes_v1', result)
                self.assertIn('manifest_digest:', result)
                self.assertIn('aug_policy: omnidfa_paper', result)

    def test_iab_training_and_standalone_evaluation_with_real_last(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(model_class_to_label, {'R3GAN': 0, 'real': 1}, clear=True):
            work = Path(directory)
            for name in model_class_to_label:
                folder = work / 'images' / name / 'AnimalFace' / 'cat'
                folder.mkdir(parents=True)
                for i in range(10):
                    Image.new('RGB', (260, 270), (i * 10, 40, 80)).save(folder / f'img_p{i}_i0.png')
            self.run_training(work / 'images', work / 'logs')
            path = next((work / 'logs').rglob('ckpt_best.pth'))
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint['class_names'], ['R3GAN', 'real'])
            self.assertEqual(checkpoint['training_config']['real_class_index'], 1)
            self.assertEqual(checkpoint['support_counts'], [8, 8])
            argv = ['--config', 'comparison/training/config/model/omnidfa.yaml',
                    '--resume_checkpoint', str(path), '--root_dir', str(work / 'images'),
                    '--num_workers', '0', '--batch_size', '3', '--log_dir', str(work / 'test')]
            with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', TinyTwin):
                test.main(argv)
            bad = deepcopy(checkpoint)
            del bad['model_state_dict']['prototypes']
            with self.assertRaises(ValueError):
                validate_omnidfa_checkpoint(bad)
            image = next((work / 'images').rglob('*.png'))
            image.unlink()
            with patch('comparison.training.attributors.attributor_omnidfa.TwinNeXt', side_effect=AssertionError('backbone loaded')):
                with self.assertRaisesRegex(ValueError, 'membership'):
                    test.main(argv)


if __name__ == '__main__':
    unittest.main()
