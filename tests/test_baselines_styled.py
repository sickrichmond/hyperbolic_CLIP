"""Styled baseline checks: python -m tests.test_baselines_styled.

Metadata, inventory and launcher checks use stdlib only. The optional inference
check uses native adapters/metrics with replacement backbones; no downloads.
"""
import contextlib
from copy import deepcopy
import csv
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from comparison.training import test_baselines_styled as evaluator
from comparison.training.scripts.styled_checkpoints import discover_checkpoints, main as make_map


class StyledBaselineTests(unittest.TestCase):
    def test_discovery_freezes_newest_best_and_excludes_dna_pretraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in evaluator.METHODS:
                for run, mtime in (('old_run', 100), ('new_run', 200)):
                    path = root / method / run / 'ckpt_best.pth'
                    path.parent.mkdir(parents=True)
                    path.touch()
                    os.utime(path, (mtime, mtime))
                (path.parent / 'ckpt_last.pth').touch()
            pretrain = root / 'dna/_pretrain_newest/ckpt_best.pth'
            pretrain.parent.mkdir()
            pretrain.touch()
            mapping = discover_checkpoints(root)
            self.assertEqual(set(mapping), set(evaluator.METHODS))
            self.assertTrue(all(Path(path).parent.name == 'new_run' for path in mapping.values()))
            output = root / 'map.json'
            with contextlib.redirect_stdout(io.StringIO()):
                make_map(['--log_root', str(root), '--output', str(output)])
            self.assertEqual(json.loads(output.read_text()), mapping)
            with self.assertRaises(FileExistsError):
                make_map(['--log_root', str(root), '--output', str(output)])
            for path in (root / 'dna').glob('*run/ckpt_best.pth'):
                path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'stage-two'):
                discover_checkpoints(root)

    def test_checkpoint_map_and_cli(self):
        with tempfile.TemporaryDirectory(prefix='styled checkpoints ') as directory:
            root = Path(directory)
            checkpoint = root / 'resnet checkpoint.pth'
            checkpoint.touch()
            path = root / 'checkpoints.json'
            path.write_text(json.dumps({'resnet50': checkpoint.name}))
            mapping = evaluator.load_checkpoint_map(path)
            self.assertEqual(mapping, {'resnet50': str(checkpoint)})
            args = evaluator.parse_args(['--method', 'resnet50', '--checkpoints', str(path),
                                         '--root_dir', 'styles', '--log_dir', 'results'])
            self.assertEqual(args.checkpoint, str(checkpoint))
            for bad in ({'bad': str(checkpoint)}, {'resnet50': ''}, {}, [],
                        {'resnet50': None}):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    evaluator.load_checkpoint_map(path)
            path.write_text(json.dumps({'resnet50': 'missing.pth'}))
            with self.assertRaises(FileNotFoundError):
                evaluator.load_checkpoint_map(path)

    def test_all_configurations_keep_head_semantics(self):
        names = ['FLUX', 'SDXL', 'real']
        for method in evaluator.METHODS:
            with self.subTest(method=method):
                fallback = {'model_name': method, 'load_param': True,
                            'backbone_config': {'num_classes': 2},
                            'image_size': 256, 'train_stage': 2}
                original = deepcopy(fallback)
                checkpoint = {'model_state_dict': {'weights': 1}}
                config = evaluator.prepare_config(checkpoint, method, fallback, names)
                self.assertEqual(fallback, original)
                self.assertEqual(config['class_names'], names)
                self.assertEqual(config['num_classes'], 3)
                self.assertEqual(config['specific_task_number'], 3)
                self.assertEqual(config['real_class_index'], 2)
                self.assertEqual(config['backbone_config']['num_classes'], 2)
                self.assertFalse(config['load_param'])
                self.assertFalse(config['ddp'])
                self.assertEqual(config['image_size'], 256)
                checkpoint['training_config'] = dict(fallback, image_size=128)
                self.assertEqual(evaluator.prepare_config(checkpoint, method, fallback, names)[
                    'image_size'], 128)

    def test_bad_checkpoint_metadata_fails(self):
        names = ['FLUX', 'SDXL', 'real']
        fallback = {'model_name': 'dna', 'train_stage': 2}
        base = {'model_state_dict': {'weights': 1}}
        for bad in ({'dataset': 'iabench'}, {'model_name': 'pose'},
                    {'class_names': list(reversed(names))}, {'num_classes': 170},
                    {'output_class_names': names[:2]}, {'train_stage': 1},
                    {'output_label_space': 'transformations'}, {'model_state_dict': {}},
                    {'training_config': None}, {'training_config': {'model_name': 'pose'}},
                    {'model_state_dict': {'backbone.classification_head.1.weight':
                                          SimpleNamespace(shape=(170, 512))}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluator.prepare_config(dict(base, **bad), 'dna', fallback, names)
        with self.assertRaisesRegex(ValueError, 'two-class detection'):
            evaluator.prepare_config(base, 'ucf', {'model_name': 'ucf',
                                      'backbone_config': {'num_classes': 22}}, names)
        with self.assertRaisesRegex(ValueError, 'Attribution head'):
            evaluator.prepare_config(base, 'ucf', {'model_name': 'ucf',
                'specific_task_number': 4, 'backbone_config': {'num_classes': 2}}, names)

    def test_inventory_has_no_filename_cap_or_split_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / 'SDXL/COCO/not_iab_named.JPG',
                     root / 'SDXL/AnimalFace/cat/cat_p9999_i99.png',
                     root / 'FLUX/COCO/COCO-new_p0_i0.PNG']
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / 'SDXL/manifest.jsonl').touch()
            paths, labels = {'COCO': 'COCO', 'cat': 'AnimalFace/cat'}, {'COCO': 3, 'cat': 0}
            samples = evaluator.collect_samples(root, ['real', 'SDXL', 'FLUX'], paths, labels)
            self.assertEqual({p for p, *_ in samples}, {str(p) for p in files})
            self.assertEqual([row[1] for row in samples], [1, 1, 2])
            self.assertEqual([row[2] for row in samples], [3, 0, 3])
            extra = root / 'FLUX/unexpected/deep.png'
            extra.parent.mkdir()
            extra.touch()
            with self.assertRaisesRegex(ValueError, 'outside the supported'):
                evaluator.collect_samples(root, ['real', 'SDXL', 'FLUX'], paths, labels)
            extra.unlink()
            (root / 'real').mkdir()
            with self.assertRaisesRegex(ValueError, 'No images'):
                evaluator.collect_samples(root, ['real', 'SDXL', 'FLUX'], paths, labels)

    def test_generator_filter_preserves_full_labels_and_ignores_excluded_images(self):
        names = ['real', 'FLUX', 'PLAYGROUND_2_5', 'SD3', 'SDXL', 'hunyuan']
        requested = ['PLAYGROUND_2_5', 'hunyuan', 'SDXL']
        with tempfile.TemporaryDirectory() as directory:
            style = Path(directory) / 'watercolor'
            for name in names[1:]:
                path = style / name / 'COCO/image.png'
                path.parent.mkdir(parents=True)
                path.touch()
            styles = evaluator.discover_styles(directory, names)
            selected = evaluator.select_style_generators(styles, requested, names)
            self.assertEqual(selected, [(style, ['PLAYGROUND_2_5', 'SDXL', 'hunyuan'])])
            self.assertIs(evaluator.select_style_generators(styles, None, names), styles)
            # Excluded folders must not be read or validated as input samples.
            bad = style / 'FLUX/unsupported/image.png'
            bad.parent.mkdir()
            bad.touch()
            samples = evaluator.collect_samples(style, names, {'COCO': 'COCO'}, {'COCO': 3},
                                                selected[0][1])
            self.assertEqual([row[1] for row in samples], [2, 4, 5])
            self.assertEqual({Path(row[0]).parent.parent.name for row in samples}, set(requested))
            with self.assertRaisesRegex(ValueError, 'absent from the IAB class map'):
                evaluator.select_style_generators(styles, ['playground'], names)
            with self.assertRaisesRegex(ValueError, 'folders missing'):
                evaluator.select_style_generators(styles, ['real'], names)

    def test_array_launcher_selects_every_method_and_preserves_gpu(self):
        repo = Path(__file__).resolve().parents[1]
        scripts = [(repo / 'slurm/slurm_eval_iab_styled_baselines.sh', 'baselines_iab_styled', []),
                   (repo / 'slurm/slurm_eval_iab_styled_baselines_3models.sh',
                    'baselines_iab_styled_3models', ['PLAYGROUND_2_5', 'hunyuan', 'SDXL'])]
        stub = r'''
module() { :; }
python() {
    "$TEST_PYTHON" -c 'import json, os, sys; from pathlib import Path; Path(os.environ["CAPTURE"]).write_text(json.dumps({"argv": sys.argv[1:], "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}))' "$@"
}
export -f module python
exec bash "$1" "$2"
'''
        with tempfile.TemporaryDirectory(prefix='styled baseline ') as directory:
            root = Path(directory)
            (root / 'hyp_fine_tuning/bin').mkdir(parents=True)
            (root / 'hyp_fine_tuning/bin/activate').touch()
            (root / 'hyp_fine_tuning/hyperbolic_CLIP_riccardo').mkdir()
            (root / 'datasets/iab_styled_eval').mkdir(parents=True)
            path = root / 'checkpoints.json'
            path.write_text('{}')
            capture = root / 'command.json'
            env = dict(os.environ, WORK=str(root), FAST=str(root), TEST_PYTHON=sys.executable,
                       CAPTURE=str(capture), CUDA_VISIBLE_DEVICES='4', SLURM_ARRAY_JOB_ID='123')
            for key in ('REPO', 'DATA', 'LOGDIR', 'BATCH_SIZE', 'NUM_WORKERS'):
                env.pop(key, None)
            for script, output_prefix, selected in scripts:
                for index, method in enumerate(evaluator.METHODS):
                    with self.subTest(script=script.name, method=method):
                        subprocess.run(['bash', '-c', stub, '_', str(script), str(path)],
                            env=dict(env, SLURM_ARRAY_TASK_ID=str(index)), check=True,
                            capture_output=True, text=True)
                        command = json.loads(capture.read_text())
                        argv = command['argv']
                        self.assertEqual(argv[:2], ['-m', 'comparison.training.test_baselines_styled'])
                        if selected:
                            start = argv.index('--generators')
                            self.assertEqual(argv[start + 1:], selected)
                            argv = argv[:start]
                        else:
                            self.assertNotIn('--generators', argv)
                        flags = dict(zip(argv[2::2], argv[3::2]))
                        self.assertEqual(flags['--method'], method)
                        self.assertEqual(flags['--checkpoints'], str(path))
                        self.assertEqual(flags['--root_dir'], str(root / 'datasets/iab_styled_eval'))
                        self.assertEqual(flags['--log_dir'],
                            str(root / f'outputs/{output_prefix}_123/{method}'))
                        self.assertEqual(flags['--batch_size'], '8')
                        self.assertEqual(command['gpu'], '4')
            capture.unlink()
            result = subprocess.run(['bash', '-c', stub, '_', str(script), str(path)],
                env=dict(env, SLURM_ARRAY_TASK_ID='11'), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture.exists())

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
        ('torch', 'torchvision', 'PIL', 'transformers', 'datasets', 'yaml', 'sklearn')),
        'CPU inference requires the baseline training dependencies')
    def test_native_adapters_and_metrics_all_baselines(self):
        from PIL import Image
        import torch
        from comparison.dataset.ImageAttributionDataset.dataset import model_class_to_label
        from comparison.training.attributors import ATTRIBUTOR
        from tests.test_iabench_baseline_evaluation import EvaluationFixture
        from tests.test_iabench_baselines import config_for

        torch.set_num_threads(1)
        names = list(model_class_to_label)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for style in ('noir', 'watercolor'):
                for gen, count in (('FLUX', 1), ('SDXL', 2)):
                    folder = root / 'data' / style / gen / 'COCO'
                    folder.mkdir(parents=True)
                    for i in range(count):
                        Image.new('RGB', (260, 270), (80, 40, 120)).save(folder / f'{i}.png')
            for method in evaluator.METHODS:
                with self.subTest(method=method):
                    config = config_for(method)
                    if method == 'dna':
                        config.update(resize_size=[64, 64], multi_size=[[16, 16]])
                    config = evaluator.prepare_config({'model_state_dict': {'dummy': 1}},
                                                       method, config, names)
                    fixture = EvaluationFixture(config)
                    with torch.no_grad():
                        fixture.fc.weight.zero_()
                        fixture.fc.bias.zero_()
                        fixture.fc.bias[names.index('real')] = 20.
                        if method == 'repmix':
                            fixture.detection.weight.zero_()
                            fixture.detection.bias.zero_()
                    checkpoint = root / f'{method}.pth'
                    torch.save({'model_state_dict': fixture.state_dict(),
                                'training_config': config}, checkpoint)
                    batch_sizes = []
                    original_forward = EvaluationFixture.forward
                    def forward(model, batch, inference=False):
                        batch_sizes.append(len(batch['label']))
                        return original_forward(model, batch, inference)
                    output = root / method
                    with patch.dict(ATTRIBUTOR.data, {method: EvaluationFixture}), \
                            patch.object(EvaluationFixture, 'forward', forward), \
                            patch('torch.cuda.is_available', return_value=False), \
                            patch('clip.load', side_effect=AssertionError('second CLIP load')), \
                            patch('comparison.training.logger.create_logger',
                                  return_value=logging.getLogger('styled-fixture')), \
                            contextlib.redirect_stdout(io.StringIO()):
                        evaluator.main(['--method', method, '--checkpoint', str(checkpoint),
                            '--root_dir', str(root / 'data'), '--log_dir', str(output),
                            '--batch_size', '2', '--num_workers', '0'])
                        evaluator.main(['--method', method, '--checkpoint', str(checkpoint),
                            '--root_dir', str(root / 'data'), '--log_dir', str(output / 'filtered'),
                            '--batch_size', '3', '--num_workers', '0', '--generators', 'SDXL'])
                    self.assertEqual(batch_sizes, [2, 1, 2, 1, 2, 2])
                    for style in ('noir', 'watercolor'):
                        result = json.loads((output / style / 'metrics.json').read_text())
                        self.assertEqual(result['class_names'], names)
                        self.assertEqual(result['num_images'], 3)
                        self.assertEqual(sum(result['support_per_class']), 3)
                        self.assertEqual(result['support_per_class'][names.index('SDXL')], 2)
                        self.assertEqual(result['balanced_accuracy'], 0.)
                        self.assertEqual(sum(row[names.index('real')] for row in
                                             result['confusion_matrix']), 3)
                        if method == 'patch':
                            self.assertIn('acc_D_voted', result)
                        filtered = json.loads((output / 'filtered' / style / 'metrics.json').read_text())
                        self.assertEqual(filtered['num_images'], 2)
                        self.assertEqual(filtered['class_names'], names)
                        self.assertEqual(filtered['evaluated_generators'], ['SDXL'])
                        self.assertEqual(filtered['support_per_class'][names.index('SDXL')], 2)
                        self.assertEqual(sum(filtered['support_per_class']), 2)
                        self.assertEqual(sum(row[names.index('real')] for row in
                                             filtered['confusion_matrix']), 2)
                    with (output / 'summary.csv').open() as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), 2)


if __name__ == '__main__':
    unittest.main()
