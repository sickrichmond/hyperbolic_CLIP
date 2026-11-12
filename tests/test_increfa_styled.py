"""Run with python -m tests.test_increfa_styled (Pillow for image checks)."""
from copy import deepcopy
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from comparison.training import test_increfa_styled as evaluator


def checkpoint():
    return {
        'config': {'data': {'manifest_path': 'manifest.json', 'total_classes': 3}},
        'results': {'class_order': [['real', 'SDXL', 'FLUX']]},
        'task_id': 0, 'num_learned_classes': 3, 'model': {'head': 1},
        'class_to_family': {0: 0, 1: 1, 2: 1},
    }


class IncreFAStyledTests(unittest.TestCase):
    def test_saved_order_and_completed_iab_task(self):
        self.assertEqual(evaluator.checkpoint_classes(checkpoint()), ['real', 'SDXL', 'FLUX'])
        for edits in ({'task_id': 1}, {'num_learned_classes': 2},
                      {'results': {'class_order': [['real'], ['SDXL', 'FLUX']]}},
                      {'results': {'class_order': [['SDXL', 'real', 'FLUX']]}},
                      {'results': {'class_order': [['real', 'SDXL', 'SDXL']]}},
                      {'config': {'data': {'split_manifest_path': 'iab.json', 'total_classes': 3}}}):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                evaluator.checkpoint_classes(dict(checkpoint(), **edits))

    def test_native_restore_validates_mapping_and_trainable_weights(self):
        class Model:
            class_to_family = {0: 0, 1: 1, 2: 1}
            missing = ['backbone.frozen_weight']
            unexpected = []

            def __init__(self):
                self.orth_space = SimpleNamespace(update_indicator=lambda n: setattr(self, 'active', n))

            def load_state_dict(self, state, strict):
                self.state = state
                return self.missing, self.unexpected

            def eval(self):
                self.evaluating = True

        class Trainer:
            order = [[0, 1, 2]]

            def setseed(self):
                pass

            def load_dataset(self):
                return SimpleNamespace(order=self.order, transform='native-transform',
                    idx_to_label_str={0: 'real', 1: 'SDXL', 2: 'FLUX'})

            def configure_predefined_families(self):
                self.model.class_to_family = dict(Model.class_to_family)

        train = ModuleType('train')
        train.IncreTrainer = Trainer
        models = ModuleType('models')
        models.load_model = lambda config: Model()
        with patch.dict(sys.modules, train=train, models=models):
            saved = checkpoint()
            original = deepcopy(saved)
            model, transform, names = evaluator.restore_model(saved)
            self.assertEqual(saved, original)
            self.assertEqual(model.active, 3)
            self.assertTrue(model.evaluating)
            self.assertEqual(model.state, saved['model'])
            self.assertEqual(transform, 'native-transform')
            self.assertEqual(names, ['real', 'SDXL', 'FLUX'])
            with patch.object(Trainer, 'order', [[0, 2, 1]]), self.assertRaisesRegex(ValueError, 'class order'):
                evaluator.restore_model(saved)
            saved['class_to_family'][1] = 0
            with self.assertRaisesRegex(ValueError, 'family mapping'):
                evaluator.restore_model(saved)
            for missing, unexpected in ((['head.weight'], []), ([], ['extra'])):
                with patch.object(Model, 'missing', missing), patch.object(Model, 'unexpected', unexpected):
                    with self.assertRaisesRegex(ValueError, 'model state'):
                        evaluator.restore_model(checkpoint())

    def test_inventory_and_full_class_metrics(self):
        names = ['real', 'SDXL', 'FLUX']
        with tempfile.TemporaryDirectory() as directory:
            style = Path(directory) / 'style'
            paths = [style / 'SDXL/COCO/arbitrary.JPG', style / 'FLUX/AnimalFace/cat/p999_i99.PNG']
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(evaluator.collect_samples(style, names), [(paths[0], 1), (paths[1], 2)])
            extra = style / 'FLUX/unsupported.png'
            extra.touch()
            with self.assertRaisesRegex(ValueError, 'semantic layout'):
                evaluator.collect_samples(style, names)
            extra.unlink()
            (style / 'real').mkdir()
            with self.assertRaisesRegex(ValueError, 'No images'):
                evaluator.collect_samples(style, names)
        report = evaluator.summarize([[0, 0, 0], [1, 1, 0], [0, 1, 3]], names)
        self.assertEqual(report['num_images'], 6)
        self.assertAlmostEqual(report['accuracy'], 4 / 6)
        self.assertAlmostEqual(report['balanced_accuracy'], (1 / 2 + 3 / 4) / 2)
        self.assertIsNone(report['per_class_recall']['real'])

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow is unavailable')
    def test_jpeg99_before_native_crop_transform_and_read_retries(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            original = Image.new('RGBA', (160, 90))
            original.putdata([(i % 256, (i * 11) % 256, (i * 17) % 256, 80)
                              for i in range(160 * 90)])
            original.save(path)
            encoded = BytesIO()
            original.convert('RGB').save(encoded, format='JPEG', quality=99)
            encoded.seek(0)
            with Image.open(encoded) as compressed:
                expected = compressed.convert('RGB').crop((0, 0, 60, 40))
            dataset = evaluator.StyledDataset([(path, 0)], lambda image: image, ['grok3'])
            actual, label = dataset[0]
            self.assertEqual(label, 0)
            self.assertEqual(actual.size, expected.size)
            self.assertEqual(actual.tobytes(), expected.tobytes())
            with Image.open(path) as source:
                self.assertEqual(source.mode, 'RGBA')
            open_image = Image.open
            calls = []

            def transient(source, *args, **kwargs):
                if source == path:
                    calls.append(source)
                    if len(calls) < 3:
                        raise PermissionError('temporary read failure')
                return open_image(source, *args, **kwargs)

            with patch.object(Image, 'open', transient), patch.object(evaluator.time, 'sleep'):
                self.assertEqual(dataset[0][0].tobytes(), expected.tobytes())
            self.assertEqual(len(calls), 3)
            with patch.object(Image, 'open', side_effect=OSError('permanent')) as failed:
                with patch.object(evaluator.time, 'sleep'), self.assertRaises(OSError):
                    dataset[0]
                self.assertEqual(failed.call_count, 8)

    def test_launcher_uses_usual_environment_and_separate_results(self):
        script = Path(__file__).resolve().parents[1] / 'slurm/slurm_eval_iab_styled_increfa.sh'
        source = script.read_text()
        self.assertIn('module load python/3.11.7', source)
        self.assertIn('module load cuda/12.6', source)
        self.assertNotIn('anaconda3/2023.09-0', source)
        stub = '''
module() { :; }
srun() { "$TEST_PYTHON" -c 'import json, os, sys; from pathlib import Path; Path(os.environ["CAPTURE"]).write_text(json.dumps({"argv": sys.argv[1:], "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "activated": os.environ.get("INCREFA_TEST_ACTIVE")}))' "$@"; }
export -f module srun
exec bash "$@"
'''
        with tempfile.TemporaryDirectory(prefix='increfa styled ') as directory:
            root = Path(directory)
            native = root / 'IncreFA'
            python = root / 'hyp_fine_tuning/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            python.chmod(0o755)
            (python.parent / 'activate').write_text('export INCREFA_TEST_ACTIVE=1\n')
            broken_python = native / '.venv/bin/python'
            broken_python.parent.mkdir(parents=True)
            broken_python.symlink_to(root / 'missing-anaconda/bin/python')
            (native / 'train.py').touch()
            data = root / 'datasets/iab_styled_eval'
            data.mkdir(parents=True)
            checkpoint_path = root / 'latest.pt'
            checkpoint_path.touch()
            capture = root / 'captured.json'
            env = dict(os.environ, WORK=str(root), FAST=str(root), CKPT=str(checkpoint_path),
                       CAPTURE=str(capture), TEST_PYTHON=sys.executable,
                       CUDA_VISIBLE_DEVICES='3', SLURM_JOB_ID='456')
            for key in ('INCREFA_REPO', 'REPO', 'DATA', 'LOGDIR', 'BATCH_SIZE', 'NUM_WORKERS'):
                env.pop(key, None)
            subprocess.run(['bash', '-c', stub, '_', str(script)], env=env,
                           check=True, capture_output=True, text=True)
            result = json.loads(capture.read_text())
            self.assertEqual(result['gpu'], '3')
            self.assertEqual(result['activated'], '1')
            self.assertEqual(result['argv'][:4], [str(python), '-u', '-m',
                                                'comparison.training.test_increfa_styled'])
            flags = dict(zip(result['argv'][4::2], result['argv'][5::2]))
            self.assertEqual(flags['--increfa_dir'], str(native))
            self.assertEqual(flags['--root_dir'], str(data))
            self.assertEqual(flags['--log_dir'], str(root / 'outputs/increfa_iab_styled_jpeg99_456'))
            explicit_output = root / 'explicit output'
            subprocess.run(['bash', '-c', stub, '_', str(script), str(checkpoint_path),
                            str(data), str(explicit_output)],
                           env=dict(env, CKPT='stale.pt', DATA='stale-data', LOGDIR='stale-output'),
                           check=True, capture_output=True, text=True)
            command = json.loads(capture.read_text())['argv']
            flags = dict(zip(command[4::2], command[5::2]))
            self.assertEqual(flags['--checkpoint'], str(checkpoint_path))
            self.assertEqual(flags['--root_dir'], str(data))
            self.assertEqual(flags['--log_dir'], str(explicit_output))


if __name__ == '__main__':
    unittest.main()
