"""Styled IAB inventory and launcher checks; optional CPU inference, no downloads.

Run: python -m tests.test_hypclip_styled
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from comparison.training import test_hypclip_styled as evaluator


class StyledEvaluationTests(unittest.TestCase):
    def test_inventory_preserves_checkpoint_order_and_all_styles(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for style in ('van_gogh_impasto', 'botanical_watercolor'):
                for generator in ('SDXL', 'FLUX'):
                    (root / style / generator / 'COCO').mkdir(parents=True)
                    (root / style / generator / 'manifest.jsonl').touch()
            (root / 'README.txt').touch()
            styles = evaluator.discover_styles(root, ['real', 'SDXL', 'FLUX', 'SD3'])
            self.assertEqual([p.name for p, _ in styles],
                             ['botanical_watercolor', 'van_gogh_impasto'])
            self.assertEqual([names for _, names in styles], [['SDXL', 'FLUX']] * 2)

    def test_bad_inventory_fails(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'No style folders'):
                evaluator.discover_styles(root, ['SD3'])
            style = Path(root) / 'noir'
            style.mkdir()
            with self.assertRaisesRegex(ValueError, 'no generator folders'):
                evaluator.discover_styles(root, ['SD3'])
            (style / 'unknown_generator').mkdir()
            with self.assertRaisesRegex(ValueError, 'absent from checkpoint'):
                evaluator.discover_styles(root, ['SD3'])
        with self.assertRaisesRegex(ValueError, 'not a directory'):
            evaluator.discover_styles(root, ['SD3'])

    def test_launcher_paths_gpu_visibility_and_missing_checkpoint(self):
        repo = Path(__file__).resolve().parents[1]
        script = repo / 'slurm/slurm_eval_iab_styled.sh'
        stub = r'''
module() { :; }
python() {
    "$TEST_PYTHON" -c 'import json, os, sys; from pathlib import Path; Path(os.environ["CAPTURE"]).write_text(json.dumps({"argv": sys.argv[1:], "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}))' "$@"
}
export -f module python
exec bash "$1"
'''
        with tempfile.TemporaryDirectory(prefix='styled eval ') as directory:
            root = Path(directory)
            base = root / 'hyp_fine_tuning'
            (base / 'bin').mkdir(parents=True)
            (base / 'bin/activate').touch()
            (base / 'hyperbolic_CLIP_riccardo').mkdir()
            data = root / 'datasets/iab_styled_eval'
            data.mkdir(parents=True)
            checkpoint = root / 'checkpoint augmented.pt'
            checkpoint.touch()
            capture = root / 'command.json'
            env = dict(os.environ, WORK=str(root), FAST=str(root), CKPT=str(checkpoint),
                       TEST_PYTHON=sys.executable, CAPTURE=str(capture),
                       CUDA_VISIBLE_DEVICES='3', SLURM_JOB_ID='123')
            for key in ('DATA', 'REPO', 'LOGDIR', 'NUM_WORKERS'):
                env.pop(key, None)
            for override in (False, True):
                if override:
                    env.update(DATA=str(data), REPO=str(repo), NUM_WORKERS='0',
                               LOGDIR=str(root / 'custom results'))
                subprocess.run(['bash', '-c', stub, '_', str(script)], env=env,
                               check=True, capture_output=True, text=True)
                command = json.loads(capture.read_text())
                self.assertEqual(command['argv'][:2],
                                 ['-m', 'comparison.training.test_hypclip_styled'])
                args = evaluator.parse_args(command['argv'][2:])
                self.assertEqual(args.checkpoint, str(checkpoint))
                self.assertEqual(args.root_dir, str(data))
                self.assertEqual(args.num_workers, 0 if override else 8)
                self.assertEqual(args.log_dir, env.get(
                    'LOGDIR', str(root / 'outputs/hypclip_iab_styled_123')))
                self.assertEqual(command['gpu'], '3')
            capture.unlink()
            checkpoint.unlink()
            result = subprocess.run(['bash', '-c', stub, '_', str(script)], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('checkpoint not found', result.stdout)
            self.assertFalse(capture.exists())

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                           ('torch', 'PIL', 'transformers', 'peft', 'sklearn')),
                         'CPU inference needs the training dependencies')
    def test_cpu_inference_keeps_full_label_space_and_final_batch(self):
        from PIL import Image
        import torch
        from torch import nn
        from comparison.training import test_hypclip as shared
        from geometry.lorentz import exp_map0

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.clip = nn.Identity()
                self.projection = nn.Identity()

            def encode_image(self, pixel):
                return exp_map0(pixel, curv=1.), pixel

        class Processor:
            def __call__(self, images, **kwargs):
                color = images.getpixel((0, 0))
                return {'pixel_values': torch.tensor(
                    [[.8, 0.] if color[0] else [0., .8]])}

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for style in ('noir', 'watercolor'):
                for generator, count, color in (
                        ('SDXL', 2, (255, 0, 0)), ('FLUX', 1, (0, 255, 0))):
                    directory = root / 'data' / style / generator / 'COCO'
                    directory.mkdir(parents=True)
                    for i in range(count):
                        Image.new('RGB', (8, 8), color).save(directory / f'{i}.png')
            checkpoint = root / 'model.pt'
            names = ['real', 'FLUX', 'SDXL', 'SD3']
            # FLUX images prefer the absent real class, while SDXL is correct.
            # Balanced accuracy must average the two supported classes only.
            torch.save({'dataset': 'iab', 'class_names': names, 'clip_name': 'tiny',
                        'lora_state': {}, 'projection': {},
                        'anchor_tangent': torch.tensor([[0., .2], [0., -.2],
                                                        [.2, 0.], [-.2, 0.]])}, checkpoint)
            with patch.object(shared.AttributionCLIP, 'from_checkpoint', return_value=TinyModel()), \
                    patch('data.iab_dataset.CLIPImageProcessor.from_pretrained',
                          return_value=Processor()), \
                    patch('torch.cuda.is_available', return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluator.main(['--checkpoint', str(checkpoint), '--root_dir', str(root / 'data'),
                                '--log_dir', str(root / 'results'), '--num_workers', '0',
                                '--batch_size', '2'])
            for style in ('noir', 'watercolor'):
                result = json.loads((root / 'results' / style / 'metrics.json').read_text())
                self.assertEqual(result['num_images'], 3)
                self.assertEqual(result['class_names'], names)
                self.assertEqual(result['support_per_class'], [0, 1, 2, 0])
                self.assertEqual(result['balanced_accuracy'], .5)
                self.assertEqual([row[0] for row in result['confusion_matrix']], [0, 1, 0, 0])
                self.assertEqual(result['confusion_matrix'][2][2], 2)
            self.assertEqual(len((root / 'results/summary.csv').read_text().splitlines()), 3)


if __name__ == '__main__':
    unittest.main()
