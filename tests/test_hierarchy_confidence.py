"""CPU checks for all-generator confidence scores, correctness, saved points and launcher."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from comparison.training import hierarchy_confidence as confidence
from geometry.lorentz import exp_map0, oxy_angle, pairwise_dist
from tests import test_hierarchy_experiment as fixtures
from tests.test_attribution_training import TinyModel
from tests.test_iabench_training import PixelProcessor
from training.families import increfa_mapping


class HierarchyConfidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.HierarchyExperimentTests.setUpClass()
        cls.data = fixtures.HierarchyExperimentTests.data

    @classmethod
    def tearDownClass(cls):
        fixtures.HierarchyExperimentTests.tearDownClass()

    def example(self):
        desc = self.data.descriptor
        models, families = torch.full((32, 20), 9., dtype=torch.float64), torch.full((32, 4), 9., dtype=torch.float64)
        parents = {name: i for i, group in enumerate(desc['family_groups'].values()) for name in group}
        for row, name in enumerate(desc['class_names']):
            models[row, desc['seen_classes'].index(name) if name in desc['seen_classes'] else 0] = 1.5
            families[row, parents.get(name, 0)] = 1.2
        return models, families, torch.arange(32), desc

    def test_formula_correctness_and_missing_family_targets(self):
        models, families, labels, desc = self.example()
        families[0] = torch.tensor([9., 9., 9., 1.2])  # Correct model name can coexist with wrong family.
        row = desc['class_names'].index('E4S')
        models[row] = 9.
        models[row, 0] = 1.5
        points = confidence.prediction_points(models, families, labels, desc)
        expected = models.clamp_min(1e-8).reciprocal().softmax(1)
        torch.testing.assert_close(points['confidence'], expected.max(1).values)
        self.assertEqual(int(points['is_seen'].sum()), 20)
        self.assertEqual(int(points['model_correct'].sum()), 19)
        self.assertFalse(points['model_correct'][~points['is_seen']].any())
        self.assertEqual(int((points['target_family_indices'] == -1).sum()), 2)
        self.assertTrue(points['model_correct'][0])
        self.assertFalse(points['family_correct'][0])
        result = confidence.confidence_statistics(points, desc)
        self.assertEqual(result['seen']['model_accuracy'], .95)
        self.assertEqual(result['seen']['family_accuracy'], .95)
        self.assertEqual(result['unseen']['samples'], 12)
        self.assertEqual(result['unseen']['family_target_samples'], 10)
        self.assertEqual(result['unseen']['model_accuracy'], 0)
        self.assertEqual(result['unseen']['family_accuracy'], 1)
        self.assertEqual(result['seen_unseen_score_ordering_probability'], .5)
        for name in ('midjourneyv6', 'nano-banana'):
            self.assertIsNone(result['per_generator'][name]['family_accuracy'])
        for invalid in (torch.full_like(models, float('nan')), -models, models[:, :3]):
            with self.assertRaises(ValueError):
                confidence.prediction_points(invalid, families, labels, desc)

    def test_every_point_density_curves_and_unchanged_scores(self):
        import matplotlib.axes
        models, families, labels, desc = self.example()
        points = dict(confidence.prediction_points(models, families, labels, desc), model_anchor_names=desc['seen_classes'])
        points['confidence'][points['is_seen']] = .0504
        points['confidence'][~points['is_seen']] = torch.linspace(.0503, .0505, 12, dtype=torch.float64)
        before = {key: value.clone() for key, value in points.items() if isinstance(value, torch.Tensor)}
        plotted, curves = [], []
        scatter = matplotlib.axes.Axes.scatter
        plot = matplotlib.axes.Axes.plot
        def record(ax, x, y, **kwargs):
            plotted.append((np.asarray(x).copy(), np.asarray(y).copy()))
            return scatter(ax, x, y, **kwargs)
        def record_curve(ax, x, y, **kwargs):
            curves.append((x.copy(), y.copy()))
            return plot(ax, x, y, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(matplotlib.axes.Axes, 'scatter', record), \
                patch.object(matplotlib.axes.Axes, 'plot', record_curve):
            confidence.plot_confidence(points, tmp)
            self.assertEqual([len(x) for x, _ in plotted], [20, 12, 20, 12])
            np.testing.assert_array_equal(plotted[0][0], points['confidence'][points['is_seen']].numpy())
            np.testing.assert_array_equal(plotted[1][0], points['confidence'][~points['is_seen']].numpy())
            self.assertEqual(int((plotted[1][1] < .5).sum()), 2)
            self.assertEqual(len(curves), 4)
            for (grid, density), detail_curve in zip(curves[:2], curves[2:]):
                np.testing.assert_array_equal((grid, density), detail_curve)
                self.assertTrue(np.isfinite(density).all())
                area = np.sum(np.diff(grid) * (density[:-1] + density[1:]) / 2)
                self.assertAlmostEqual(area, 1, places=3)
            for suffix in ('', '_detail'):
                for extension in ('png', 'pdf'):
                    self.assertGreater((Path(tmp) / f'confidence_family{suffix}.{extension}').stat().st_size, 1000)
        for key, value in before.items():
            torch.testing.assert_close(points[key], value)

    def test_full_test_collection_and_offline_replot(self):
        desc = self.data.descriptor
        model = TinyModel().eval()
        g = torch.Generator().manual_seed(32)
        anchors = exp_map0(torch.nn.functional.normalize(torch.randn(20, 4, generator=g), dim=1) * 2)
        families = exp_map0(torch.nn.functional.normalize(torch.randn(4, 4, generator=g), dim=1) * 1.2)
        config = dict(clip_name='synthetic', curv=1., family_of=increfa_mapping(desc['seen_classes'], desc['family_groups'])[1])
        original, reads = self.data.read_image, []
        def read(key):
            self.assertIn(key, desc['splits']['test'])
            reads.append(key)
            return original(key)
        with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.is_available', return_value=False), \
                patch.object(confidence, 'load_model', return_value=('ours', model, config, anchors, families)), \
                patch('transformers.CLIPImageProcessor.from_pretrained', return_value=PixelProcessor()), \
                patch.object(self.data, 'read_image', side_effect=read), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            output = Path(tmp) / 'results'
            result = confidence.evaluate_confidence(self.data, 'checkpoint.pt', output, batch_size=7, workers=0)
            self.assertEqual(reads, desc['splits']['test'])
            self.assertEqual(result['seen']['samples'], 20)
            self.assertEqual(result['unseen']['samples'], 12)
            self.assertFalse(result['calibrated'])
            path = output / 'confidence_points.pt'
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved['sample_keys'], desc['splits']['test'])
            pixels = torch.cat([PixelProcessor()(images=original(key))['pixel_values'] for key in saved['sample_keys']])
            with torch.no_grad():
                images, _ = model.encode_image(pixels)
            torch.testing.assert_close(saved['model_distances'], pairwise_dist(images.double(), anchors.double()))
            torch.testing.assert_close(saved['family_distances'], pairwise_dist(images.double(), families.double()))
            self.assertEqual(json.loads((output / 'confidence_results.json').read_text()), result)
            before = path.read_bytes()
            with patch.object(confidence, 'load_model', side_effect=AssertionError('model loaded')), \
                    patch.object(confidence, 'ExperimentData', side_effect=AssertionError('images loaded')):
                confidence.main(['--points', str(path), '--output_dir', str(Path(tmp) / 'replot')])
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'fresh confidence'):
                confidence.evaluate_confidence(self.data, 'checkpoint.pt', output, workers=0)
            saved['confidence'][0] = 0.
            torch.save(saved, Path(tmp) / 'damaged.pt')
            with self.assertRaisesRegex(ValueError, 'do not match'):
                confidence.main(['--points', str(Path(tmp) / 'damaged.pt'), '--output_dir', str(Path(tmp) / 'invalid')])
            reads.clear()
            angle_output = Path(tmp) / 'angle'
            angle_result = confidence.evaluate_confidence(self.data, 'checkpoint.pt', angle_output,
                                                          batch_size=7, workers=0, metric='exterior_angle')
            self.assertEqual(reads, desc['splits']['test'])
            self.assertEqual(angle_result['model_score'], 'max_softmax_reciprocal_exterior_angle')
            angle_path = angle_output / 'confidence_points.pt'
            angles = torch.load(angle_path, weights_only=True)
            self.assertEqual(angles['angle_unit'], 'radians')
            for key, anchor_set in (('model_angles', anchors), ('family_angles', families)):
                expected = torch.stack([oxy_angle(a.double().expand_as(images), images.double())
                                        for a in anchor_set], dim=1)
                torch.testing.assert_close(angles[key], expected)
            expected = angles['model_angles'].clamp_min(1e-8).reciprocal().softmax(1)
            torch.testing.assert_close(angles['confidence'], expected.max(1).values)
            torch.testing.assert_close(angles['predicted_model_indices'], angles['model_angles'].argmin(1))
            torch.testing.assert_close(angles['predicted_family_indices'], angles['family_angles'].argmin(1))
            with patch.object(confidence, 'load_model', side_effect=AssertionError('model loaded')):
                confidence.main(['--points', str(angle_path), '--output_dir', str(Path(tmp) / 'angle_replot')])

    def test_slurm_launch_and_overrides(self):
        script = Path(confidence.__file__).resolve().parents[2] / 'slurm/slurm_plot_hierarchy_confidence.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        self.assertIn('#SBATCH --gpus-per-node=1', script.read_text())
        self.assertNotIn('export CUDA_VISIBLE_DEVICES', script.read_text())
        with tempfile.TemporaryDirectory(prefix='confidence launcher ') as tmp:
            root = Path(tmp)
            binary = root / 'bin'
            binary.mkdir()
            (binary / 'module').write_text('#!/bin/sh\nexit 0\n')
            (binary / 'python').write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CONFIDENCE_CAPTURE"\n'
                                         'printf "%s\\n" "$CUDA_VISIBLE_DEVICES" >> "$CONFIDENCE_CAPTURE"\n')
            for path in binary.iterdir():
                path.chmod(0o755)
            activate = root / 'hyp_fine_tuning/bin/activate'
            activate.parent.mkdir(parents=True)
            activate.touch()
            protocol = root / 'protocol/experiment.json'
            protocol.parent.mkdir()
            protocol.touch()
            ckpt = root / 'selected checkpoint.pt'
            ckpt.touch()
            capture = root / 'capture'
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'], WORK=str(root),
                       REPO=str(script.parents[1]), EXPERIMENT=str(protocol), SLURM_JOB_ID='123',
                       CUDA_VISIBLE_DEVICES='3', LOGDIR=str(root / 'output folder'), NUM_WORKERS='0',
                       IABENCH_DATA=str(root / 'bench root'), IAB_DATA=str(root / 'iab root'),
                       CONFIDENCE_CAPTURE=str(capture))
            env.pop('CKPT', None)
            missing = subprocess.run(['bash', str(script)], env=env, capture_output=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn(b'Set CKPT', missing.stderr)
            self.assertFalse(capture.exists())
            env['CKPT'] = str(ckpt)
            env.pop('METRIC', None)
            subprocess.run(['bash', str(script)], env=env, check=True, capture_output=True)
            args = capture.read_text().splitlines()
            self.assertEqual(args[:2], ['-m', 'comparison.training.hierarchy_confidence'])
            self.assertEqual(args[args.index('--checkpoint') + 1], str(ckpt))
            self.assertEqual(args[args.index('--metric') + 1], 'geodesic')
            self.assertEqual(args[args.index('--iabench_root') + 1], str(root / 'bench root'))
            self.assertEqual(args[args.index('--iab_root') + 1], str(root / 'iab root'))
            self.assertEqual(args[-1], '3')
            subprocess.run(['bash', str(script)], env=dict(env, METRIC='exterior_angle'), check=True, capture_output=True)
            args = capture.read_text().splitlines()
            self.assertEqual(args[args.index('--metric') + 1], 'exterior_angle')
            invalid = subprocess.run(['bash', str(script), 'SD2'], env=env, capture_output=True)
            self.assertEqual(invalid.returncode, 2)


if __name__ == '__main__':
    unittest.main()
