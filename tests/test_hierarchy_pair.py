"""CPU checks for pair distances, family recovery, plots and positional Slurm arguments."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from comparison.training import hierarchy_pair as pair
from geometry.lorentz import exp_map0, pairwise_dist
from tests import test_hierarchy_experiment as fixtures
from tests.test_attribution_training import TinyModel
from tests.test_iabench_training import PixelProcessor
from training.families import increfa_mapping


class HierarchyPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.HierarchyExperimentTests.setUpClass()
        cls.data = fixtures.HierarchyExperimentTests.data

    @classmethod
    def tearDownClass(cls):
        fixtures.HierarchyExperimentTests.tearDownClass()

    def test_ordering_probability_and_ties(self):
        seen, unseen = torch.tensor([1., 2., 3.]), torch.tensor([2., 3., 4., 5.])
        expected = ((seen[:, None] < unseen).double() + .5 * (seen[:, None] == unseen)).mean().item()
        self.assertAlmostEqual(pair.ordering_probability(seen, unseen), expected)
        self.assertEqual(pair.ordering_probability(torch.tensor([1., 1.]), torch.tensor([1.])), .5)
        self.assertEqual(pair.ordering_probability(torch.tensor([1.]), torch.tensor([2., 3.])), 1.)
        self.assertEqual(pair.ordering_probability(torch.tensor([3.]), torch.tensor([1., 2.])), 0.)
        for invalid in (torch.empty(0), torch.tensor([-1.]), torch.tensor([float('nan')]), torch.ones(2, 2)):
            with self.assertRaisesRegex(ValueError, 'Distances'):
                pair.ordering_probability(invalid, unseen)

    def test_paired_counts_mixed_competition_and_zero_denominator(self):
        models = torch.tensor([[1., 4., 5.], [1.2, 3., 4.],
                               [.5, 2., 3.], [.3, 2., 4.], [.2, 2., 4.], [4., 3., .1]])
        families = torch.tensor([[4., 3., 2., 1.], [4., 3., 2., 1.],
                                 [4., 3., 2., .4], [4., 3., 2., .3], [.7, 3., 2., .9], [4., .2, 3., 2.]])
        seen = torch.tensor([True, True, False, False, False, False])
        mapping = torch.tensor([3, 3, 1])
        result = pair.pair_statistics(models, families, seen, mapping, 0, 3)
        self.assertEqual(result['ordering_probability'], .25)
        self.assertEqual(result['nearest_overall_counts'],
                         dict(explicit_target_family=1, wrong_generator_same_family=2, other_family=1))
        self.assertEqual(result['nearest_family_counts'], [1, 1, 0, 2])
        self.assertEqual(result['paired_recovery'], dict(reference_assignments=2, correct_family=1, fraction=.5))
        # The .3 model/family tie selects the model; family-only attribution still succeeds.
        result = pair.pair_statistics(models, families, seen, mapping, 1, 3)
        self.assertEqual(result['paired_recovery'], dict(reference_assignments=0, correct_family=0, fraction=None))
        json.dumps(result, allow_nan=False)
        for bad_seen, bad_mapping in ((torch.ones(6, dtype=torch.bool), mapping),
                                      (seen, torch.tensor([-1, 3, 1]))):
            with self.assertRaisesRegex(ValueError, 'Invalid pair'):
                pair.pair_statistics(models, families, bad_seen, bad_mapping, 0, 3)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            pair.pair_statistics(-models, families, seen, mapping, 0, 3)

    def test_only_selected_test_rows_and_saved_distances(self):
        desc = self.data.descriptor
        model = TinyModel().eval()
        g = torch.Generator().manual_seed(8)
        anchors = exp_map0(torch.nn.functional.normalize(torch.randn(20, 4, generator=g), dim=1) * 2)
        families = exp_map0(torch.nn.functional.normalize(torch.randn(4, 4, generator=g), dim=1) * 1.2)
        config = dict(clip_name='synthetic', curv=1., family_of=increfa_mapping(desc['seen_classes'], desc['family_groups'])[1])
        selected = [key for key in desc['splits']['test']
                    if self.data.records[key]['generator'] in (pair.DEFAULT_SEEN, pair.DEFAULT_UNSEEN)]
        original, reads = self.data.read_image, []
        def read(key):
            self.assertIn(key, selected)
            reads.append(key)
            return original(key)
        with tempfile.TemporaryDirectory() as tmp, \
                patch('torch.cuda.is_available', return_value=False), \
                patch.object(pair, 'load_model', return_value=('ours', model, config, anchors, families)) as load, \
                patch('transformers.CLIPImageProcessor.from_pretrained', return_value=PixelProcessor()), \
                patch.object(self.data, 'read_image', side_effect=read), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            output = Path(tmp) / 'result'
            result = pair.evaluate_pair(self.data, 'checkpoint.pt', output, batch_size=3, workers=0)
            load.assert_called_once()
            self.assertEqual(reads, selected)
            self.assertEqual((result['seen_samples'], result['unseen_samples']), (1, 1))
            saved = torch.load(output / 'pair_distances.pt', weights_only=True)
            self.assertEqual(saved['sample_keys'], selected)
            self.assertEqual(saved['model_anchor_names'], desc['seen_classes'])
            self.assertEqual(saved['experiment_digest'], self.data.digest)
            self.assertEqual(saved['degradation'], 'clean')
            pixels = torch.cat([PixelProcessor()(images=original(key))['pixel_values'] for key in selected])
            with torch.no_grad():
                images, _ = model.encode_image(pixels)
            torch.testing.assert_close(saved['model_distances'], pairwise_dist(images.double(), anchors.double()))
            torch.testing.assert_close(saved['family_distances'], pairwise_dist(images.double(), families.double()))
            self.assertEqual(json.loads((output / 'pair_results.json').read_text()), result)
            for name in ('distance_to_seen_anchor', 'family_recovery'):
                for extension in ('png', 'pdf'):
                    self.assertGreater((output / f'{name}.{extension}').stat().st_size, 1000)
            with self.assertRaisesRegex(ValueError, 'fresh diagnostic'):
                pair.evaluate_pair(self.data, 'checkpoint.pt', output, workers=0)
            for seen, unseen in ((pair.DEFAULT_UNSEEN, pair.DEFAULT_UNSEEN),
                                 (pair.DEFAULT_SEEN, pair.DEFAULT_SEEN), (pair.DEFAULT_SEEN, 'nano-banana')):
                with self.assertRaisesRegex(ValueError, 'generator must'):
                    pair.evaluate_pair(self.data, 'checkpoint.pt', Path(tmp) / 'invalid', seen, unseen)
            self.assertEqual(load.call_count, 1)

    def test_histogram_kde_and_empty_paired_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = dict(model_anchor_names=['seen', 'other', 'gan'],
                         model_distances=torch.tensor([[.5, 2., 3.], [1., 2., 3.], [1.5, 2., 3.],
                                                       [2., 1., 3.], [3., 1., 3.], [4., 1., 3.]]),
                         is_seen=torch.tensor([True, True, True, False, False, False]))
            result = dict(seen_generator='seen', unseen_generator='unseen', target_family='diffusion',
                          family_names=['real', 'gan', 'autoregressive', 'diffusion'], unseen_samples=3,
                          ordering_probability=1., nearest_overall_counts=dict(
                              explicit_target_family=0, wrong_generator_same_family=3, other_family=0),
                          nearest_family_counts=[1, 1, 0, 1],
                          paired_recovery=dict(correct_family=0, reference_assignments=0, fraction=None))
            pair.plot_pair(saved, result, Path(tmp))
            self.assertTrue((Path(tmp) / 'family_recovery.png').is_file())

    def test_slurm_defaults_overrides_and_argument_validation(self):
        script = Path(pair.__file__).resolve().parents[2] / 'slurm/slurm_plot_hierarchy_pair.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        self.assertIn('#SBATCH --gpus-per-node=1', script.read_text())
        self.assertNotIn('export CUDA_VISIBLE_DEVICES', script.read_text())
        with tempfile.TemporaryDirectory(prefix='pair launcher ') as tmp:
            root = Path(tmp)
            binary = root / 'bin'
            binary.mkdir()
            (binary / 'module').write_text('#!/bin/sh\nexit 0\n')
            (binary / 'python').write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$PAIR_CAPTURE"\n'
                                         'printf "%s\\n" "$CUDA_VISIBLE_DEVICES" >> "$PAIR_CAPTURE"\n')
            for path in binary.iterdir():
                path.chmod(0o755)
            activate = root / 'hyp_fine_tuning/bin/activate'
            activate.parent.mkdir(parents=True)
            activate.touch()
            protocol = root / 'frozen protocol/experiment.json'
            protocol.parent.mkdir()
            protocol.touch()
            checkpoint = protocol.parent / 'runs/ours_62755694/ours_anchor0p5_z2_ep30.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            capture = root / 'capture'
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'], WORK=str(root),
                       REPO=str(script.parents[1]), EXPERIMENT=str(protocol), LOGDIR=str(root / 'output folder'),
                       SLURM_JOB_ID='123', CUDA_VISIBLE_DEVICES='3', NUM_WORKERS='0', BATCH_SIZE='3',
                       IABENCH_DATA=str(root / 'bench root'), IAB_DATA=str(root / 'iab root'), PAIR_CAPTURE=str(capture))
            env.pop('CKPT', None)
            for positional in ([], ['E4S'], ['stabilityai@stable-diffusion-xl-base-1.0', 'stabilityai@sdxl-turbo']):
                subprocess.run(['bash', str(script), *positional], env=env, check=True, capture_output=True)
                args = capture.read_text().splitlines()
                self.assertEqual(args[:4], ['-m', 'comparison.training.hierarchy_pair',
                                           positional[0] if positional else pair.DEFAULT_SEEN,
                                           positional[1] if len(positional) > 1 else pair.DEFAULT_UNSEEN])
                self.assertEqual(args[args.index('--checkpoint') + 1], str(checkpoint))
                self.assertEqual(args[args.index('--iabench_root') + 1], str(root / 'bench root'))
                self.assertEqual(args[args.index('--iab_root') + 1], str(root / 'iab root'))
                self.assertEqual(args[args.index('--output_dir') + 1], str(root / 'output folder'))
                self.assertEqual(args[-1], '3')
            override = root / 'custom checkpoint.pt'
            override.touch()
            subprocess.run(['bash', str(script)], env=dict(env, CKPT=str(override)), check=True, capture_output=True)
            args = capture.read_text().splitlines()
            self.assertEqual(args[args.index('--checkpoint') + 1], str(override))
            invalid = subprocess.run(['bash', str(script), 'one', 'two', 'three'], env=env, capture_output=True)
            self.assertEqual(invalid.returncode, 2)


if __name__ == '__main__':
    unittest.main()
