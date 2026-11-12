"""Checks for the clean IAB-to-IABench family attribution experiment."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
from importlib.util import find_spec
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from comparison.training import iab_family as experiment
from training.families import INCREFA_FAMILIES, increfa_mapping


class CohortAndReportTests(unittest.TestCase):
    def test_release_aliases_and_manifest_order(self):
        names = [name for family in INCREFA_FAMILIES.values() for name in family]
        samples = [(i, name, f'{i}.png') for i, name in enumerate(reversed(names))]
        selected = experiment.select_test_rows(samples)
        expected = {
            'cogview4', 'CompVis@ldm-text2im-large-256', 'CompVis@stable-diffusion-v1-4',
            'Imagen3', 'black-forest-labs@FLUX.1-dev', 'latent-consistency@lcm-lora-sdv1-5',
            'latent-consistency@lcm-lora-sdxl', 'playgroundai@playground-v2-1024px-aesthetic',
            'segmind@SSD-1B', 'segmind@SegMoE-SD-4x2-v0', 'segmind@small-sd', 'segmind@tiny-sd',
            'stabilityai@sdxl-turbo', 'stabilityai@stable-diffusion-2', 'nano-banana',
        }
        self.assertEqual(selected, [row for row in samples if row[1] in expected])
        self.assertEqual(len(selected), 15)
        with self.assertRaisesRegex(ValueError, 'nano-banana'):
            experiment.select_test_rows([row for row in samples if row[1] != 'nano-banana'])

    def test_scored_support_and_nano_banana_assignments(self):
        names = [*experiment.SCORED, experiment.EXPLORATORY]
        samples = [(i, name, f'{i}.png') for i, name in enumerate(names)]
        summed, top = experiment.BASELINE_RULES
        batches = [(names[:-1], {summed: [2] * 14, top: [0] * 14}),
                   (names[-1:], {summed: [3], top: [2]})]
        report = experiment.score_batches(samples, batches, experiment.FAMILIES, (summed, top))
        self.assertEqual(report[summed]['accuracy'], 1)
        self.assertEqual(report[top]['accuracy'], 0)
        self.assertEqual(report[top]['num_scored_images'], 14)
        self.assertEqual(report[top]['family_prediction_counts']['real'], 14)
        self.assertEqual(report[top]['per_generator']['cogview4']['recall'], 0)
        self.assertEqual(report[summed]['nano_banana']['family_assignment_shares']['unknown'], 1)
        self.assertNotIn('accuracy', report[summed]['nano_banana'])
        with self.assertRaisesRegex(ValueError, 'every IABench test row'):
            experiment.score_batches(samples, batches[:-1], experiment.FAMILIES, (summed, top))
        with self.assertRaisesRegex(ValueError, 'ordered IABench test cohort'):
            experiment.score_batches(samples, batches[::-1], experiment.FAMILIES, (summed, top))

    def test_summary_requires_twelve_matching_reports(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory)
            paths = []
            for method in experiment.METHODS:
                rules = experiment.HYPCLIP_RULES if method == 'hypclip' else experiment.BASELINE_RULES
                path = output / method / 'metrics.json'
                paths.append(path)
                path.parent.mkdir()
                path.write_text(json.dumps(dict(
                    protocol=experiment.PROTOCOL, method=method, family_names=list(experiment.FAMILIES),
                    cohort=dict(ordered_test_rows_digest='ordered', num_scored_images=14),
                    checkpoint=f'{method}.pt', epoch=10,
                    rules={rule: dict(accuracy=.5, num_scored_images=14) for rule in rules},
                )))
            summary = output / 'summary.csv'
            changed = json.loads(paths[-1].read_text())
            paths[-1].unlink()
            with self.assertRaisesRegex(ValueError, 'Missing .*pose'):
                experiment.summarize(output)
            changed['cohort']['ordered_test_rows_digest'] = 'reordered'
            paths[-1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'cohort mismatch'):
                experiment.summarize(output)
            self.assertFalse(summary.exists())
            changed['cohort']['ordered_test_rows_digest'] = 'ordered'
            paths[-1].write_text(json.dumps(changed))
            self.assertEqual(experiment.summarize(output), summary)
            with summary.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 24)
            self.assertEqual({row['num_scored_images'] for row in rows}, {'14'})
            self.assertEqual({row['rule'] for row in rows}, set(
                experiment.HYPCLIP_RULES + experiment.BASELINE_RULES))

    def test_cli_requires_explicit_hypclip_checkpoint(self):
        common = ['evaluate', '--method', 'hypclip', '--root-dir', 'data',
                  '--split-manifest', 'splits.json', '--output-dir', 'results']
        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            experiment.parse_args([*common, '--checkpoints', 'map.json'])
        self.assertEqual(experiment.parse_args([*common, '--checkpoint', 'ours.pt']).checkpoint,
                         Path('ours.pt'))

    def test_launchers_dispatch_baselines_and_hypclip_independently(self):
        repo = Path(__file__).resolve().parents[1]
        baseline_script = repo / 'slurm/slurm_eval_iab_family_baselines.sh'
        hypclip_script = repo / 'slurm/slurm_eval_iab_family_hypclip.sh'
        self.assertIn('#SBATCH --array=0-10', baseline_script.read_text())
        self.assertNotIn('#SBATCH --array', hypclip_script.read_text())
        for script in (baseline_script, hypclip_script):
            self.assertIn('#SBATCH --gpus-per-node=1', script.read_text())
            subprocess.run(['bash', '-n', str(script)], check=True)
        with TemporaryDirectory(prefix='family attribution ') as directory:
            work = Path(directory)
            base = work / 'hyp_fine_tuning'
            (base / 'bin').mkdir(parents=True)
            (base / 'bin/activate').touch()
            (base / 'checkpoints').mkdir()
            manifest = base / 'checkpoints/attribution_iabench_random_vitl14.splits.json'
            manifest.touch()
            root = work / 'datasets/IABench_images'
            root.mkdir(parents=True)
            checkpoint = work / 'selected checkpoint.pt'
            custom_map = work / 'alternate map.json'
            custom_map.touch()
            env = dict(os.environ, WORK=str(work), SCRATCH=str(work), REPO=str(repo),
                       SLURM_ARRAY_JOB_ID='123', SLURM_JOB_ID='456', TEST_PYTHON=sys.executable)
            for name in ('DATA', 'SPLIT_MANIFEST', 'LOGDIR', 'BATCH_SIZE', 'NUM_WORKERS',
                         'SLURM_ARRAY_TASK_ID'):
                env.pop(name, None)
            stub = '''
module() { :; }
python() {
    "$TEST_PYTHON" -c 'import json, os, sys; print(json.dumps([sys.argv[1:], os.getenv("IAB_EXCLUDE_GENERATORS")]))' "$@"
}
export -f module python
exec bash "$@"
'''
            baseline = ['bash', '-c', stub, '_', str(baseline_script)]
            hypclip = ['bash', '-c', stub, '_', str(hypclip_script)]

            def capture(command, **overrides):
                result = subprocess.run(command, env=dict(env, **overrides), check=True,
                                        capture_output=True, text=True)
                argv, exclusion = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(argv[:4], ['-u', '-m', 'comparison.training.iab_family', 'evaluate'])
                flags = dict(zip(argv[4::2], argv[5::2]))
                self.assertEqual(flags['--root-dir'], str(root))
                self.assertEqual(flags['--split-manifest'], str(manifest))
                return flags, exclusion

            for task, method in enumerate(experiment.BASELINES):
                command = [*baseline, str(custom_map)] if task == 10 else baseline
                flags, exclusion = capture(command, SLURM_ARRAY_TASK_ID=str(task))
                self.assertEqual(flags['--method'], method)
                self.assertEqual(flags['--output-dir'], str(work / 'outputs/iab_family_baselines_123'))
                self.assertEqual(exclusion, 'dalle3')
                expected = repo / 'comparison/training/config/iab_styled_final_checkpoints.json'
                self.assertEqual(flags['--checkpoints'], str(custom_map if task == 10 else expected))
                self.assertNotIn('--checkpoint', flags)

            for command, overrides in ((hypclip, {}), ([*hypclip, str(checkpoint)], {}),
                                       (baseline, {'SLURM_ARRAY_TASK_ID': '11'})):
                result = subprocess.run(command, env=dict(env, **overrides),
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)

            checkpoint.touch()
            flags, _ = capture([*hypclip, str(checkpoint)])
            self.assertEqual(flags['--method'], 'hypclip')
            self.assertEqual(flags['--checkpoint'], str(checkpoint))
            self.assertNotIn('--checkpoints', flags)
            self.assertEqual(flags['--output-dir'], str(work / 'outputs/iab_family_hypclip_456'))
            shared = str(work / 'shared results')
            flags, _ = capture([*hypclip, str(checkpoint)], LOGDIR=shared)
            self.assertEqual(flags['--output-dir'], shared)
            flags, _ = capture(baseline, SLURM_ARRAY_TASK_ID='0', LOGDIR=shared)
            self.assertEqual(flags['--output-dir'], shared)


@unittest.skipUnless(find_spec('torch'), 'requires PyTorch')
class ReadoutTests(unittest.TestCase):
    def test_baseline_rules_can_disagree_and_dna_uses_raw_scores(self):
        import torch

        family_of = [2, 3, 2, 1, 0]
        raw = torch.tensor([[4., 5., 4., 0., -1.]])
        for method, outputs in (('resnet50', {'logits': raw}),
                                ('dna', {'logits': raw.softmax(1), 'raw_logits': raw})):
            scores = experiment.baseline_predictions(outputs, family_of, method)
            self.assertEqual({key: value.tolist() for key, value in scores.items()},
                             {'summed_logits': [2], 'top_generator_family': [3]})
        with self.assertRaisesRegex(ValueError, 'invalid scores'):
            experiment.baseline_predictions({'logits': raw[:, :-1]}, family_of, 'resnet50')

    def test_patch_native_prediction_uses_average_probabilities(self):
        import torch

        logits = torch.tensor([[[-2., -2., -2.], [0., 4., 4.], [30., 0., 0.], [-2., -2., -2.]]])
        scores = experiment.baseline_predictions({'logits': logits.unsqueeze(2)}, [0, 1, 2, 3], 'patch')
        self.assertEqual(scores['summed_logits'].tolist(), [2])
        self.assertEqual(scores['top_generator_family'].tolist(), [1])

    def test_dna_backbone_preserves_its_native_probabilities(self):
        import torch
        from comparison.training.backbones.dna.models import Simple_CNN

        model = Simple_CNN(5).eval()
        model.main = torch.nn.Identity()
        model.classification_head = torch.nn.Linear(3, 5)
        image = torch.ones(1, 3, 2, 2)
        scores, _ = model(image)
        native, _, raw = model(image, include_raw_logits=True)
        torch.testing.assert_close(native, scores)
        torch.testing.assert_close(raw.softmax(1), scores)

    def test_hypclip_rules_and_validation_use_saved_anchor_order(self):
        import torch
        from geometry.lorentz import exp_map0

        classes = ['real', *sorted(experiment.IAB_CLASSES - {'real'})]
        families, mapping = increfa_mapping(classes, dataset='iab')
        saved_order = ['unknown', 'diffusion', 'real', 'autoregressive']
        checkpoint = dict(
            class_names=classes, family_names=saved_order, hierarchy='increfa', hyperbolic_dim=2,
            family_of=[saved_order.index(families[i]) for i in mapping],
            family_tangent=torch.tensor([[0., 2.], [.9, 0.], [-2., 0.], [2., .1]]),
        )
        _, names, tangents, curv = experiment.validate_hypclip_checkpoint(checkpoint)
        images = exp_map0(torch.tensor([[2., 0.]]), curv=curv)
        scores = experiment.hypclip_predictions(images, exp_map0(tangents, curv=curv), curv)
        self.assertEqual({key: names[value[0]] for key, value in scores.items()},
                         {'minimum_exterior_angle': 'diffusion', 'nearest_geodesic_anchor': 'autoregressive'})
        for change in (dict(dataset='iabench'), dict(hierarchy='none'), dict(loss='ce'),
                       dict(class_names=classes[:-1]), dict(family_of=mapping),
                       dict(family_tangent=torch.zeros(4, 2))):
            with self.subTest(keys=list(change)), self.assertRaises(ValueError):
                experiment.validate_hypclip_checkpoint(dict(checkpoint, **change))


@unittest.skipUnless(all(find_spec(name) for name in ('torch', 'torchvision', 'datasets', 'transformers')),
                     'requires the IABench data and tensor dependencies')
class NativeLoaderTests(unittest.TestCase):
    def test_validated_test_only_native_batches_keep_final_rows(self):
        from datasets import Features, Image
        from PIL import Image as PILImage
        from comparison.training.iabench import get_iabench_test_loader
        from data.iabench_dataset import IABenchDataset, IMAGE_INDEX

        names = [name for family in INCREFA_FAMILIES.values() for name in family]
        labels = [name for name in names for _ in range(10)]
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'data'
            root.mkdir()
            PILImage.new('RGB', (32, 32), 'red').save(root / 'sample.png')
            rows = [dict(label=label, file_name=f'{i}.png', image_path='sample.png')
                    for i, label in enumerate(labels)]
            metadata = ''.join(json.dumps([row['label'], row['file_name']]) + '\n' for row in rows)
            (root / IMAGE_INDEX).write_text(json.dumps(dict(
                format='iabench-images-v1', complete=True, rows=rows,
                dataset_digest=hashlib.sha256(metadata.encode()).hexdigest(),
                image_feature=Features({'image': Image()}).to_dict()['image'],
            )))
            manifest = IABenchDataset(root, processor_name=None).make_split_manifest(generators=names)
            path = Path(directory) / 'splits.json'
            path.write_text(json.dumps(manifest))
            source, cohort = experiment.load_cohort(root, path)
            self.assertEqual(source.indices, [i for i in manifest['test']
                                              if labels[i] in (*experiment.SCORED, experiment.EXPLORATORY)])
            self.assertEqual(cohort['num_scored_images'], 14)
            loader = get_iabench_test_loader(source, {'model_name': 'resnet50'}, 4, 0)
            batches = [batch['label'].tolist() for batch in loader]
            self.assertEqual(list(map(len, batches)), [4, 4, 4, 3])
            self.assertEqual([source.class_names[i] for batch in batches for i in batch],
                             [labels[i] for i in source.indices])
            manifest['test'][0] = manifest['train'][0]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'metadata or membership'):
                experiment.load_cohort(root, path)


if __name__ == '__main__':
    unittest.main()
