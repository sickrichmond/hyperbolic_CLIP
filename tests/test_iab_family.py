"""Checks for the clean IAB-to-IABench family attribution experiment."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
from importlib.util import find_spec
from io import StringIO
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

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

    def test_summary_requires_thirteen_matching_reports(self):
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
            with self.assertRaisesRegex(ValueError, 'Missing .*increfa'):
                experiment.summarize(output)
            changed['cohort']['ordered_test_rows_digest'] = 'reordered'
            paths[-1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'cohort mismatch'):
                experiment.summarize(output)
            self.assertFalse(summary.exists())
            changed['cohort']['ordered_test_rows_digest'] = 'ordered'
            changed['protocol'] = 'iab-family-iabench-unseen-v1'
            paths[-1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Incompatible family attribution report'):
                experiment.summarize(output)
            changed['protocol'] = experiment.PROTOCOL
            paths[-1].write_text(json.dumps(changed))
            self.assertEqual(experiment.summarize(output), summary)
            with summary.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 26)
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
        self.assertIn('#SBATCH --array=0-11', baseline_script.read_text())
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
            native_repo = work / 'IncreFA'
            increfa_checkpoint = native_repo / 'outputs/iab22_static_vitb16_replica2/latest.pt'
            increfa_checkpoint.parent.mkdir(parents=True)
            increfa_checkpoint.touch()
            gallery = work / 'IAB gallery'
            runs = work / 'LIDA runs'
            lida_code = work / 'Igor LIDA'
            lida_env = work / 'igor.env.sh'
            lida_env.write_text(f'DATA="${{DATA:-{gallery}}}"\nRUNS={shlex.quote(str(runs))}\n'
                                f'CODE={shlex.quote(str(lida_code))}\nREPO="$CODE"\n')
            env = dict(os.environ, WORK=str(work), SCRATCH=str(work), REPO=str(repo),
                       SLURM_ARRAY_JOB_ID='123', SLURM_JOB_ID='456', TEST_PYTHON=sys.executable)
            for name in ('DATA', 'SPLIT_MANIFEST', 'LOGDIR', 'BATCH_SIZE', 'NUM_WORKERS',
                         'SLURM_ARRAY_TASK_ID', 'INCREFA_REPO', 'INCREFA_CKPT', 'IAB_EXCLUDE_GENERATORS'):
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
                result = subprocess.run(command, cwd=work, env=dict(env, **overrides), check=True,
                                        capture_output=True, text=True)
                argv, exclusion = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(argv[:4], ['-u', '-m', 'comparison.training.iab_family', 'evaluate'])
                flags = dict(zip(argv[4::2], argv[5::2]))
                self.assertEqual(flags['--root-dir'], str(root))
                self.assertEqual(flags['--split-manifest'], str(manifest))
                return flags, exclusion

            for task, method in enumerate(experiment.BASELINES):
                command = [*baseline, str(custom_map)] if task == 10 else baseline
                flags, exclusion = capture(command, SLURM_ARRAY_TASK_ID=str(task),
                                           LIDA_ENV=str(lida_env), DATA=str(root))
                self.assertEqual(flags['--method'], method)
                self.assertEqual(flags['--output-dir'], str(work / 'outputs/iab_family_baselines_123'))
                if method == 'lida':
                    self.assertEqual(flags['--checkpoint'], str(runs / 'lida_full_iab/T0.pt'))
                    self.assertEqual(flags['--gallery-root'], str(gallery))
                    self.assertNotIn('--checkpoints', flags)
                elif method == 'increfa':
                    self.assertEqual(flags['--checkpoint'], str(increfa_checkpoint))
                    self.assertEqual(flags['--increfa-dir'], str(native_repo))
                    self.assertNotIn('--checkpoints', flags)
                else:
                    self.assertEqual(exclusion, 'dalle3')
                    expected = repo / 'comparison/training/config/iab_styled_final_checkpoints.json'
                    self.assertEqual(flags['--checkpoints'], str(custom_map if task == 10 else expected))
                    self.assertNotIn('--checkpoint', flags)

            for command, overrides in ((hypclip, {}), ([*hypclip, str(checkpoint)], {}),
                                       (baseline, {'SLURM_ARRAY_TASK_ID': '12'}),
                                       ([*hypclip, '--output-dir'], {}),
                                       ([*baseline, '--output-dir'], {'SLURM_ARRAY_TASK_ID': '0'})):
                result = subprocess.run(command, env=dict(env, **overrides),
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)

            checkpoint.touch()
            flags, _ = capture([*hypclip, str(checkpoint)])
            self.assertEqual(flags['--method'], 'hypclip')
            self.assertEqual(flags['--checkpoint'], str(checkpoint))
            self.assertNotIn('--checkpoints', flags)
            self.assertEqual(flags['--output-dir'], str(work / 'outputs/iab_family_hypclip_456'))
            shared = 'shared results'
            flags, _ = capture([*hypclip, '--output-dir', shared, str(checkpoint)])
            self.assertEqual(flags['--output-dir'], str(work / shared))
            flags, _ = capture([*baseline, '--output-dir', shared, str(custom_map)],
                               SLURM_ARRAY_TASK_ID='0', LOGDIR=str(work / 'old results'))
            self.assertEqual(flags['--output-dir'], str(work / shared))
            self.assertEqual(flags['--checkpoints'], str(custom_map))


@unittest.skipUnless(find_spec('torch'), 'requires PyTorch')
class ReadoutTests(unittest.TestCase):
    def test_baselines_sum_softmax_and_dna_uses_native_probabilities(self):
        import torch

        family_of = [2, 3, 2, 1, 0]
        raw = torch.tensor([[-1.1, -1., -1.1, -5., -5.],
                            [3., 4., 3., 0., -1.]])
        for method in ('resnet50', 'lida', 'increfa', 'dna'):
            scores = experiment.baseline_predictions(raw.softmax(1) if method == 'dna' else raw,
                                                     family_of, method)
            self.assertEqual(scores, {'summed_probabilities': [2, 3], 'top_generator_family': [3, 3]})
        probabilities = torch.tensor([[.15, .4, .15, .15, .15]])
        scores = experiment.baseline_predictions(probabilities, family_of, 'dna')
        self.assertEqual(scores['summed_probabilities'], [3])
        with self.assertRaisesRegex(ValueError, 'invalid scores'):
            experiment.baseline_predictions(raw[:, :-1], family_of, 'resnet50')

    def test_patch_native_prediction_uses_average_probabilities(self):
        import torch

        logits = torch.tensor([[[-2., -2., -2.], [0., 4., 4.], [30., 0., 0.], [-2., -2., -2.]]])
        scores = experiment.baseline_predictions(logits.unsqueeze(2), [0, 1, 2, 3], 'patch')
        self.assertEqual(scores, {'summed_probabilities': [1], 'top_generator_family': [1]})

    def test_igor_lida_uses_registered_gallery_and_checkpoint_order(self):
        import torch
        from comparison.training import evaluate_lida_retrieval as native

        names = ['SD3', 'SDXL', '4o', *sorted(experiment.IAB_CLASSES - {'SD3', 'SDXL', '4o'})]
        class Source:
            samples = [(5, 'cogview4', 'a'), (3, 'nano-banana', 'b')]

            def __len__(self):
                return len(self.samples)

        source = Source()
        database = {name: [f'{name}/one.png'] for name in names}
        checkpoint = dict(class_names=names, step=0, database=database, image_size=256,
                          fingerprint='native', backbone={})
        scores = torch.full((2, 22), -10.)
        scores[:, :3] = torch.tensor([-1.1, -1.1, -1.])
        retrieval = ModuleType('lida.retrieval')
        data = ModuleType('lida.data')
        model = ModuleType('lida.model')

        class Gallery:
            def __init__(self, root, paths, classes, **fp):
                self.paths = paths
                self.args = root, classes, fp

            def __len__(self):
                return len(self.paths)

        class Encoder(torch.nn.Module):
            def __init__(self, imagenet_init):
                super().__init__()
                self.backbone = torch.nn.Identity()

        data.FingerprintDataset = Gallery
        data.class_of = lambda path: Path(path).parts[0]
        model.LIDAEncoder = Encoder

        def extract(encoder, dataset, **kwargs):
            if isinstance(dataset, native.RetryGalleryReads):
                self.assertEqual(dataset.dataset.paths, [f'{name}/one.png' for name in names])
                self.assertEqual(dataset.dataset.args, ('gallery', names, dict(size=256, mode='native')))
                return torch.zeros(22, 1), torch.arange(22)
            self.assertEqual(dataset.source.samples, source.samples)
            return torch.zeros(2, 1), torch.arange(2)

        def class_scores(queries, gallery_features, labels, count):
            self.assertEqual((len(queries), len(gallery_features), len(labels), count), (2, 22, 22, 22))
            return scores.numpy(), None

        retrieval.extract_features = extract
        retrieval.class_scores = class_scores
        args = SimpleNamespace(gallery_root=Path('gallery'), batch_size=2, num_workers=0)
        with patch.dict(sys.modules, {'lida': ModuleType('lida'), 'lida.retrieval': retrieval,
                                     'lida.data': data, 'lida.model': model}):
            classes, families, batches = experiment.lida_batches(checkpoint, source, args)
            self.assertEqual(classes, names)
            self.assertEqual(families, experiment.FAMILIES)
            self.assertEqual(list(batches), [
                (['cogview4', 'nano-banana'],
                 {'summed_probabilities': [2, 2], 'top_generator_family': [1, 1]})])
            with self.assertRaisesRegex(ValueError, 'LIDA retrieval checkpoint'):
                experiment.lida_batches(dict(checkpoint, step=1), source, args)

    def test_increfa_uses_native_checkout_transform_and_learned_logits(self):
        import torch

        names = ['real', 'SD3', '4o', 'SDXL',
                 *sorted(experiment.IAB_CLASSES - {'real', 'SD3', '4o', 'SDXL'})]
        checkpoint = dict(config=dict(device='cpu', data=dict(manifest_path='iab', total_classes=22)),
                          results=dict(class_order=[names]), task_id=0, num_learned_classes=22)
        # An extra head column must be sliced before the probability vote.
        columns = torch.full((23,), -10.)
        columns[1:4] = torch.tensor([-1.1, -1., -1.1])
        columns[-1] = 100

        class Source:
            samples = [(5, 'cogview4', 'a'), (3, 'nano-banana', 'b'), (8, 'Imagen3', 'c')]

            def __len__(self):
                return len(self.samples)

            def _read_image(self, row):
                return row

        with TemporaryDirectory() as directory:
            native = Path(directory)
            (native / 'data').mkdir()
            (native / 'data' / '__init__.py').write_text("ORIGIN = 'IncreFA'\n")
            (native / 'train.py').write_text('')
            args = SimpleNamespace(increfa_dir=native, batch_size=2, num_workers=0)
            original_cwd = Path.cwd()
            from comparison.training import test_increfa_styled
            from data import image_io

            def model(images, *, y_gt, store_samples):
                self.assertIsNone(y_gt)
                self.assertFalse(store_samples)
                self.assertEqual(images.tolist(), [15, 13] if len(images) == 2 else [18])
                return columns.expand(len(images), -1), None

            def restore(saved):
                import data
                self.assertEqual(data.ORIGIN, 'IncreFA')
                self.assertEqual(Path.cwd(), native)
                self.assertIs(saved, checkpoint)
                return model, lambda image: image + 10, names

            with patch.object(test_increfa_styled, 'restore_model', side_effect=restore):
                classes, _, batches = experiment.increfa_batches(checkpoint, Source(), args)
                self.assertEqual(classes, names)
                self.assertEqual(list(batches), [
                    (['cogview4', 'nano-banana'],
                     {'summed_probabilities': [2, 2], 'top_generator_family': [1, 1]}),
                    (['Imagen3'], {'summed_probabilities': [2], 'top_generator_family': [1]}),
                ])
            self.assertEqual(Path.cwd(), original_cwd)
            self.assertIs(sys.modules['data.image_io'], image_io)

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
