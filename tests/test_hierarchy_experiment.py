"""Hybrid experiment regression checks with synthetic images and small features."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from comparison.training import hierarchy_experiment as experiment
from comparison.dataset.ImageAttributionDataset import DATASET
from comparison.dataset.ImageAttributionDataset.dataset import hifi_label_mapping
from data.degradations import apply_degradation
from data.hierarchy_experiment import (
    ExperimentData, FAMILY_GROUPS, IAB_GENERATORS, NEARBY, UNSEEN_LINES, UNSEEN, EXPLORATORY,
    prepare_experiment, protocol_digest,
)
from tests.test_iabench_baselines import make_data
from tests.test_iabench_training import PixelProcessor
from tests.test_attribution_training import TinyModel, TinyTokenizer
from training.families import INCREFA_FAMILIES, increfa_mapping


class HierarchyExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.bench_root, cls.iab_root = cls.root / 'bench', cls.root / 'iab'
        names = [name for family in INCREFA_FAMILIES.values() for name in family]
        cls.bench_manifest, cls.bm = make_data(cls.bench_root, names)
        cls.im = dict(seed=42, use_semantic_split=False, num_images_per_semantic_per_class=2000,
                      train=[], val=[], test=[])
        y, x = np.mgrid[:140, :180]
        pixels = np.stack(((x * 13 + y * 7) % 256, (x * 11 + y) % 256,
                           ((x // 3 + y // 4) % 2) * 255), axis=-1).astype(np.uint8)
        cls.image = Image.fromarray(pixels)
        for name in IAB_GENERATORS:
            for i in range(10):
                relative = f'{name}/COCO/image_p{i}_i0.png'
                path = cls.iab_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                cls.image.save(path)
                cls.im['train' if i < 8 else 'val' if i == 8 else 'test'].append(relative)
        # These aliases must never be imported, even if the source manifest includes them.
        for name in ('FLUX', 'PIXART', 'CogView3_PLUS'):
            for split in ('train', 'val', 'test'):
                cls.im[split].append(f'{name}/COCO/unused_{split}.png')
        cls.iab_manifest = cls.root / 'iab.splits.json'
        cls.iab_manifest.write_text(json.dumps(cls.im))
        cls.directory = cls.root / 'experiment'
        with redirect_stdout(io.StringIO()):
            cls.path = prepare_experiment(cls.bench_root, cls.bench_manifest, cls.iab_root,
                                          cls.iab_manifest, cls.directory)
        cls.data = ExperimentData(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_protocol_membership_and_taxonomy_isolation(self):
        desc = self.data.descriptor
        self.assertEqual(len(desc['class_names']), 32)
        self.assertEqual(len(desc['seen_classes']), 20)
        self.assertEqual(len(desc['unseen_classes']), 10)
        self.assertEqual(desc['exploratory_classes'], list(EXPLORATORY))
        self.assertEqual(list(desc['family_groups']), ['real', 'gan', 'autoregressive', 'diffusion'])
        self.assertEqual([len(group) for group in INCREFA_FAMILIES.values()], [1, 4, 1, 21, 2])
        self.assertEqual(increfa_mapping(desc['seen_classes'], FAMILY_GROUPS)[0], list(FAMILY_GROUPS))
        self.assertIn('latent-consistency@lcm-lora-sdv1-5', desc['seen_classes'])
        self.assertIn('gpt4o', desc['seen_classes'])
        self.assertNotIn('janus-pro', desc['seen_classes'])
        self.assertEqual(len(NEARBY), 3)
        self.assertEqual(sum(map(len, UNSEEN_LINES.values())), 5)
        for line in UNSEEN_LINES.values():
            self.assertFalse(set(line) & set(desc['seen_classes']))
        for split in ('train', 'val', 'test'):
            bench = [f'iabench:{row}' for row in self.bm[split]]
            iab = [f'iab:{path}' for path in self.im[split] if path.split('/')[0] in IAB_GENERATORS]
            expected = bench + iab
            if split != 'test':
                expected = [key for key in expected if self.data.records[key]['generator'] in desc['seen_classes']]
            self.assertEqual(desc['splits'][split], expected)
            for key in desc['splits'][split]:
                self.assertFalse(key.startswith(('iab:FLUX/', 'iab:PIXART/', 'iab:CogView3_PLUS/')))
        groups = [set(desc['splits'][s]) for s in ('train', 'val', 'test')]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_prepare_only_immutable_and_root_overrides(self):
        original = self.bench_manifest.read_bytes(), self.iab_manifest.read_bytes()
        with patch.object(ExperimentData, 'read_image', side_effect=AssertionError('decoded image')), \
                patch('data.iabench_dataset.CLIPImageProcessor.from_pretrained', side_effect=AssertionError('CLIP')), \
                redirect_stdout(io.StringIO()):
            prepare_experiment(self.bench_root, self.bench_manifest, self.iab_root, self.iab_manifest, self.directory)
        self.assertEqual(original, (self.bench_manifest.read_bytes(), self.iab_manifest.read_bytes()))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            shutil.copytree(self.bench_root, target / 'bench')
            shutil.copytree(self.iab_root, target / 'iab')
            relocated = ExperimentData(self.path, target / 'bench', target / 'iab')
            self.assertEqual(relocated.digest, self.data.digest)
            shutil.copytree(self.directory, target / 'sorted')
            sorted_path = target / 'sorted/experiment.json'
            sorted_path.write_text(json.dumps(self.data.descriptor, sort_keys=True))
            reordered = ExperimentData(sorted_path)
            self.assertEqual(list(reordered.descriptor['family_groups']), list(FAMILY_GROUPS))
            self.assertEqual(reordered.digest, self.data.digest)
            damaged = deepcopy(self.im)
            damaged['val'].append(damaged['train'][0])
            path = target / 'bad.json'
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                prepare_experiment(self.bench_root, self.bench_manifest, self.iab_root, path, target / 'bad')
            descriptor = json.loads(self.path.read_text())
            descriptor['seen_classes'][0] = 'wrong'
            shutil.copytree(self.directory, target / 'modified')
            (target / 'modified/experiment.json').write_text(json.dumps(descriptor))
            with self.assertRaisesRegex(ValueError, 'protocol differ'):
                ExperimentData(target / 'modified/experiment.json')
            with self.assertRaisesRegex(ValueError, 'Refusing'):
                prepare_experiment(self.bench_root, self.bench_manifest, self.iab_root,
                                   self.iab_manifest, target / 'modified')

    def test_grok_crop_before_degradation_for_all_methods(self):
        source = self.data.view('test', PixelProcessor())
        row = next(i for i, (_, name, _) in enumerate(source.samples) if name == 'grok3')
        key = source.samples[row][0]
        self.assertEqual(source._read_image(key).size, (80, 90))
        cropped = self.image.crop((0, 0, 80, 90))
        for level in range(7):
            source.degraded = level
            expected = apply_degradation(cropped, level)
            torch.testing.assert_close(source[row]['pixel_values'], PixelProcessor()(images=expected)['pixel_values'][0])
            for method in ('resnet50', 'hifi_net'):
                config = dict(hierarchy_mapping=hifi_label_mapping(source.class_names, 'hybrid'))
                dataset = DATASET[method]('', config=config, iabench_source=source, degraded=level)
                dataset.set_test()
                torch.testing.assert_close(dataset[row]['image'], dataset.transform(expected))
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.data.view('train')._read_image(key)

    def test_probability_aggregation_and_stratified_metrics(self):
        probability = experiment.family_probabilities(torch.tensor([[.4, .3, .3]]).log(), [0, 1, 1], 4)
        torch.testing.assert_close(probability, torch.tensor([[.4, .6, 0., 0.]]))
        desc = self.data.descriptor
        labels, logits, scores = torch.arange(32), torch.zeros(32, 20), torch.zeros(32, 4)
        for i, name in enumerate(desc['class_names']):
            if name in desc['seen_classes']:
                logits[i, desc['seen_classes'].index(name)] = 1
        scores[:, 3] = 1  # Always diffusion: perfect diffusion recall, poor cross-family attribution.
        result = experiment.prediction_metrics(logits, scores, labels, self.data)
        self.assertEqual(result['seen_model_balanced_accuracy'], 1)
        self.assertAlmostEqual(result['unseen_family_balanced_accuracy'], 1 / 3)
        self.assertAlmostEqual(result['unseen_family_macro_generator_recall'], .8)
        self.assertEqual(result['diffusion_nearby_macro_recall'], 1)
        for value in result['exploratory'].values():
            self.assertNotIn('recall', value)
            self.assertNotIn('accuracy', value)
            self.assertEqual(value['family_assignment_proportions']['diffusion'], 1)
        # Only one FLUX variant is correct: five-model mean=.2, four-line mean=.125.
        scores[:, :] = 0
        scores[:, 0] = 1
        scores[desc['class_names'].index(UNSEEN_LINES['FLUX'][0])] = torch.tensor([0., 0., 0., 1.])
        result = experiment.prediction_metrics(logits, scores, labels, self.data)
        self.assertAlmostEqual(result['diffusion_unseen_line_macro_generator_recall'], .2)
        self.assertAlmostEqual(result['diffusion_unseen_line_macro_line_recall'], .125)

    def test_selected_recipes(self):
        args = experiment.ours_args(self.data, Path('/tmp/unused'), 0)
        self.assertEqual((args.num_epochs, args.batch_size, args.lambda_anchor, args.geodesic_margin), (30, 256, .5, 2))
        self.assertEqual(experiment.RECIPES['resnet50'], (10, 32))
        self.assertEqual(experiment.RECIPES['hifi_net'], (10, 8))
        self.assertFalse(args.train_augment)
        self.assertEqual(args.generators, self.data.descriptor['seen_classes'])

    def test_training_restore_and_all_levels(self):
        import train_attribution
        from comparison.training.attributors.attributor_resnet50 import Resnet50Attributor
        from comparison.training.attributors.attributor_hifi_net import HiFiNetAttributor
        from geometry.lorentz import exp_map0
        from losses.attribution_loss import _pairwise_xi
        for method in experiment.METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                stack.enter_context(patch('torch.cuda.is_available', return_value=False))
                stack.enter_context(patch('torch.cuda.device_count', return_value=0))
                stack.enter_context(patch.dict(experiment.RECIPES, {method: (1, 128 if method == 'ours' else 8)}))
                if method == 'ours':
                    real_args = experiment.ours_args
                    def tiny_args(*args, **kwargs):
                        parsed = real_args(*args, **kwargs)
                        parsed.hyperbolic_dim, parsed.init_depth = 4, 0
                        return parsed
                    stack.enter_context(patch.object(experiment, 'ours_args', side_effect=tiny_args))
                    stack.enter_context(patch.object(train_attribution, 'AttributionCLIP', TinyModel))
                    stack.enter_context(patch('transformers.CLIPImageProcessor.from_pretrained', return_value=PixelProcessor()))
                    stack.enter_context(patch('transformers.CLIPTokenizer.from_pretrained', return_value=TinyTokenizer()))
                elif method == 'resnet50':
                    stack.enter_context(patch('torchvision.models.resnet50', return_value=torch.nn.Identity()))
                    stack.enter_context(patch.object(Resnet50Attributor, 'extract_features',
                        side_effect=lambda batch: torch.zeros(len(batch['image']), 2048)))
                else:
                    stack.enter_context(patch('comparison.training.attributors.attributor_hifi_net.get_seg_model',
                                              return_value=torch.nn.Identity()))
                    stack.enter_context(patch.object(HiFiNetAttributor, 'extract_features', return_value=None))
                    def classify(model, features, images):
                        return tuple(getattr(model.SegNet, f'branch_cls_level_{i}').fc(torch.zeros(len(images), 18))
                                     for i in range(1, 5))
                    stack.enter_context(patch.object(HiFiNetAttributor, 'classifier', classify))
                reads = []
                original = self.data.read_image
                def read_training(key):
                    self.assertIn(key, self.data.descriptor['splits']['train'] + self.data.descriptor['splits']['val'])
                    self.assertNotIn(self.data.records[key]['generator'], UNSEEN + EXPLORATORY)
                    reads.append(key)
                    return original(key)
                with patch.object(self.data, 'read_image', side_effect=read_training):
                    checkpoint = experiment.train_experiment(self.data, method, Path(tmp) / 'train', 0)
                self.assertTrue(reads)
                ckpt = torch.load(checkpoint, weights_only=False, map_location='cpu')
                self.assertEqual(len(ckpt['class_names']), 20)
                self.assertEqual(ckpt['family_names'], list(FAMILY_GROUPS))
                self.assertEqual(ckpt['experiment_digest'], self.data.digest)
                if method == 'ours':
                    self.assertEqual(ckpt['anchor_reduction'], 'max')
                    stack.enter_context(patch('models.attribution_clip.AttributionCLIP.from_checkpoint', return_value=TinyModel()))
                loaded = experiment.load_model(self.data, checkpoint, 'cpu')
                self.assertFalse(loaded[1].training)
                reads.clear()
                def read_testing(key):
                    self.assertIn(key, self.data.descriptor['splits']['test'])
                    reads.append(key)
                    return original(key)
                real_metrics = experiment.prediction_metrics
                def check_scores(logits, families, labels, data):
                    self.assertEqual(set(labels.tolist()), set(range(32)))
                    if method == 'ours' and not scored:
                        pixels = torch.cat([PixelProcessor()(images=original(key))['pixel_values']
                                            for key in self.data.descriptor['splits']['test']])
                        images, _ = loaded[1].encode_image(pixels)
                        torch.testing.assert_close(families, -_pairwise_xi(loaded[4], images, 1).T)
                        torch.testing.assert_close(loaded[4], exp_map0(ckpt['family_tangent']))
                    scored.append(True)
                    return real_metrics(logits, families, labels, data)
                scored = []
                output = Path(tmp) / 'results'
                with patch.object(self.data, 'read_image', side_effect=read_testing), \
                        patch.object(experiment, 'prediction_metrics', side_effect=check_scores):
                    experiment.evaluate_experiment(self.data, checkpoint, output, 7, 0)
                self.assertEqual(reads, self.data.descriptor['splits']['test'] * 7)
                self.assertEqual(len(scored), 7)
                for level in range(7):
                    result = json.loads((output / f'test_results_degraded_{level}.json').read_text())
                    self.assertEqual(result['test_samples'], 32)
                    self.assertEqual(sum(v['support'] for v in result['seen_per_generator'].values()), 20)
                    self.assertEqual(sum(v['support'] for v in result['unseen_per_generator'].values()), 10)
                    self.assertEqual(sum(v['support'] for v in result['exploratory'].values()), 2)
                summary = io.StringIO()
                with redirect_stdout(summary):
                    experiment.summarize([output])
                self.assertIn(experiment.LABELS[method], summary.getvalue())
                self.assertIn('diffusion_unseen_line_macro_line_recall', summary.getvalue())
                self.assertIn('midjourneyv6: predicted family share', summary.getvalue())
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    experiment.summarize([output, output])
                damaged = dict(ckpt, experiment_digest='bad')
                torch.save(damaged, Path(tmp) / 'bad.pt')
                with self.assertRaisesRegex(ValueError, 'experiment_digest'):
                    experiment.load_model(self.data, Path(tmp) / 'bad.pt', 'cpu')

    def test_slurm_syntax_and_resources(self):
        for script, gpu, cpus in [('slurm_train_hierarchy_holdout.sh', 2, 16),
                                   ('slurm_train_hierarchy_baseline.sh', 1, 8),
                                   ('slurm_eval_hierarchy_holdout.sh', 1, 8)]:
            path = experiment.REPO / 'slurm' / script
            subprocess.run(['bash', '-n', str(path)], check=True)
            text = path.read_text()
            self.assertIn(f'#SBATCH --gpus-per-node={gpu}', text)
            self.assertIn(f'#SBATCH --cpus-per-task={cpus}', text)
            self.assertNotIn('export CUDA_VISIBLE_DEVICES', text)

    def test_launchers_forward_root_overrides(self):
        with tempfile.TemporaryDirectory(prefix='hybrid launch ') as tmp:
            root = Path(tmp)
            binary = root / 'bin'
            binary.mkdir()
            (binary / 'module').write_text('#!/bin/sh\nexit 0\n')
            (binary / 'python').write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$HYBRID_CAPTURE"\n'
                                         'printf "%s\\n" "$CUDA_VISIBLE_DEVICES" >> "$HYBRID_CAPTURE"\n')
            for path in binary.iterdir():
                path.chmod(0o755)
            activate = root / 'hyp_fine_tuning/bin/activate'
            activate.parent.mkdir(parents=True)
            activate.touch()
            checkpoint = root / 'selected checkpoint.pt'
            checkpoint.touch()
            capture = root / 'capture'
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                       WORK=str(root), REPO=str(experiment.REPO), EXPERIMENT=str(self.path),
                       LOGDIR=str(root / 'output folder'), NUM_WORKERS='0', BATCH_SIZE='7',
                       IABENCH_DATA=str(root / 'bench root'), IAB_DATA=str(root / 'iab root'),
                       SLURM_JOB_ID='123', CUDA_VISIBLE_DEVICES='3', HYBRID_CAPTURE=str(capture))
            for script, argument, command in [('slurm_train_hierarchy_holdout.sh', 'ours', 'train'),
                                              ('slurm_train_hierarchy_baseline.sh', 'hifi_net', 'train'),
                                              ('slurm_eval_hierarchy_holdout.sh', str(checkpoint), 'evaluate')]:
                subprocess.run(['bash', str(experiment.REPO / 'slurm' / script), argument], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                args = capture.read_text().splitlines()
                self.assertEqual(args[:3], ['-m', 'comparison.training.hierarchy_experiment', command])
                self.assertEqual(args[args.index('--iab_root') + 1], str(root / 'iab root'))
                self.assertEqual(args[args.index('--iabench_root') + 1], str(root / 'bench root'))
                self.assertEqual(args[args.index('--output_dir') + 1], str(root / 'output folder'))
                self.assertEqual(args[-1], '3')


if __name__ == '__main__':
    unittest.main()
