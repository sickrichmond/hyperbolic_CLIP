"""Evaluate IAB-trained models on unseen generators in the clean IABench test split."""
import argparse
from collections import Counter
import csv
from functools import partial
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from comparison.training.test_baselines_styled import (
    METHODS as STYLED_BASELINES, load_checkpoint_map, prepare_config,
)
from training.families import IAB_FAMILIES, INCREFA_FAMILIES, increfa_mapping


BASELINES = (*STYLED_BASELINES, 'increfa')
METHODS = ('hypclip', *BASELINES)
FAMILIES = tuple(IAB_FAMILIES)
BASELINE_RULES = ('summed_probabilities', 'top_generator_family')
HYPCLIP_RULES = ('minimum_exterior_angle', 'nearest_geodesic_anchor')
PROTOCOL = 'iab-family-iabench-unseen-v2'
IAB_CLASSES = {name for group in IAB_FAMILIES.values() for name in group} - {'dalle3'}
# IABench labels for generator releases already present in the 22-class IAB runs.
SEEN_RELEASES = {
    'gpt4o': '4o',
    'midjourneyv6': 'mid-6.0',
    'black-forest-labs@FLUX.1-schnell': 'FLUX',
    'PixArt-alpha@PixArt-XL-2-1024-MS': 'PIXART',
    'playgroundai@playground-v2.5-1024px-aesthetic': 'PLAYGROUND_2_5',
    'runwayml@stable-diffusion-v1-5': 'SD1_5',
    'stabilityai@stable-diffusion-2-1': 'SD2_1',
    'stabilityai@stable-diffusion-3-medium-diffusers': 'SD3',
    'stabilityai@stable-diffusion-xl-base-1.0': 'SDXL',
}
SCORED = tuple(name for name in INCREFA_FAMILIES['diffusion'] if name not in SEEN_RELEASES)
EXPLORATORY = 'nano-banana'


def select_test_rows(samples):
    eligible = {*SCORED, EXPLORATORY}
    selected = [sample for sample in samples if sample[1] in eligible]
    missing = eligible - {name for _, name, _ in selected}
    if missing:
        raise ValueError(f'IABench test split is missing eligible generators: {sorted(missing)}')
    return selected


def load_cohort(root, split_manifest):
    from data.iabench_dataset import IABenchDataset, manifest_digest

    manifest = json.loads(split_manifest.read_text())
    names = manifest['class_names']
    expected_names = {name for group in INCREFA_FAMILIES.values() for name in group}
    if (names[:1] != ['real'] or len(names) != len(expected_names) or set(names) != expected_names
            or (manifest['seed'], manifest['val_frac'], manifest['test_frac']) != (42, .1, .1)):
        raise ValueError('Expected the full, real-first IABench seed-42 80/10/10 manifest')
    dataset = IABenchDataset(root, processor_name=None)
    test = dataset.split_view(manifest, 'test')
    selected = select_test_rows(test.samples)
    test._set_indices([row for row, _, _ in selected])
    counts = Counter(name for _, name, _ in test.samples)
    cohort = dict(dataset='iabench', split='test', condition='clean',
                  dataset_digest=test.dataset_digest, split_manifest_digest=manifest_digest(manifest),
                  ordered_test_rows_digest=manifest_digest(test.samples),
                  excluded_seen_releases=SEEN_RELEASES,
                  support_by_generator=dict(counts), num_images=len(test),
                  num_scored_images=sum(counts[name] for name in SCORED))
    return test, cohort


def baseline_predictions(scores, family_of, method):
    import torch

    if (scores.ndim != (4 if method == 'patch' else 2)
            or scores.shape[1] != len(family_of) or not torch.isfinite(scores).all()):
        raise ValueError(f'{method}: invalid scores for the IAB generator head')
    # DNA already returns post-softmax probabilities from its native head.
    probabilities = scores.float() if method == 'dna' else scores.float().softmax(dim=1)
    if method == 'patch':
        probabilities = probabilities.mean(dim=(2, 3))
    mapping = torch.as_tensor(family_of, device=scores.device)
    sums = torch.stack([probabilities[:, mapping == i].sum(dim=1) for i in range(len(FAMILIES))], dim=1)
    top = probabilities if method in ('patch', 'repmix') else scores
    return dict(zip(BASELINE_RULES, (sums.argmax(dim=1).tolist(), mapping[top.argmax(dim=1)].tolist())))


def baseline_batches(method, checkpoint, source, args):
    import torch
    import yaml
    from comparison.dataset.ImageAttributionDataset.dataset import model_class_to_label
    from comparison.training.attributors import ATTRIBUTOR
    from comparison.training.iabench import get_iabench_test_loader
    from comparison.training.logger import create_logger
    from comparison.training.test import init_seed
    from comparison.training.trainer.trainer import Trainer

    class_names = list(model_class_to_label)
    if set(class_names) != IAB_CLASSES:
        raise ValueError('Set IAB_EXCLUDE_GENERATORS=dalle3 for the 22-class IAB checkpoints')
    _, family_of = increfa_mapping(class_names, dataset='iab')
    config_name = 'dna_default' if method == 'dna' else method
    config_path = Path(__file__).parent / 'config' / 'model' / f'{config_name}.yaml'
    config = prepare_config(checkpoint, method, yaml.safe_load(config_path.read_text()), class_names)
    init_seed(config.get('manualSeed', 42), torch.cuda.is_available())
    if method not in ATTRIBUTOR.data:
        raise RuntimeError(f'{method} is unavailable; check the optional dependency import warnings')
    model = ATTRIBUTOR[method](config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    log_dir = args.output_dir / method
    logger = create_logger(str(log_dir / 'test.log'))
    trainer = Trainer(config, model, None, None, logger, log_dir=str(log_dir), test_mode=True)
    preprocess = model.semantic_extractor.clip_preprocess if method == 'defl' else None
    # The adapter's labels index the IABench source while the model's head stays in IAB order.
    loader_config = {key: value for key, value in config.items() if key != 'hierarchy_mapping'}
    loader = get_iabench_test_loader(source, loader_config, args.batch_size, args.num_workers,
                                    clip_preprocess=preprocess)

    def batches():
        for batch in loader:
            generators = [source.class_names[i] for i in batch['label'].reshape(-1).tolist()]
            batch = {key: value.to(trainer.device) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            yield generators, baseline_predictions(trainer.inference(batch)['logits'], family_of, method)

    return class_names, FAMILIES, batches()


class ExternalQueries:
    def __init__(self, source, transform, retry):
        self.source = source
        self.transform = transform
        self.retry = retry

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        row, _, _ = self.source.samples[index]
        image = self.retry(lambda: self.source._read_image(row))
        return self.transform(image), index


def lida_batches(checkpoint, source, args):
    import torch
    from comparison.training.evaluate_lida_retrieval import fingerprint_image, restore_gallery
    from data.image_io import retry_image_read
    from lida.retrieval import class_scores, extract_features

    names = checkpoint.get('class_names', [])
    if len(names) != len(IAB_CLASSES) or set(names) != IAB_CLASSES or checkpoint.get('step') != 0:
        raise ValueError("Expected Igor's full IAB LIDA retrieval checkpoint (T0)")
    _, mapping = increfa_mapping(names, dataset='iab')
    encoder, gallery, fp = restore_gallery(checkpoint, args.gallery_root, names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = dict(device=device, batch_size=args.batch_size, num_workers=args.num_workers)

    def batches():
        encoder.to(device)
        gallery_features, gallery_labels = extract_features(encoder, gallery, desc='gallery', **run)
        if (len(gallery_features) != len(gallery) or len(gallery_labels) != len(gallery)
                or set(gallery_labels.tolist()) != set(range(len(names)))):
            raise ValueError('LIDA feature extraction did not cover the registered IAB gallery')
        queries = ExternalQueries(source, partial(fingerprint_image, **fp), retry_image_read)
        features, indices = extract_features(encoder, queries, desc='IABench', **run)
        if len(features) != len(queries) or indices.tolist() != list(range(len(queries))):
            raise ValueError('LIDA feature extraction changed IABench query order or coverage')
        for start in range(0, len(features), 256):
            block = slice(start, start + 256)
            # Native class scores aggregate cosine retrieval against the saved IAB gallery.
            scores, similarities = class_scores(features[block], gallery_features, gallery_labels, len(names))
            del similarities
            generators = [name for _, name, _ in source.samples[block]]
            yield generators, baseline_predictions(torch.as_tensor(scores), mapping, 'lida')

    return list(names), FAMILIES, batches()


def increfa_batches(checkpoint, source, args):
    from torch.utils.data import DataLoader
    from comparison.training.test_increfa_styled import checkpoint_classes, generator_logits, restore_model
    from data.image_io import retry_image_read

    names = checkpoint_classes(checkpoint)
    if set(names) != IAB_CLASSES:
        raise ValueError('Expected the static 22-class IAB IncreFA checkpoint')
    _, mapping = increfa_mapping(names, dataset='iab')
    device = checkpoint['config']['device']
    native_path = str(args.increfa_dir)
    if not (args.increfa_dir / 'train.py').is_file():
        raise ValueError('increfa-dir must contain the native train.py')

    def batches():
        # IncreFA resolves its own data, model and relative config paths from its checkout.
        cwd = Path.cwd()
        native_roots = {'data', 'models', 'train', 'logger'}
        existing = {name: module for name, module in sys.modules.copy().items()
                    if name.partition('.')[0] in native_roots}
        for name in existing:
            del sys.modules[name]
        sys.path.insert(0, native_path)
        try:
            os.chdir(native_path)
            model, transform, _ = restore_model(checkpoint)
            dataset = ExternalQueries(source, transform, retry_image_read)
            # Workers inherit the loaded IABench reader while IncreFA owns the data namespace.
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                shuffle=False, drop_last=False,
                                multiprocessing_context='fork' if args.num_workers else None)
            offset = 0
            for images, indices in loader:
                end = offset + len(indices)
                if indices.tolist() != list(range(offset, end)):
                    raise ValueError('IncreFA loader changed IABench query order')
                generators = [name for _, name, _ in source.samples[offset:end]]
                scores = generator_logits(model, images.to(device), len(names))
                yield generators, baseline_predictions(scores, mapping, 'increfa')
                offset = end
        finally:
            os.chdir(cwd)
            sys.path.remove(native_path)
            for name in list(sys.modules):
                if name.partition('.')[0] in native_roots:
                    del sys.modules[name]
            sys.modules.update(existing)

    return names, FAMILIES, batches()


def validate_hypclip_checkpoint(checkpoint):
    import torch

    if (checkpoint.get('dataset', 'iab') != 'iab' or checkpoint.get('hierarchy') != 'increfa'
            or checkpoint.get('loss', 'cone') != 'cone'):
        raise ValueError('Expected an IAB HypCLIP entailment-cone checkpoint with learned hierarchy')
    classes = checkpoint.get('class_names')
    if (not isinstance(classes, list) or classes[:1] != ['real']
            or len(classes) != len(IAB_CLASSES) or set(classes) != IAB_CLASSES):
        raise ValueError('HypCLIP checkpoint must use the 22-class IAB inventory with real first')
    families, expected = increfa_mapping(classes, dataset='iab')
    names, mapping = checkpoint.get('family_names'), checkpoint.get('family_of')
    if (not isinstance(names, list) or len(names) != len(families) or set(names) != set(families)
            or not isinstance(mapping, list) or len(mapping) != len(classes)
            or any(type(i) is not int or not 0 <= i < len(names) for i in mapping)
            or [names[i] for i in mapping] != [families[i] for i in expected]):
        raise ValueError('HypCLIP checkpoint family mapping does not match the IAB taxonomy')
    tangent = checkpoint.get('family_tangent')
    if (not isinstance(tangent, torch.Tensor) or not tangent.is_floating_point()
            or tangent.shape != (len(names), checkpoint.get('hyperbolic_dim', 128))
            or not torch.isfinite(tangent).all() or (tangent.norm(dim=1) == 0).any()):
        raise ValueError('HypCLIP checkpoint has invalid learned family anchors')
    curv = checkpoint.get('curv', 1.0)
    if not isinstance(curv, (int, float)) or not math.isfinite(curv) or curv <= 0:
        raise ValueError('HypCLIP checkpoint curvature must be positive and finite')
    return classes, names, tangent, curv


def hypclip_predictions(images, anchors, curv):
    from geometry.lorentz import pairwise_dist
    from losses.attribution_loss import _pairwise_xi

    images, anchors = images.double(), anchors.double()
    angles = _pairwise_xi(anchors, images, curv).T
    distances = pairwise_dist(images, anchors, curv)
    predictions = {}
    for rule, values in zip(HYPCLIP_RULES, (angles, distances)):
        if not values.isfinite().all():
            raise ValueError(f'HypCLIP: invalid family measurements for {rule}')
        predictions[rule] = values.argmin(dim=1).tolist()
    return predictions


def hypclip_batches(checkpoint, source, args):
    import torch
    from transformers import CLIPImageProcessor
    from geometry.lorentz import exp_map0
    from models.attribution_clip import AttributionCLIP

    classes, names, tangent, curv = validate_hypclip_checkpoint(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    anchors = exp_map0(tangent.float().to(device), curv=curv)
    model = AttributionCLIP.from_checkpoint(checkpoint).to(device)
    model.clip.load_state_dict(checkpoint['lora_state'])
    model.projection.load_state_dict(checkpoint['projection'])
    model.eval()
    source.processor = CLIPImageProcessor.from_pretrained(checkpoint['clip_name'])
    loader = torch.utils.data.DataLoader(source, batch_size=args.batch_size,
                                         num_workers=args.num_workers, shuffle=False, drop_last=False)

    def batches():
        for batch in loader:
            images, _ = model.encode_image(batch['pixel_values'].to(device))
            yield batch['generator'], hypclip_predictions(images, anchors, curv)

    return classes, names, batches()


def score_batches(samples, batches, family_names, rules):
    expected = [name for _, name, _ in samples]
    counts = {rule: Counter() for rule in rules}
    offset = 0
    for generators, predictions in batches:
        end = offset + len(generators)
        if list(generators) != expected[offset:end] or predictions.keys() != counts.keys():
            raise ValueError('Inference batches differ from the ordered IABench test cohort or readouts')
        for rule, indices in predictions.items():
            if len(indices) != len(generators) or any(not 0 <= i < len(family_names) for i in indices):
                raise ValueError(f'Invalid family predictions for {rule}')
            counts[rule].update(zip(generators, (family_names[i] for i in indices)))
        offset = end
    if offset != len(expected):
        raise ValueError('Inference did not cover every IABench test row')

    reports = {}
    for rule, counts_by_generator in counts.items():
        per_generator = {}
        totals = Counter(dict.fromkeys(FAMILIES, 0))
        for name in SCORED:
            assignments = {family: counts_by_generator[name, family] for family in FAMILIES}
            support = sum(assignments.values())
            per_generator[name] = dict(support=support, recall=assignments['diffusion'] / support,
                                       family_prediction_counts=assignments)
            totals.update(assignments)
        nano_counts = {family: counts_by_generator[EXPLORATORY, family] for family in FAMILIES}
        nano_support = sum(nano_counts.values())
        num_scored = sum(totals.values())
        reports[rule] = dict(
            accuracy=totals['diffusion'] / num_scored, num_scored_images=num_scored,
            per_generator=per_generator, family_prediction_counts=dict(totals),
            nano_banana=dict(support=nano_support, family_assignment_counts=nano_counts,
                             family_assignment_shares={k: n / nano_support for k, n in nano_counts.items()}),
        )
    return reports


def evaluate(args):
    import torch
    from data.hierarchy_experiment import write_json

    method = args.method
    path = args.checkpoint
    if path is None:
        mapping = load_checkpoint_map(args.checkpoints, method=method)
        path = Path(mapping[method])
    path = path.expanduser().resolve(strict=True)
    args.root_dir = args.root_dir.resolve()
    if method == 'lida' and args.gallery_root == args.root_dir:
        raise ValueError('LIDA requires separate IAB gallery and IABench query roots')
    source, cohort = load_cohort(args.root_dir, args.split_manifest)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    rules = HYPCLIP_RULES if method == 'hypclip' else BASELINE_RULES
    output = args.output_dir / method
    output.mkdir(parents=True, exist_ok=True)
    print(f'{method}: {cohort["num_scored_images"]} scored / {len(source)} total test images', flush=True)
    with torch.no_grad():
        external_batches = {'hypclip': hypclip_batches, 'lida': lida_batches, 'increfa': increfa_batches}
        if method in external_batches:
            classes, family_names, batches = external_batches[method](checkpoint, source, args)
        else:
            classes, family_names, batches = baseline_batches(method, checkpoint, source, args)
        epoch = checkpoint.get('epoch')
        num_epochs = checkpoint.get('num_epochs')
        del checkpoint
        metrics = score_batches(source.samples, batches, family_names, rules)
    report = dict(protocol=PROTOCOL, method=method, checkpoint=str(path), epoch=epoch,
                  num_epochs=num_epochs, class_names=classes, family_names=list(FAMILIES),
                  readout_family_order=list(family_names), cohort=cohort,
                  data_root=str(args.root_dir.resolve()), split_manifest=str(args.split_manifest.resolve()),
                  rules=metrics)
    if method == 'lida':
        report.update(implementation='igor_lida_retrieval', gallery_root=str(args.gallery_root))
    if method == 'increfa':
        report.update(implementation='static_iab_increfa', increfa_dir=str(args.increfa_dir))
    write_json(output / 'metrics.json', report, immutable=True)
    for rule, scores in metrics.items():
        print(f'{method} {rule}: {scores["accuracy"]:.2%}', flush=True)
    if all((args.output_dir / name / 'metrics.json').is_file() for name in METHODS):
        summarize(args.output_dir)


def summarize(output_dir):
    paths = {method: output_dir / method / 'metrics.json' for method in METHODS}
    missing = [method for method, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f'Missing family attribution reports for {", ".join(missing)}')
    rows = []
    cohort = None
    for method, path in paths.items():
        report = json.loads(path.read_text())
        rules = HYPCLIP_RULES if method == 'hypclip' else BASELINE_RULES
        if (report['protocol'] != PROTOCOL or report['method'] != method
                or report['family_names'] != list(FAMILIES) or report['rules'].keys() != set(rules)):
            raise ValueError(f'Incompatible family attribution report: {path}')
        if cohort is None:
            cohort = report['cohort']
        if not cohort.get('ordered_test_rows_digest') or report['cohort'] != cohort:
            raise ValueError(f'IABench test cohort mismatch in {path}')
        for rule in rules:
            scores = report['rules'][rule]
            support = scores['num_scored_images']
            accuracy = scores['accuracy']
            if support <= 0 or support != cohort['num_scored_images'] or not 0 <= accuracy <= 1:
                raise ValueError(f'Invalid accuracy or scored support in {path}')
            rows.append(dict(method=method, rule=rule, accuracy=accuracy, num_scored_images=support,
                             checkpoint=report['checkpoint'], epoch=report['epoch']))
    path = output_dir / 'summary.csv'
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', newline='', dir=output_dir, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Saved {len(rows)} model–rule rows to {path}')
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--method', choices=METHODS, required=True)
    checkpoints = evaluation.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument('--checkpoint', type=Path)
    checkpoints.add_argument('--checkpoints', type=Path, help='IAB baseline checkpoint map')
    evaluation.add_argument('--root-dir', type=Path, required=True)
    evaluation.add_argument('--split-manifest', type=Path, required=True)
    evaluation.add_argument('--output-dir', type=Path, required=True)
    evaluation.add_argument('--batch-size', type=int, default=8)
    evaluation.add_argument('--num-workers', type=int, default=8)
    evaluation.add_argument('--gallery-root', type=Path, help="IAB gallery root for Igor's LIDA")
    evaluation.add_argument('--increfa-dir', type=Path, help='Native static IAB IncreFA checkout')
    commands.add_parser('summarize').add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'evaluate':
        if args.batch_size < 1 or args.num_workers < 0:
            evaluation.error('batch-size must be positive and num-workers must be non-negative')
        if args.method in ('hypclip', 'lida', 'increfa') and args.checkpoint is None:
            evaluation.error(f'{args.method} requires --checkpoint')
        if args.method == 'lida':
            if args.gallery_root is None or not args.gallery_root.is_dir():
                evaluation.error('lida requires --gallery-root pointing to the IAB gallery')
            args.gallery_root = args.gallery_root.resolve()
        if args.method == 'increfa':
            if args.increfa_dir is None:
                evaluation.error('increfa requires --increfa-dir')
            args.increfa_dir = args.increfa_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'evaluate':
        evaluate(args)
    else:
        summarize(args.output_dir)


if __name__ == '__main__':
    main()
