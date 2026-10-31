"""Prepare, train and evaluate the fixed 20/10/2 hybrid hierarchy experiment."""
import argparse
import json
from pathlib import Path

import torch

from checkpoint_io import atomic_torch_save
from geometry.lorentz import pairwise_dist
from data.degradations import LEVEL_LABELS
from data.hierarchy_experiment import (
    ExperimentData, METHODS, prepare_experiment, write_json,
)
from losses.attribution_loss import _pairwise_xi
from training.families import increfa_mapping

REPO = Path(__file__).resolve().parents[2]
RECIPES = dict(ours=(30, 256))
LABELS = dict(ours='ours (hierarchy)')
FAMILY_READOUTS = dict(geodesic='softmax_inverse_geodesic_distance', cone='minimum_exterior_angle')


def checkpoint_metadata(data):
    desc = data.descriptor
    families, family_of = increfa_mapping(desc['seen_classes'], desc['family_groups'])
    return dict(dataset='iab_hybrid', experiment_digest=data.digest,
                source_digests=desc['source_digests'], source_manifest_digests=desc['source_manifest_digests'],
                class_names=desc['seen_classes'], full_class_names=desc['class_names'],
                family_names=families, family_of=family_of, family_groups=desc['family_groups'],
                train_augment=False)


def ours_args(data, output, workers):
    from training.attribution_args import parse_args
    epochs, batch = RECIPES['ours']
    settings = dict(
        dataset_path=data.roots['iabench'], output=output / f'ours_anchor0p5_z2_ep{epochs}.pt',
        diag_plot_dir=output / 'diagnostics', clip_name='openai/clip-vit-large-patch14',
        hierarchy='increfa', anchor_init='random', anchor_init_norm=2, family_init_norm=1.2,
        family_depth_margin=.5, lambda_family=1, lambda_family_depth=.5,
        lora_target=r'vision_model\.encoder\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj)',
        lora_r=16, lora_alpha=32, hyperbolic_dim=128, curv=1, min_radius=.5, margin=.3,
        lambda_neg=1, lambda_norm=.5, target_norm=4, lambda_anchor=.5, geodesic_margin=2,
        init_depth=3, batch_size=batch, num_epochs=epochs, lr=3e-4, weight_decay=.01,
        optimizer='adamw', lr_schedule='cosine', num_workers=workers, seed=42, log_every=10,
    )
    argv = ['--generators', *data.descriptor['seen_classes']]
    for key, value in settings.items():
        argv.extend((f'--{key}', str(value)))
    return parse_args(argv, dataset='hybrid')


def train_experiment(data, method, output_dir, workers=8):
    if method not in METHODS or workers < 0:
        raise ValueError('Invalid method or worker count')
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Use a fresh training output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint_metadata(data), model_name=method, num_epochs=RECIPES[method][0])
    write_json(output / 'run.json', dict(method=method, experiment=str(data.path), roots=data.roots, **metadata))
    from transformers import CLIPImageProcessor
    from train_attribution import run_training
    args = ours_args(data, output, workers)
    processor = CLIPImageProcessor.from_pretrained(args.clip_name)
    run_training(args, datasets=(data.view('train', processor), data.view('val', processor)),
                 checkpoint_metadata=metadata, plot_each_epoch=False,
                 family_groups=data.descriptor['family_groups'])
    return Path(args.output)


def family_probabilities(distances):
    """Reciprocal geodesic-distance softmax; clamp coincident points to stay finite."""
    if (distances.ndim != 2 or not distances.numel() or not torch.isfinite(distances).all()
            or (distances < 0).any()):
        raise ValueError('Invalid family geodesic distances')
    return distances.double().clamp_min(1e-8).reciprocal().softmax(1)


def prediction_metrics(family_scores, labels, data):
    desc = data.descriptor
    names, unseen = desc['class_names'], desc['unseen_classes']
    families = list(desc['family_groups'])
    parents = {name: f for f, group in desc['family_groups'].items() for name in group}
    if (labels.ndim != 1 or labels.dtype != torch.long or not len(labels)
            or family_scores.shape != (len(labels), len(families))
            or labels.min() < 0 or labels.max() >= len(names)
            or not torch.isfinite(family_scores).all()):
        raise ValueError('Invalid family predictions or test labels')
    fp = family_scores.argmax(1)
    generator_family = torch.zeros(len(unseen), len(families), dtype=torch.long)
    family_conf = torch.zeros(len(families), len(families), dtype=torch.long)
    per_unseen = {}
    for row, name in enumerate(unseen):
        mask = labels == names.index(name)
        support = int(mask.sum())
        if not support:
            raise ValueError(f'No test samples for {name}')
        counts = torch.bincount(fp[mask], minlength=len(families))
        generator_family[row] = counts
        parent = families.index(parents[name])
        family_conf[parent] += counts
        per_unseen[name] = dict(recall=counts[parent].item() / support, support=support,
                               source='iab' if name in ('grok3', 'janus-pro', 'infinity') else 'iabench',
                               family=families[parent])
    represented = family_conf.sum(1) > 0
    def average(group):
        return sum(per_unseen[name]['recall'] for name in group) / len(group)
    unseen_lines = desc['unseen_lines']
    metrics = dict(
        unseen_family_accuracy=family_conf.diag().sum().item() / family_conf.sum().item(),
        unseen_family_balanced_accuracy=(family_conf.diag()[represented].double() /
                                        family_conf.sum(1)[represented]).mean().item(),
        unseen_family_macro_generator_recall=average(unseen),
        diffusion_nearby_macro_recall=average(desc['nearby_diffusion']),
        diffusion_unseen_line_macro_generator_recall=average([n for group in unseen_lines.values() for n in group]),
        diffusion_unseen_line_macro_line_recall=sum(average(group) for group in unseen_lines.values()) / len(unseen_lines),
        unseen_per_line={line: dict(recall=average(group), generators=group) for line, group in unseen_lines.items()},
        unseen_per_generator=per_unseen,
        unseen_per_family={families[i]: dict(recall=family_conf[i, i].item() / family_conf[i].sum().item(),
                                             support=family_conf[i].sum().item())
                           for i in range(len(families)) if represented[i]},
        unseen_family_confusion_matrix=family_conf.tolist(),
        unseen_generator_family_confusion_matrix=generator_family.tolist(),
    )
    metrics['per_source'] = {}
    for source in ('iab', 'iabench'):
        values = [v for v in per_unseen.values() if v['source'] == source]
        support = sum(v['support'] for v in values)
        metrics['per_source'][source] = dict(
            unseen_family_accuracy=sum(v['recall'] * v['support'] for v in values) / support,
            unseen_family_support=support,
            unseen_family_macro_generator_recall=sum(v['recall'] for v in values) / len(values))
    return metrics


def exploratory_assignments(family_scores, nearest_models, labels, data):
    """Unknown generators have assignment frequencies, without family targets."""
    desc = data.descriptor
    families, models = list(desc['family_groups']), desc['seen_classes']
    if (nearest_models.shape != labels.shape or nearest_models.dtype != torch.long
            or (nearest_models < 0).any() or (nearest_models >= len(models)).any()):
        raise ValueError('Invalid nearest model-anchor indices')
    predictions = family_scores.argmax(1)
    assignments = {}
    for name in desc['exploratory_classes']:
        mask = labels == desc['class_names'].index(name)
        support = int(mask.sum())
        if not support:
            raise ValueError(f'No test samples for {name}')
        family_counts = torch.bincount(predictions[mask], minlength=len(families))
        model_counts = torch.bincount(nearest_models[mask], minlength=len(models))
        most_frequent = model_counts.argmax().item()
        assignments[name] = dict(
            support=support,
            family_assignment_counts=dict(zip(families, family_counts.tolist())),
            family_assignment_proportions=dict(zip(families, (family_counts.double() / support).tolist())),
            nearest_model_counts=dict(zip(models, model_counts.tolist())),
            nearest_model_proportions=dict(zip(models, (model_counts.double() / support).tolist())),
            most_frequent_nearest_model=models[most_frequent],
            most_frequent_nearest_model_share=model_counts[most_frequent].item() / support,
        )
    return assignments


def load_model(data, checkpoint, device):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    method = ckpt.get('model_name')
    if method not in METHODS:
        raise ValueError('Checkpoint must be from our hierarchy experiment model')
    for key, value in checkpoint_metadata(data).items():
        if ckpt.get(key) != value:
            raise ValueError(f'Incompatible experiment checkpoint {key}')
    from geometry.lorentz import exp_map0
    from models.attribution_clip import AttributionCLIP
    from comparison.training.test_hypclip import load_anchors
    for key, value in dict(hierarchy='increfa', anchor_penalty='geodesic',
                           anchor_scope='families', lambda_anchor=.5, target_norm=4, geodesic_margin=2).items():
        if ckpt.get(key) != value:
            raise ValueError(f'Incompatible hierarchy checkpoint {key}')
    # Both reductions have the same inference rule; keep older runs evaluable.
    if ckpt.get('anchor_reduction') not in ('mean', 'max'):
        raise ValueError('Incompatible hierarchy checkpoint anchor_reduction')
    tangent, class_tangent = ckpt.get('family_tangent'), ckpt.get('anchor_tangent')
    if (not isinstance(tangent, torch.Tensor) or not isinstance(class_tangent, torch.Tensor)
            or class_tangent.ndim != 2 or class_tangent.shape[0] != 20
            or tangent.shape != (4, class_tangent.shape[1])
            or not torch.isfinite(tangent).all() or not torch.isfinite(class_tangent).all()):
        raise ValueError('Invalid learned class or family anchors')
    model = AttributionCLIP.from_checkpoint(ckpt).to(device)
    model.clip.load_state_dict(ckpt['lora_state'])
    model.projection.load_state_dict(ckpt['projection'])
    model.eval()
    with torch.no_grad():
        anchors = load_anchors(ckpt, model, ckpt['curv'], device, data.descriptor['seen_classes'])
        families = exp_map0(tangent.float().to(device), curv=ckpt['curv'])
    return method, model, ckpt, anchors, families


@torch.no_grad()
def evaluate_experiment(data, checkpoint, output_dir, batch_size=8, workers=8):
    from tqdm import tqdm
    if batch_size < 1 or workers < 0:
        raise ValueError('Invalid batch size or worker count')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method, model, config, anchors, families = load_model(data, checkpoint, device)
    source = data.view('test')
    desc = data.descriptor
    test_indices = [i for i, (_, name, _) in enumerate(source.samples)
                    if name in desc['unseen_classes'] + desc['exploratory_classes']]
    dataset = torch.utils.data.Subset(source, test_indices)
    test_keys = [source.samples[i][0] for i in test_indices]
    from transformers import CLIPImageProcessor
    source.processor = CLIPImageProcessor.from_pretrained(config['clip_name'])
    output = Path(output_dir)
    provenance = dict(method=method, method_label=LABELS[method], checkpoint=str(Path(checkpoint).resolve()),
                      experiment_digest=data.digest, family_names=list(desc['family_groups']),
                      class_names=desc['class_names'], seen_classes=desc['seen_classes'],
                      unseen_classes=desc['unseen_classes'], exploratory_classes=desc['exploratory_classes'],
                      nearby_diffusion=desc['nearby_diffusion'], unseen_lines=desc['unseen_lines'],
                      source_digests=desc['source_digests'], source_manifest_digests=desc['source_manifest_digests'],
                      family_readout=FAMILY_READOUTS['geodesic'], inverse_distance_epsilon=1e-8,
                      compared_family_readouts=FAMILY_READOUTS,
                      model_anchor_readout='geodesic_distance', metrics_scope='scored_unseen',
                      routing=desc['routing'], train_augment=False, test_samples=len(dataset),
                      test_scope='unseen_and_exploratory')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'evaluation.json', provenance, immutable=True)
    target_families = {name: i for i, group in enumerate(desc['family_groups'].values()) for name in group}
    for level in range(7):
        source.degraded = level
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                             shuffle=False, drop_last=False)
        distances_all, families_all, labels_all, top_distances, top_indices, sample_keys = [], [], [], [], [], []
        cone_angles_all = []
        for batch in tqdm(loader, desc=f'{LABELS[method]} {LEVEL_LABELS[level]}'):
            images, _ = model.encode_image(batch['pixel_values'].to(device))
            # Float64 avoids cancellation near coincident Lorentz points.
            distances = pairwise_dist(images.double(), families.double(), config['curv'])
            probabilities = family_probabilities(distances)
            cone_angles = _pairwise_xi(families.double(), images.double(), config['curv']).T
            if not torch.isfinite(cone_angles).all():
                raise ValueError('Invalid family cone angles')
            model_distances = pairwise_dist(images.double(), anchors.double(), config['curv'])
            if not torch.isfinite(model_distances).all():
                raise ValueError('Invalid model-anchor geodesic distances')
            nearest = model_distances.topk(3, dim=1, largest=False, sorted=True)
            labels_all.append(torch.tensor([desc['class_names'].index(name) for name in batch['generator']]))
            distances_all.append(distances.cpu())
            families_all.append(probabilities.cpu())
            cone_angles_all.append(cone_angles.cpu())
            top_distances.append(nearest.values.cpu())
            top_indices.append(nearest.indices.cpu())
            sample_keys.extend(batch['sample_key'])
        if sample_keys != test_keys:
            raise ValueError('Evaluation must retain every unseen and exploratory test row in order')
        labels, probabilities = torch.cat(labels_all), torch.cat(families_all)
        nearest_models = torch.cat(top_indices)
        cone_angles = torch.cat(cone_angles_all)
        cone_probabilities = (-cone_angles).softmax(1)
        metrics = prediction_metrics(probabilities, labels, data)
        exploratory = exploratory_assignments(probabilities, nearest_models[:, 0], labels, data)
        cone_results = dict(
            family_readout=FAMILY_READOUTS['cone'],
            exploratory=exploratory_assignments(cone_probabilities, nearest_models[:, 0], labels, data),
            **prediction_metrics(cone_probabilities, labels, data),
        )
        diagnostics_path = output / f'sample_diagnostics_degraded_{level}.pt'
        atomic_torch_save(dict(
            provenance, level=level, degradation=LEVEL_LABELS[level], sample_keys=sample_keys,
            generator_indices=labels,
            target_family_indices=torch.tensor([target_families.get(desc['class_names'][i], -1) for i in labels.tolist()]),
            predicted_family_indices=probabilities.argmax(1),
            family_distances=torch.cat(distances_all), family_probabilities=probabilities,
            cone_family_angles=cone_angles, cone_family_probabilities=cone_probabilities,
            cone_predicted_family_indices=cone_angles.argmin(1),
            model_anchor_names=desc['seen_classes'],
            model_anchor_family_indices=torch.tensor(increfa_mapping(desc['seen_classes'], desc['family_groups'])[1]),
            top3_model_indices=nearest_models, top3_model_distances=torch.cat(top_distances),
        ), diagnostics_path)
        result = dict(provenance, level=level, degradation=LEVEL_LABELS[level],
                      sample_diagnostics=diagnostics_path.name,
                      exploratory=exploratory, cone=cone_results, **metrics)
        write_json(output / f'test_results_degraded_{level}.json', result)
        (output / f'test_results_degraded_{level}.txt').write_text(
            '\n'.join(f'{key}: {json.dumps(value, allow_nan=False)}' for key, value in result.items()) + '\n')
        for rule, scores in (('geodesic', result), ('cone', cone_results)):
            print(f"{LEVEL_LABELS[level]} {rule}: unseen family accuracy={scores['unseen_family_accuracy']:.3%}, "
                  f"balanced accuracy={scores['unseen_family_balanced_accuracy']:.3%}", flush=True)


def summarize(paths):
    records, digest = {}, None
    for item in paths:
        path = Path(item)
        files = sorted(path.glob('test_results_degraded_*.json')) if path.is_dir() else [path]
        if not files:
            raise ValueError(f'No result files in {path}')
        for file in files:
            result = json.loads(file.read_text())
            if digest is not None and result['experiment_digest'] != digest:
                raise ValueError('Cannot compare different experiments')
            digest = result['experiment_digest']
            if (result.get('method') != 'ours'
                    or result.get('family_readout') != FAMILY_READOUTS['geodesic']
                    or result.get('metrics_scope') != 'scored_unseen'
                    or result.get('test_scope') != 'unseen_and_exploratory'):
                raise ValueError('Expected unseen-only geodesic results from our hierarchy model')
            key = ('geodesic', result['level'])
            if key in records:
                raise ValueError(f'Duplicate method/level {key}')
            records[key] = result
            if 'cone' in result:
                if result['cone'].get('family_readout') != FAMILY_READOUTS['cone']:
                    raise ValueError('Expected minimum-exterior-angle cone results')
                records['cone', result['level']] = result['cone']
    metrics = ('unseen_family_accuracy', 'unseen_family_balanced_accuracy',
               'unseen_family_macro_generator_recall', 'diffusion_nearby_macro_recall',
               'diffusion_unseen_line_macro_generator_recall', 'diffusion_unseen_line_macro_line_recall')
    print(f"{LABELS['ours']}: family decision rules")
    for metric in metrics:
        print('\n' + metric + ' (%)')
        print(f"{'Rule':<18}" + ''.join(f'{LEVEL_LABELS[i]:>10}' for i in range(7)))
        for rule in FAMILY_READOUTS:
            print(f'{rule:<18}' + ''.join(
                f'{records[rule, i][metric] * 100:10.2f}' if (rule, i) in records else f"{'missing':>10}"
                for i in range(7)))
    for name in next(iter(records.values()))['exploratory_classes']:
        print(f'\n{name}: family assignments (no family accuracy target)')
        for rule in FAMILY_READOUTS:
            for level in range(7):
                result = records.get((rule, level))
                if result is None:
                    print(f'{rule} {LEVEL_LABELS[level]}: missing')
                    continue
                stats = result['exploratory'][name]
                shares = ', '.join(f'{family} {share:.1%}'
                                   for family, share in stats['family_assignment_proportions'].items())
                print(f"{rule} {LEVEL_LABELS[level]}: {shares}; most frequently nearest model: "
                      f"{stats['most_frequent_nearest_model']} ({stats['most_frequent_nearest_model_share']:.1%})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Validate and freeze both existing source manifests')
    for name in ('iabench_root', 'iabench_manifest', 'iab_root', 'iab_manifest', 'output_dir'):
        prep.add_argument('--' + name, required=True)
    train = commands.add_parser('train')
    evaluate = commands.add_parser('evaluate')
    for command in (train, evaluate):
        command.add_argument('--experiment', required=True)
        command.add_argument('--iabench_root', help='Override the stored IABench root')
        command.add_argument('--iab_root', help='Override the stored IAB root')
        command.add_argument('--output_dir', required=True)
        command.add_argument('--num_workers', type=int, default=8)
    train.add_argument('--method', choices=METHODS, required=True)
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--batch_size', type=int, default=8)
    summary = commands.add_parser('summarize')
    summary.add_argument('paths', nargs='+')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare_experiment(args.iabench_root, args.iabench_manifest, args.iab_root, args.iab_manifest, args.output_dir)
    elif args.command == 'summarize':
        summarize(args.paths)
    else:
        data = ExperimentData(args.experiment, args.iabench_root, args.iab_root)
        if args.command == 'train':
            checkpoint = train_experiment(data, args.method, args.output_dir, args.num_workers)
            print(f'Best checkpoint: {checkpoint}')
        else:
            evaluate_experiment(data, args.checkpoint, args.output_dir, args.batch_size, args.num_workers)


if __name__ == '__main__':
    main()
