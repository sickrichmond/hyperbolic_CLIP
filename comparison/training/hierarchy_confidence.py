"""Uncalibrated model confidence and family correctness for all frozen clean test rows."""
import argparse
import json
from pathlib import Path

import torch

from checkpoint_io import atomic_torch_save
from comparison.training.hierarchy_experiment import family_probabilities, load_model
from comparison.training.hierarchy_pair import ordering_probability
from data.hierarchy_experiment import ExperimentData, write_json
from geometry.lorentz import pairwise_dist


def prediction_points(model_distances, family_distances, labels, descriptor):
    names, models = descriptor['class_names'], descriptor['seen_classes']
    groups = descriptor['family_groups']
    if (labels.ndim != 1 or labels.dtype != torch.long or not len(labels)
            or labels.min() < 0 or labels.max() >= len(names)
            or model_distances.shape != (len(labels), len(models))
            or family_distances.shape != (len(labels), len(groups))):
        raise ValueError('Invalid confidence distances or generator labels')
    for distances in (model_distances, family_distances):
        if not torch.isfinite(distances).all() or (distances < 0).any():
            raise ValueError('Distances must be finite and nonnegative')
    parent = {name: i for i, group in enumerate(groups.values()) for name in group}
    targets = torch.tensor([parent.get(name, -1) for name in names], dtype=torch.long)[labels]
    model_labels = torch.tensor([names.index(name) for name in models], dtype=torch.long)
    seen = torch.tensor([name in models for name in names], dtype=torch.bool)[labels]
    if not seen.any() or seen.all():
        raise ValueError('Confidence diagnostic requires both seen and unseen test images')
    candidate = model_distances.argmin(1)
    scores = family_probabilities(model_distances).gather(1, candidate[:, None]).squeeze(1)
    family_prediction = family_distances.argmin(1)
    return dict(
        generator_indices=labels, is_seen=seen, confidence=scores,
        predicted_model_indices=candidate, model_correct=model_labels[candidate] == labels,
        target_family_indices=targets, predicted_family_indices=family_prediction,
        family_correct=(targets >= 0) & (family_prediction == targets),
    )


def confidence_statistics(points, descriptor):
    def summarize(mask):
        supported = mask & (points['target_family_indices'] >= 0)
        n, family_n = int(mask.sum()), int(supported.sum())
        if not n:
            raise ValueError('Every declared generator/cohort must have test samples')
        return dict(
            samples=n, model_correct=int(points['model_correct'][mask].sum()),
            model_accuracy=points['model_correct'][mask].double().mean().item(),
            family_target_samples=family_n,
            family_correct=int(points['family_correct'][supported].sum()),
            family_accuracy=points['family_correct'][supported].double().mean().item() if family_n else None,
            mean_confidence=points['confidence'][mask].mean().item(),
            median_confidence=points['confidence'][mask].median().item(),
        )
    seen = points['is_seen']
    return dict(
        seen=summarize(seen), unseen=summarize(~seen),
        seen_unseen_score_ordering_probability=1 - ordering_probability(
            points['confidence'][seen], points['confidence'][~seen]),
        per_generator={name: summarize(points['generator_indices'] == index)
                       for index, name in enumerate(descriptor['class_names'])},
    )


def plot_confidence(points, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    seen = points['is_seen'].numpy()
    confidence = points['confidence'].numpy()
    correct = points['model_correct'].numpy()
    has_target = points['target_family_indices'].numpy() >= 0
    family_correct = points['family_correct'].numpy()
    # Jitter changes display positions only; score and correctness tensors remain exact.
    rows = np.where(has_target, np.where(family_correct, 2., 1.), 0.)
    jitter = np.random.default_rng(42).uniform(-.12, .12, len(confidence))
    colors = np.where(correct, '#167c73', '#c75b3d')
    low, high = float(confidence.min()), float(confidence.max())
    padding = max((high - low) * .04, .0001)
    detail = (max(0., low - padding), min(1., high + padding))
    for suffix, limits in (('', (0., 1.)), ('_detail', detail)):
        fig = plt.figure(figsize=(13, 7.5))
        layout = fig.add_gridspec(2, 2, height_ratios=[1, 2.5], hspace=.08, wspace=.13)
        marginals = []
        for col, (mask, title, color) in enumerate(((seen, 'Seen generators', '#167c73'),
                                                   (~seen, 'Held-out generators', '#c75b3d'))):
            marginal = fig.add_subplot(layout[0, col])
            scatter = fig.add_subplot(layout[1, col], sharex=marginal)
            marginals.append(marginal)
            weights = np.full(int(mask.sum()), 100 / int(mask.sum()))
            marginal.hist(confidence[mask], bins=np.linspace(*limits, 51), weights=weights,
                          histtype='stepfilled', color=color, alpha=.25)
            marginal.set_title(f'{title} (n={int(mask.sum()):,})', fontsize=13)
            marginal.set_ylabel('Images per bin (%)' if col == 0 else '')
            marginal.tick_params(axis='x', labelbottom=False)
            marginal.spines[['top', 'right']].set_visible(False)
            scatter.scatter(confidence[mask], rows[mask] + jitter[mask], c=colors[mask],
                            s=5, alpha=.22, linewidths=0, rasterized=True)
            scatter.axhline(.5, color='#dddddd', linewidth=1)
            scatter.axhline(1.5, color='#dddddd', linewidth=1)
            scatter.set(xlim=limits, ylim=(-.35, 2.35), yticks=[0, 1, 2])
            scatter.ticklabel_format(axis='x', style='plain', useOffset=False)
            scatter.xaxis.set_major_locator(MaxNLocator(5))
            scatter.set_yticklabels(['No family target', 'Family incorrect', 'Family correct'] if col == 0 else [])
            scatter.spines[['top', 'right', 'left']].set_visible(False)
            scatter.tick_params(axis='y', length=0)
            uniform = 1 / len(points['model_anchor_names'])
            scatter.axvline(uniform, color='#777777', linewidth=.8, linestyle=':')
        top = max(axis.get_ylim()[1] for axis in marginals)
        for axis in marginals:
            axis.set_ylim(0, top)
        title = 'Specific-generator score and family attribution'
        if suffix:
            title += '\nObserved score range; numeric values unchanged'
        fig.suptitle(title, fontsize=16)
        fig.supxlabel('Maximum reciprocal-distance softmax score (uncalibrated)', y=.10)
        handles = [Line2D([], [], marker='o', linestyle='', color=color, label=label)
                   for color, label in (('#167c73', 'Specific name correct'), ('#c75b3d', 'Specific name incorrect'))]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .02), ncol=2, frameon=False)
        fig.text(.5, .005, 'All test images shown; unsmoothed histograms. Dotted reference, when in view: uniform score (1 / catalog size).',
                 ha='center', fontsize=9)
        fig.subplots_adjust(left=.13, right=.98, top=.88, bottom=.20)
        for extension in ('png', 'pdf'):
            fig.savefig(Path(output) / f'confidence_family{suffix}.{extension}', dpi=300)
        plt.close(fig)


@torch.no_grad()
def evaluate_confidence(data, checkpoint, output_dir, batch_size=8, workers=8):
    if batch_size < 1 or workers < 0:
        raise ValueError('Invalid batch size or worker count')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Use a fresh confidence output directory: {output}')
    desc = data.descriptor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, model, config, anchors, families = load_model(data, checkpoint, device)
    from transformers import CLIPImageProcessor
    from tqdm import tqdm
    source = data.view('test', CLIPImageProcessor.from_pretrained(config['clip_name']))
    loader = torch.utils.data.DataLoader(source, batch_size=batch_size, num_workers=workers,
                                         shuffle=False, drop_last=False)
    models_all, families_all, labels_all, sample_keys = [], [], [], []
    for batch in tqdm(loader, desc='All seen/held-out clean test images'):
        images, _ = model.encode_image(batch['pixel_values'].to(device))
        models_all.append(pairwise_dist(images.double(), anchors.double(), config['curv']).cpu())
        families_all.append(pairwise_dist(images.double(), families.double(), config['curv']).cpu())
        labels_all.append(torch.tensor([desc['class_names'].index(name) for name in batch['generator']]))
        sample_keys.extend(batch['sample_key'])
    if sample_keys != [key for key, _, _ in source.samples]:
        raise ValueError('Confidence diagnostic must retain every test row in order')
    models, families, labels = torch.cat(models_all), torch.cat(families_all), torch.cat(labels_all)
    points = prediction_points(models, families, labels, desc)
    provenance = dict(
        checkpoint=str(Path(checkpoint).resolve()), experiment_digest=data.digest,
        source_digests=desc['source_digests'], source_manifest_digests=desc['source_manifest_digests'],
        class_names=desc['class_names'], model_anchor_names=desc['seen_classes'],
        unseen_classes=desc['unseen_classes'], exploratory_classes=desc['exploratory_classes'],
        family_names=list(desc['family_groups']), family_groups=desc['family_groups'],
        model_score='max_softmax_reciprocal_geodesic_distance', inverse_distance_epsilon=1e-8,
        calibrated=False, family_readout='nearest_family_anchor', split='test', degradation='clean',
        test_scope='all_seen_unseen_and_exploratory',
    )
    saved = dict(provenance, **points, sample_keys=sample_keys, model_distances=models, family_distances=families)
    results = dict(provenance, **confidence_statistics(points, desc))
    output.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(saved, output / 'confidence_points.pt')
    write_json(output / 'confidence_results.json', results)
    plot_confidence(saved, output)
    summary = '\n'.join(f'{group}: {json.dumps(results[group], allow_nan=False)}' for group in ('seen', 'unseen'))
    (output / 'summary.txt').write_text(summary + '\n')
    print(summary)
    print(f'Points, metrics and plots: {output.resolve()}')
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--experiment')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--points', help='Replot saved confidence_points.pt without model or image loading')
    parser.add_argument('--iabench_root')
    parser.add_argument('--iab_root')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.points:
        saved = torch.load(args.points, map_location='cpu', weights_only=True)
        desc = dict(class_names=saved['class_names'], seen_classes=saved['model_anchor_names'],
                    family_groups=saved['family_groups'])
        points = prediction_points(saved['model_distances'], saved['family_distances'], saved['generator_indices'], desc)
        for key, value in points.items():
            match = (torch.allclose(saved[key], value, rtol=1e-10, atol=1e-12)
                     if value.is_floating_point() else torch.equal(saved[key], value))
            if not match:
                raise ValueError('Saved confidence points do not match the stored distances and labels')
        output = Path(args.output_dir)
        if output.exists() and any(output.iterdir()):
            raise ValueError(f'Use a fresh confidence output directory: {output}')
        output.mkdir(parents=True, exist_ok=True)
        plot_confidence(saved, output)
        return
    if not args.checkpoint or not args.experiment:
        parser.error('Inference requires --checkpoint and --experiment; use --points to replot')
    data = ExperimentData(args.experiment, args.iabench_root, args.iab_root)
    evaluate_confidence(data, args.checkpoint, args.output_dir, args.batch_size, args.num_workers)


if __name__ == '__main__':
    main()
