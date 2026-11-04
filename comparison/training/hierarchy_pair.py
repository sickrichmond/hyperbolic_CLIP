"""Plot seen-anchor distances and paired family recovery on frozen clean test rows."""
import argparse
import json
import math
from pathlib import Path
import textwrap

import torch

from checkpoint_io import atomic_torch_save
from comparison.training.hierarchy_experiment import load_model
from data.hierarchy_experiment import ExperimentData, write_json
from geometry.lorentz import pairwise_dist

DEFAULT_SEEN = 'stabilityai@stable-diffusion-2'
DEFAULT_UNSEEN = 'stabilityai@stable-diffusion-2-1'


def ordering_probability(seen, unseen):
    """P(seen distance < unseen distance), with half credit for exact ties."""
    for values in (seen, unseen):
        if (values.ndim != 1 or not values.numel() or not torch.isfinite(values).all()
                or (values < 0).any()):
            raise ValueError('Distances must be nonempty finite nonnegative vectors')
    ordered = unseen.double().sort().values
    left = torch.searchsorted(ordered, seen.double(), right=False)
    right = torch.searchsorted(ordered, seen.double(), right=True)
    return ((len(ordered) - right).double() + .5 * (right - left)).mean().item() / len(ordered)


def pair_statistics(model_distances, family_distances, is_seen, model_families, seen_index, target_family):
    if (model_distances.ndim != 2 or family_distances.ndim != 2
            or not model_distances.numel() or not family_distances.numel()
            or len(model_distances) != len(family_distances)
            or is_seen.shape != (len(model_distances),) or is_seen.dtype != torch.bool
            or not is_seen.any() or is_seen.all()
            or model_families.shape != (model_distances.shape[1],) or model_families.dtype != torch.long
            or (model_families < 0).any() or (model_families >= family_distances.shape[1]).any()
            or not 0 <= seen_index < model_distances.shape[1]
            or not 0 <= target_family < family_distances.shape[1]):
        raise ValueError('Invalid pair distances, membership or anchor mapping')
    for distances in (model_distances, family_distances):
        if not torch.isfinite(distances).all() or (distances < 0).any():
            raise ValueError('Anchor distances must be finite and nonnegative')
    models = model_distances[~is_seen]
    families = family_distances[~is_seen]
    # Models come first, so exact model/family ties select the model consistently.
    overall = torch.cat((models, families), dim=1).argmin(1)
    family = families.argmin(1)
    model_wins = overall < models.shape[1]
    same_family_model = torch.zeros_like(model_wins)
    same_family_model[model_wins] = model_families[overall[model_wins]] == target_family
    explicit_family = overall == models.shape[1] + target_family
    reference_wins = overall == seen_index
    denominator = int(reference_wins.sum())
    recovered = int((reference_wins & (family == target_family)).sum())
    reference_distances = model_distances[:, seen_index]
    return dict(
        seen_samples=int(is_seen.sum()), unseen_samples=int((~is_seen).sum()),
        ordering_probability=ordering_probability(reference_distances[is_seen], reference_distances[~is_seen]),
        nearest_overall_counts=dict(
            explicit_target_family=int(explicit_family.sum()),
            wrong_generator_same_family=int(same_family_model.sum()),
            other_family=int((~explicit_family & ~same_family_model).sum())),
        nearest_family_counts=torch.bincount(family, minlength=families.shape[1]).tolist(),
        paired_recovery=dict(reference_assignments=denominator, correct_family=recovered,
                             fraction=recovered / denominator if denominator else None),
    )


def plot_distance(saved, result, output, xlim=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    seen_name, unseen_name = (result[key].split('@')[-1] for key in ('seen_generator', 'unseen_generator'))
    reference = saved['model_anchor_names'].index(result['seen_generator'])
    distances = saved['model_distances'][:, reference].numpy()
    mask = saved['is_seen'].numpy()
    green, orange = '#167c73', '#c75b3d'
    if xlim is not None and (len(xlim) != 2 or not all(math.isfinite(v) for v in xlim)
                             or not 0 <= xlim[0] < xlim[1]):
        raise ValueError('Distance limits must be finite with 0 <= minimum < maximum')
    low, high = float(distances.min()), float(distances.max())
    if low == high:
        low, high = max(0., low - .05), high + .05
    bins, grid = np.linspace(low, high, 41), np.linspace(low, high, 400)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for values, name, color, status in ((distances[mask], seen_name, green, 'seen'),
                                       (distances[~mask], unseen_name, orange, 'held out')):
        ax.hist(values, bins=bins, density=True, color=color, alpha=.25,
                label=textwrap.fill(f'{name} ({status}; n={len(values)})', 45))
        if xlim is None and len(values) > 1 and np.ptp(values) > 0:
            ax.plot(grid, gaussian_kde(values)(grid), color=color, linewidth=2)
    ax.set(title=textwrap.fill(f'How close are test images to the {seen_name} anchor?', 85),
           xlabel='Geodesic distance to the seen-generator anchor', ylabel='Density')
    ax.text(.98, .96, f"Ordering probability (half credit for ties): {result['ordering_probability']:.2%}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    stem = 'distance_to_seen_anchor'
    if xlim is not None:
        ax.set_xlim(*xlim)
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        visible = (distances >= xlim[0]) & (distances <= xlim[1])
        note = ('Histogram only: no smoothing; original bins and full-cohort normalization retained.\n'
                f'Original bin width: {bins[1] - bins[0]:.6f}; displayed window width: {xlim[1] - xlim[0]:.6f}.\n'
                f'Outside view: seen {int((mask & ~visible).sum())}/{int(mask.sum())}; '
                f'held out {int((~mask & ~visible).sum())}/{int((~mask).sum())}.')
        fig.text(.5, .015, note, ha='center', va='bottom', fontsize=9)
        stem += '_zoom'
    fig.tight_layout(rect=(0, .11 if xlim is not None else 0, 1, 1))
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=300)
    plt.close(fig)


def plot_pair(saved, result, output):
    plot_distance(saved, result, output)
    import matplotlib.pyplot as plt

    green, orange = '#167c73', '#c75b3d'
    seen_name, unseen_name = (result[key].split('@')[-1] for key in ('seen_generator', 'unseen_generator'))
    labels = dict(real='Real', gan='GAN', autoregressive='Autoregressive', diffusion='Diffusion')
    target = result['target_family']
    other_colors = iter(('#6b7280', '#527eb1', '#9270ad'))
    family_colors = {name: green if name == target else next(other_colors) for name in result['family_names']}
    target_label = labels[target]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)
    segments = [
        [(f'Explicit {target_label} family', result['nearest_overall_counts']['explicit_target_family'], green),
         (f'Wrong generator, {target_label} family', result['nearest_overall_counts']['wrong_generator_same_family'], orange),
         ('Other family', result['nearest_overall_counts']['other_family'], '#8b929b')],
        [(labels[name], count, family_colors[name])
         for name, count in zip(result['family_names'], result['nearest_family_counts'])],
    ]
    for ax, parts, title in zip(axes, segments, (
            f"Closest anchor overall\n{len(saved['model_anchor_names'])} model + {len(result['family_names'])} family anchors",
            f"Closest family anchor\n{len(result['family_names'])} family anchors")):
        bottom = 0.
        for label, count, color in parts:
            share = 100 * count / result['unseen_samples']
            ax.bar(0, share, bottom=bottom, width=.55, color=color,
                   label=f'{label}: {share:.1f}% ({count})')
            if share >= 5:
                ax.text(0, bottom + share / 2, f'{share:.1f}%', ha='center', va='center', color='white')
            bottom += share
        ax.set(title=title, ylim=(0, 100), xlim=(-.65, .65), xticks=[], yticks=range(0, 101, 20))
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.03), frameon=False, fontsize=9)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
    axes[0].set_ylabel('Held-out test images (%)')
    fig.suptitle(textwrap.fill(f'Family recovery for held-out {unseen_name}', 85), fontsize=15)
    paired = result['paired_recovery']
    rate = f"{paired['fraction']:.1%}" if paired['fraction'] is not None else 'N/A: no qualifying images'
    caption = (f"Among held-out images assigned to {seen_name} by the closest-anchor-overall rule, "
               f"the closest family is {target_label} for {paired['correct_family']}/"
               f"{paired['reference_assignments']} images ({rate}).")
    fig.text(.5, .025, textwrap.fill(caption, 120), ha='center', va='bottom', fontsize=9)
    fig.subplots_adjust(top=.80, bottom=.30, wspace=.30)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'family_recovery.{extension}', dpi=300)
    plt.close(fig)


@torch.no_grad()
def evaluate_pair(data, checkpoint, output_dir, seen=DEFAULT_SEEN, unseen=DEFAULT_UNSEEN, batch_size=8, workers=8):
    desc = data.descriptor
    if seen not in desc['seen_classes']:
        raise ValueError(f'Seen generator must have a trained anchor: {seen!r}')
    if unseen not in desc['unseen_classes']:
        raise ValueError(f'Unseen generator must be a scored holdout with a known family: {unseen!r}')
    if batch_size < 1 or workers < 0:
        raise ValueError('Invalid batch size or worker count')
    source = data.view('test')
    indices = [i for i, (_, name, _) in enumerate(source.samples) if name in (seen, unseen)]
    test_keys = [source.samples[i][0] for i in indices]
    if {source.samples[i][1] for i in indices} != {seen, unseen}:
        raise ValueError('Both generators require test rows in the frozen manifest')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Use a fresh diagnostic output directory: {output}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, model, config, anchors, families = load_model(data, checkpoint, device)
    from transformers import CLIPImageProcessor
    from tqdm import tqdm
    source.processor = CLIPImageProcessor.from_pretrained(config['clip_name'])
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(source, indices), batch_size=batch_size,
                                         num_workers=workers, shuffle=False, drop_last=False)
    model_distances, family_distances, membership, sample_keys = [], [], [], []
    for batch in tqdm(loader, desc=f'{seen} vs {unseen} (clean test)'):
        images, _ = model.encode_image(batch['pixel_values'].to(device))
        model_distances.append(pairwise_dist(images.double(), anchors.double(), config['curv']).cpu())
        family_distances.append(pairwise_dist(images.double(), families.double(), config['curv']).cpu())
        membership.extend(name == seen for name in batch['generator'])
        sample_keys.extend(batch['sample_key'])
    if sample_keys != test_keys:
        raise ValueError('Pair evaluation must retain every selected test row in order')
    model_distances, family_distances = torch.cat(model_distances), torch.cat(family_distances)
    is_seen = torch.tensor(membership, dtype=torch.bool)
    model_families = torch.tensor(config['family_of'], dtype=torch.long)
    target_family = next(i for i, group in enumerate(desc['family_groups'].values()) if unseen in group)
    statistics = pair_statistics(model_distances, family_distances, is_seen, model_families,
                                 desc['seen_classes'].index(seen), target_family)
    provenance = dict(
        checkpoint=str(Path(checkpoint).resolve()), experiment_digest=data.digest,
        source_digests=desc['source_digests'], source_manifest_digests=desc['source_manifest_digests'],
        seen_generator=seen, unseen_generator=unseen, family_names=list(desc['family_groups']),
        target_family=list(desc['family_groups'])[target_family], degradation='clean', split='test',
        distance='lorentz_geodesic', curvature=config['curv'],
        overall_readout='nearest_among_model_and_family_anchors',
        family_readout='nearest_family_anchor', exact_tie_rule='models_then_checkpoint_anchor_order',
        paired_condition='overall_winner_is_selected_seen_model',
    )
    saved = dict(provenance, sample_keys=sample_keys, is_seen=is_seen, model_anchor_names=desc['seen_classes'],
                 model_anchor_family_indices=model_families, model_distances=model_distances,
                 family_distances=family_distances)
    result = dict(provenance, **statistics)
    output.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(saved, output / 'pair_distances.pt')
    write_json(output / 'pair_results.json', result)
    paired = result['paired_recovery']
    rate = f"{paired['fraction']:.2%}" if paired['fraction'] is not None else 'N/A (no qualifying images)'
    summary = (f'Seen: {seen} ({result["seen_samples"]} test images)\n'
               f'Unseen: {unseen} ({result["unseen_samples"]} test images)\n'
               f'Ordering probability (half credit for ties): {result["ordering_probability"]:.2%}\n'
               f'Paired {result["target_family"]} recovery among overall {seen} assignments: '
               f'{paired["correct_family"]}/{paired["reference_assignments"]} = {rate}\n')
    (output / 'summary.txt').write_text(summary)
    plot_pair(saved, result, output)
    print(summary, end='')
    print(f'Results and plots: {output.resolve()}')
    return result


def replot_distance(results_dir, output_dir, xlim):
    results_dir, output = Path(results_dir), Path(output_dir)
    saved = torch.load(results_dir / 'pair_distances.pt', map_location='cpu', weights_only=True)
    result = json.loads((results_dir / 'pair_results.json').read_text())
    for key in ('experiment_digest', 'seen_generator', 'unseen_generator', 'family_names', 'target_family'):
        if saved.get(key) != result.get(key):
            raise ValueError(f'Saved distances and results disagree on {key}')
    statistics = pair_statistics(
        saved['model_distances'], saved['family_distances'], saved['is_seen'],
        saved['model_anchor_family_indices'], saved['model_anchor_names'].index(result['seen_generator']),
        result['family_names'].index(result['target_family']))
    if any(result.get(key) != value for key, value in statistics.items()):
        raise ValueError('Saved distances do not reproduce the saved statistics')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Use a fresh replot output directory: {output}')
    if len(xlim) != 2 or not all(math.isfinite(v) for v in xlim) or not 0 <= xlim[0] < xlim[1]:
        raise ValueError('Distance limits must be finite with 0 <= minimum < maximum')
    output.mkdir(parents=True, exist_ok=True)
    plot_distance(saved, result, output, xlim=xlim)
    write_json(output / 'zoom_view.json', dict(
        source_results=str(results_dir.resolve()), distance_limits=list(xlim),
        ordering_probability=result['ordering_probability'], histogram_normalization='full_cohort',
        histogram_bins='original_full_range_40', smoothing='none', experiment_digest=result['experiment_digest']))
    print(f'Zoomed distance plot (no model or images loaded): {output.resolve()}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seen', nargs='?', default=DEFAULT_SEEN, help='Exact trained generator name')
    parser.add_argument('unseen', nargs='?', default=DEFAULT_UNSEEN, help='Exact scored held-out generator name')
    parser.add_argument('--checkpoint')
    parser.add_argument('--experiment')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--replot', help='Directory containing saved pair_distances.pt and pair_results.json')
    parser.add_argument('--distance_xlim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='Zoom limits for --replot; full-data statistics remain unchanged')
    parser.add_argument('--iabench_root')
    parser.add_argument('--iab_root')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.replot:
        if args.distance_xlim is None:
            parser.error('--replot requires --distance_xlim MIN MAX')
        replot_distance(args.replot, args.output_dir, args.distance_xlim)
        return
    if args.distance_xlim is not None:
        parser.error('--distance_xlim requires --replot')
    if not args.checkpoint or not args.experiment:
        parser.error('Inference requires --checkpoint and --experiment')
    data = ExperimentData(args.experiment, args.iabench_root, args.iab_root)
    evaluate_pair(data, args.checkpoint, args.output_dir, args.seen, args.unseen, args.batch_size, args.num_workers)


if __name__ == '__main__':
    main()
