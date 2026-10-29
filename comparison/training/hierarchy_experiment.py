"""Prepare, train and evaluate the fixed 20/10/2 hybrid hierarchy experiment."""
import argparse
import json
import math
from pathlib import Path

import torch

from data.degradations import LEVEL_LABELS
from data.hierarchy_experiment import (
    ExperimentData, EXPLORATORY, METHODS, prepare_experiment, write_json,
)
from training.families import increfa_mapping

REPO = Path(__file__).resolve().parents[2]
RECIPES = dict(ours=(30, 256), resnet50=(10, 32), hifi_net=(10, 8))
LABELS = dict(ours='ours (hierarchy)', resnet50='ResNet50', hifi_net='HiFi-Net')


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
    if method == 'ours':
        from transformers import CLIPImageProcessor
        from train_attribution import run_training
        args = ours_args(data, output, workers)
        processor = CLIPImageProcessor.from_pretrained(args.clip_name)
        run_training(args, datasets=(data.view('train', processor), data.view('val', processor)),
                     checkpoint_metadata=metadata, plot_each_epoch=False,
                     family_groups=data.descriptor['family_groups'])
        return Path(args.output)

    import yaml
    from comparison.training.train import init_seed, choose_optimizer, choose_scheduler
    from comparison.training.attributors import ATTRIBUTOR
    from comparison.training.iabench import get_iabench_dataloaders
    from comparison.training.logger import create_logger
    from comparison.training.trainer.trainer import Trainer
    from comparison.dataset.ImageAttributionDataset.dataset import hifi_label_mapping
    init_seed(42, torch.cuda.is_available())
    config = yaml.safe_load((REPO / f'comparison/training/config/model/{method}.yaml').read_text())
    config.update(dataset='iab_hybrid', manualSeed=42, class_names=metadata['class_names'],
                  num_classes=20, specific_task_number=20, real_class_index=0, train_augment=False,
                  log_dir=str(output))
    if torch.cuda.is_available() and config.get('cudnn', True):
        torch.backends.cudnn.benchmark = True
    if method == 'hifi_net':
        config.update(hierarchy_mapping=hifi_label_mapping(metadata['class_names'], 'hybrid'),
                      hierarchy_sizes=[2, 4, 7, 20])
    metadata['training_config'] = dict(config)
    config['checkpoint_metadata'] = metadata
    model = ATTRIBUTOR[method](config)
    loaders = get_iabench_dataloaders(data.view('train'), data.view('val'), config,
                                     RECIPES[method][1], workers)
    optimizer = choose_optimizer(model, config)
    trainer = Trainer(config, model, optimizer, choose_scheduler(config, optimizer),
                      create_logger(str(output / 'training.log')), 'acc', log_dir=str(output))
    best, best_epoch = -math.inf, 0
    try:
        for epoch in range(1, RECIPES[method][0] + 1):
            value, _ = trainer.train_epoch(epoch, loaders[0], val_loader={'test': loaders[1]})
            if value is None or not math.isfinite(value):
                raise ValueError('Seen validation accuracy must be finite')
            if value > best:
                best, best_epoch = value, epoch
                trainer.save_checkpoint('ckpt_best.pth', dict(val_metric=best, best_epoch=best_epoch), epoch)
            trainer.save_checkpoint('ckpt_last.pth', dict(val_metric=best, best_epoch=best_epoch), epoch)
    finally:
        for writer in trainer.writers.values():
            writer.close()
    return output / 'ckpt_best.pth'


def family_probabilities(logits, family_of, count):
    mapping = torch.as_tensor(family_of, dtype=torch.long, device=logits.device)
    if (logits.ndim != 2 or logits.shape[1] != len(mapping) or not torch.isfinite(logits).all()
            or not len(mapping) or mapping.min() < 0 or mapping.max() >= count):
        raise ValueError('Invalid seen-model logits or family mapping')
    return logits.new_zeros((len(logits), count)).scatter_add_(
        1, mapping.expand(len(logits), -1), logits.softmax(1))


def prediction_metrics(logits, family_scores, labels, data):
    desc = data.descriptor
    names, seen, unseen = desc['class_names'], desc['seen_classes'], desc['unseen_classes']
    families = list(desc['family_groups'])
    parents = {name: f for f, group in desc['family_groups'].items() for name in group}
    if (labels.ndim != 1 or labels.dtype != torch.long or not len(labels)
            or logits.shape != (len(labels), 20) or family_scores.shape != (len(labels), 4)
            or labels.min() < 0 or labels.max() >= 32
            or not torch.isfinite(logits).all() or not torch.isfinite(family_scores).all()):
        raise ValueError('Invalid predictions or 32-class test labels')
    cp, fp = logits.argmax(1), family_scores.argmax(1)
    seen_conf = torch.zeros(20, 20, dtype=torch.long)
    generator_family = torch.zeros(32, 4, dtype=torch.long)
    family_conf = torch.zeros(4, 4, dtype=torch.long)
    per_seen, per_unseen, exploratory = {}, {}, {}
    for index, name in enumerate(names):
        mask = labels == index
        support = int(mask.sum())
        if not support:
            raise ValueError(f'No test samples for {name}')
        counts = torch.bincount(fp[mask], minlength=4)
        generator_family[index] = counts
        source = 'iab' if name in ('grok3', 'janus-pro', 'infinity') else 'iabench'
        if name in seen:
            row = seen.index(name)
            seen_conf[row] = torch.bincount(cp[mask], minlength=20)
            per_seen[name] = dict(recall=seen_conf[row, row].item() / support, support=support, source=source)
        elif name in unseen:
            parent = families.index(parents[name])
            family_conf[parent] += counts
            per_unseen[name] = dict(recall=counts[parent].item() / support, support=support,
                                    source=source, family=families[parent])
        else:
            exploratory[name] = dict(source=source, support=support,
                family_assignment_counts=dict(zip(families, counts.tolist())),
                family_assignment_proportions=dict(zip(families, (counts.double() / support).tolist())))
    represented = family_conf.sum(1) > 0
    def average(group):
        return sum(per_unseen[name]['recall'] for name in group) / len(group)
    unseen_lines = desc['unseen_lines']
    metrics = dict(
        seen_model_accuracy=seen_conf.diag().sum().item() / seen_conf.sum().item(),
        seen_model_balanced_accuracy=sum(v['recall'] for v in per_seen.values()) / 20,
        unseen_family_accuracy=family_conf.diag().sum().item() / family_conf.sum().item(),
        unseen_family_balanced_accuracy=(family_conf.diag()[represented].double() /
                                        family_conf.sum(1)[represented]).mean().item(),
        unseen_family_macro_generator_recall=average(unseen),
        diffusion_nearby_macro_recall=average(desc['nearby_diffusion']),
        diffusion_unseen_line_macro_generator_recall=average([n for group in unseen_lines.values() for n in group]),
        diffusion_unseen_line_macro_line_recall=sum(average(group) for group in unseen_lines.values()) / len(unseen_lines),
        unseen_per_line={line: dict(recall=average(group), generators=group) for line, group in unseen_lines.items()},
        seen_per_generator=per_seen, unseen_per_generator=per_unseen, exploratory=exploratory,
        unseen_per_family={families[i]: dict(recall=family_conf[i, i].item() / family_conf[i].sum().item(),
                                             support=family_conf[i].sum().item())
                           for i in range(4) if represented[i]},
        seen_confusion_matrix=seen_conf.tolist(), unseen_family_confusion_matrix=family_conf.tolist(),
        generator_family_confusion_matrix=generator_family.tolist(),
    )
    metrics['per_source'] = {}
    for source in ('iab', 'iabench'):
        result = {}
        for prefix, group in (('seen_model', per_seen), ('unseen_family', per_unseen)):
            values = [v for v in group.values() if v['source'] == source]
            support = sum(v['support'] for v in values)
            result[prefix + '_accuracy'] = sum(v['recall'] * v['support'] for v in values) / support
            result[prefix + '_support'] = support
            result[prefix + '_macro_generator_recall'] = sum(v['recall'] for v in values) / len(values)
        metrics['per_source'][source] = result
    return metrics


def load_model(data, checkpoint, device):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    method = ckpt.get('model_name')
    if method not in METHODS:
        raise ValueError('Checkpoint must be from one of the three experiment methods')
    for key, value in checkpoint_metadata(data).items():
        if ckpt.get(key) != value:
            raise ValueError(f'Incompatible experiment checkpoint {key}')
    if method == 'ours':
        from geometry.lorentz import exp_map0
        from models.attribution_clip import AttributionCLIP
        from comparison.training.test_hypclip import load_anchors
        for key, value in dict(hierarchy='increfa', anchor_penalty='geodesic', anchor_reduction='max',
                               anchor_scope='families', lambda_anchor=.5, target_norm=4, geodesic_margin=2).items():
            if ckpt.get(key) != value:
                raise ValueError(f'Incompatible hierarchy checkpoint {key}')
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
    from comparison.training.attributors import ATTRIBUTOR
    from comparison.dataset.ImageAttributionDataset.dataset import hifi_label_mapping
    config = ckpt.get('training_config', {})
    expected = dict(model_name=method, dataset='iab_hybrid', class_names=data.descriptor['seen_classes'],
                    num_classes=20, train_augment=False)
    if method == 'hifi_net':
        expected.update(hierarchy_mapping=hifi_label_mapping(expected['class_names'], 'hybrid'),
                        hierarchy_sizes=[2, 4, 7, 20])
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Baseline configuration does not match the training label space')
    model = ATTRIBUTOR[method](config).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    return method, model.eval(), config, None, None


@torch.no_grad()
def evaluate_experiment(data, checkpoint, output_dir, batch_size=8, workers=8):
    from tqdm import tqdm
    if batch_size < 1 or workers < 0:
        raise ValueError('Invalid batch size or worker count')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method, model, config, anchors, families = load_model(data, checkpoint, device)
    source = data.view('test')
    desc = data.descriptor
    family_of = increfa_mapping(desc['seen_classes'], desc['family_groups'])[1]
    if method == 'ours':
        from transformers import CLIPImageProcessor
        source.processor = CLIPImageProcessor.from_pretrained(config['clip_name'])
    else:
        from comparison.dataset.ImageAttributionDataset import DATASET
        from comparison.dataset.ImageAttributionDataset.dataset import hifi_label_mapping
        config = dict(config)
        if method == 'hifi_net':
            config['hierarchy_mapping'] = hifi_label_mapping(desc['class_names'], 'hybrid')
    output = Path(output_dir)
    provenance = dict(method=method, method_label=LABELS[method], checkpoint=str(Path(checkpoint).resolve()),
                      experiment_digest=data.digest, family_names=list(desc['family_groups']),
                      class_names=desc['class_names'], seen_classes=desc['seen_classes'],
                      unseen_classes=desc['unseen_classes'], exploratory_classes=desc['exploratory_classes'],
                      nearby_diffusion=desc['nearby_diffusion'], unseen_lines=desc['unseen_lines'],
                      source_digests=desc['source_digests'], source_manifest_digests=desc['source_manifest_digests'],
                      family_readout='minimum_exterior_angle' if method == 'ours' else 'summed_model_probabilities',
                      routing=desc['routing'], train_augment=False, test_samples=len(source))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'evaluation.json', provenance, immutable=True)
    for level in range(7):
        if method == 'ours':
            source.degraded = level
            dataset = source
        else:
            dataset = DATASET[method]('', config=config, iabench_source=source, degraded=level)
            dataset.set_test()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                             shuffle=False, drop_last=False)
        logits_all, families_all, labels_all = [], [], []
        for batch in tqdm(loader, desc=f'{LABELS[method]} {LEVEL_LABELS[level]}'):
            if method == 'ours':
                from losses.attribution_loss import _pairwise_xi
                images, _ = model.encode_image(batch['pixel_values'].to(device))
                logits = -_pairwise_xi(anchors, images, config['curv']).T
                family_scores = -_pairwise_xi(families, images, config['curv']).T
                labels = torch.tensor([desc['class_names'].index(name) for name in batch['generator']])
            else:
                logits = model(dict(image=batch['image'].to(device)), inference=True)['logits']
                family_scores = family_probabilities(logits, family_of, 4)
                labels = batch['label']
            logits_all.append(logits.cpu())
            families_all.append(family_scores.cpu())
            labels_all.append(labels.cpu())
        if sum(len(v) for v in labels_all) != len(source):
            raise ValueError('Evaluation must retain every test row')
        result = dict(provenance, level=level, degradation=LEVEL_LABELS[level],
                      **prediction_metrics(torch.cat(logits_all), torch.cat(families_all), torch.cat(labels_all), data))
        write_json(output / f'test_results_degraded_{level}.json', result)
        (output / f'test_results_degraded_{level}.txt').write_text(
            '\n'.join(f'{key}: {json.dumps(value, allow_nan=False)}' for key, value in result.items()) + '\n')
        print(f"{LEVEL_LABELS[level]}: seen BA={result['seen_model_balanced_accuracy']:.3%}, "
              f"unseen family BA={result['unseen_family_balanced_accuracy']:.3%}", flush=True)


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
            key = (result['method'], result['level'])
            if key in records:
                raise ValueError(f'Duplicate method/level {key}')
            records[key] = result
    metrics = ('seen_model_balanced_accuracy', 'unseen_family_balanced_accuracy',
               'unseen_family_macro_generator_recall', 'diffusion_nearby_macro_recall',
               'diffusion_unseen_line_macro_generator_recall', 'diffusion_unseen_line_macro_line_recall')
    for metric in metrics:
        print('\n' + metric + ' (%)')
        print(f"{'Method':<18}" + ''.join(f'{LEVEL_LABELS[i]:>10}' for i in range(7)))
        for method in METHODS:
            if any(m == method for m, _ in records):
                print(f'{LABELS[method]:<18}' + ''.join(
                    f'{records[method, i][metric] * 100:10.2f}' if (method, i) in records else f"{'missing':>10}"
                    for i in range(7)))
    for name in EXPLORATORY:
        print(f'\n{name}: predicted family share (%) — no accuracy target')
        print(f"{'Method / predicted family':<34}" + ''.join(f'{LEVEL_LABELS[i]:>10}' for i in range(7)))
        for method in METHODS:
            available = [result for (m, _), result in records.items() if m == method]
            if not available:
                continue
            for family in available[0]['family_names']:
                print(f'{LABELS[method] + " / " + family:<34}' + ''.join(
                    f"{records[method, i]['exploratory'][name]['family_assignment_proportions'][family] * 100:10.2f}"
                    if (method, i) in records else f"{'missing':>10}" for i in range(7)))


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
