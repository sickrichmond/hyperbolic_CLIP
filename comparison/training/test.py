import os
import argparse
import datetime
import json
import yaml
import torch
from torch.backends import cudnn
from comparison.training.trainer.trainer import Trainer
from comparison.training.attributors import ATTRIBUTOR
from comparison.dataset.ImageAttributionDataset.dataloader import get_dataloader
from comparison.dataset.ImageAttributionDataset.semantic_split import get_semantic
from comparison.training.logger import create_logger
from data.degradations import LEVEL_LABELS

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/model/resnet50.yaml', help='path to attributor YAML config file')
parser.add_argument('--use_semantic_split', action='store_true', default=False, help='whether to use semantic dataloader')
parser.add_argument('--root_dir', type=str, default='/remote-home/share/gzy/attribution/final_dataset_thats_real', help='root directory for dataset')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--num_images_per_semantic_per_class', '-n', type=int, default=2000)
parser.add_argument('--task_id', type=int, default=1, help='Task ID, used to select different semantic splits, such as 1, 2, 3, etc.')
parser.add_argument('--resume_checkpoint', type=str, required=True, help='checkpoint path for testing')
parser.add_argument('--log_dir', type=str, default="./logs_test")
parser.add_argument('--level_start', type=int, default=0)
parser.add_argument('--level_end', type=int, default=1)
parser.add_argument('--dataset', choices=('iab', 'iabench'), default='iab')
parser.add_argument('--split_manifest', help='Existing IABench training split manifest')


def init_seed(seed, use_cuda=True):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def create_test_logger(config, args, use_semantic_split=False):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    sub_folder = f"semantic_split_{args.task_id}" if use_semantic_split else "default_split"
    log_dir = os.path.join(config['log_dir'], sub_folder, config['model_name'], "test_" + now)
    if args.dataset == 'iabench':
        log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(os.path.join(log_dir, 'test.log'))
    logger.info(f"Testing logs saved to {log_dir}")
    return logger, log_dir


def choose_metric(config):
    metric = config.get('metric_scoring', 'auc')
    if metric not in ['auc', 'acc', 'ap']:
        raise NotImplementedError(f"Metric {metric} not implemented")
    return metric


def main(argv=None):
    args = parser.parse_args(argv)
    if not 0 <= args.level_start < args.level_end <= 7:
        raise ValueError("Evaluation levels must satisfy 0 <= level_start < level_end <= 7")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    test_source = None
    if args.dataset == 'iabench':
        from comparison.training.iabench import prepare_iabench_test, get_iabench_test_loader
        if args.use_semantic_split or args.num_images_per_semantic_per_class != 2000:
            raise ValueError("IABench has no semantic split/cap; use the saved manifest")
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
        config, test_source = prepare_iabench_test(args.root_dir, args.split_manifest, checkpoint)
        print(f"Baseline: {config['model_name']} | Checkpoint: {args.resume_checkpoint}")
        print(f"Classes ({len(config['class_names'])}): {config['class_names']}")
        print(f"Test rows: {len(test_source)} | Manifest digest: {checkpoint['manifest_digest']}")
    else:
        if args.split_manifest:
            raise ValueError("--split_manifest is supported only for IABench")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    config['log_dir'] = args.log_dir

    use_cuda = torch.cuda.is_available()
    init_seed(config.get('manualSeed', 42), use_cuda)

    if use_cuda and config.get('cudnn', True):
        cudnn.benchmark = True

    logger, log_dir = create_test_logger(config, args, use_semantic_split=args.use_semantic_split)

    model_class = ATTRIBUTOR[config['model_name']]
    model = model_class(config)

    metric_scoring = choose_metric(config)
    trainer = Trainer(config, model, None, None, logger, metric_scoring, log_dir=log_dir, test_mode=True)

    if test_source is not None:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        provenance = {key: checkpoint[key] for key in (
            'model_name', 'dataset', 'dataset_digest', 'manifest_digest', 'class_names')}
        provenance.update(checkpoint=os.path.abspath(args.resume_checkpoint),
                          split_manifest=os.path.abspath(args.split_manifest), split='test',
                          test_rows=len(test_source), output_label_space=checkpoint['output_label_space'],
                          train_augment=config.get('train_augment', False),
                          aug_policy=config.get('aug_policy', 'none'))
        del checkpoint
        clip_preprocess = model.semantic_extractor.clip_preprocess if config['model_name'] == 'defl' else None
    else:
        trainer.load_checkpoint(args.resume_checkpoint)

    degraded_levels = list(range(args.level_start, args.level_end)) 

    if args.use_semantic_split:
        train_semantics, test_semantics = get_semantic(args.task_id)

    for degraded in degraded_levels:
        logger.info(f"Testing with degraded level {degraded} ...")

        if test_source is not None:
            test_loader = get_iabench_test_loader(
                test_source, config, args.batch_size, args.num_workers, degraded,
                clip_preprocess=clip_preprocess)
        elif args.use_semantic_split:
            train_loader, val_loader, test_loader = get_dataloader(
                root_dir=args.root_dir,
                model_name=config['model_name'],
                num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,
                train_semantics=train_semantics,
                test_semantics=test_semantics,
                batch_size=args.batch_size,
                degraded=degraded,
                config=config,
                num_workers=args.num_workers,
                use_semantic_split=True,
            )
        else:
            train_loader, val_loader, test_loader = get_dataloader(
                root_dir=args.root_dir,
                model_name=config['model_name'],
                num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,
                batch_size=args.batch_size,
                degraded=degraded,
                config=config,
                num_workers=args.num_workers,
            )

        test_metrics = trainer.test(test_loader)

        result_txt_path = os.path.join(log_dir, f"test_results_degraded_{degraded}.txt")
        with open(result_txt_path, 'w') as f:
            f.write(f"Test metrics for degraded={degraded} ({LEVEL_LABELS[degraded]}) ({datetime.datetime.now()}):\n")
            if test_source is not None:
                for key, value in provenance.items():
                    f.write(f"{key}: {json.dumps(value) if key == 'class_names' else value}\n")
            for metric_name, value in test_metrics.items():
                if test_source is not None and metric_name == 'semantic_acc':
                    continue
                if metric_name == "conf_matrix":
                    f.write(f"{metric_name}:\n{value}\n")
                else:
                    f.write(f"{metric_name}: {value}\n")
        logger.info(f"Saved test results for degraded={degraded} to {result_txt_path}")


if __name__ == '__main__':
    main()
