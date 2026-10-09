import os  
import argparse  
import datetime  
import yaml  
import torch  
from torch.backends import cudnn  
from comparison.training.trainer.trainer import Trainer  
from comparison.training.attributors import ATTRIBUTOR  
from comparison.training.metrics.utils import parse_metric_for_print  
from comparison.training.logger import create_logger  

from comparison.dataset.ImageAttributionDataset.dataloader import get_dataloader
from comparison.dataset.ImageAttributionDataset.semantic_split import get_semantic

parser = argparse.ArgumentParser()  
parser.add_argument('--config', type=str, default='config/model/resnet50.yaml',help='path to attributor YAML config file')  
parser.add_argument('--use_semantic_split', action='store_true', default=False, help='whether to use semantic dataloader')  
# parser.add_argument('--root_dir', type=str, default='/home/final_dataset', help='root directory for dataset')  
parser.add_argument('--root_dir', type=str, default='/remote-home/share/gzy/attribution/final_dataset_thats_real', help='root directory for dataset')  
parser.add_argument('--batch_size', type=int, default=32)  
parser.add_argument('--n_epoch', type=int)  
parser.add_argument('--num_workers', type=int, default=2)  
parser.add_argument('--num_images_per_semantic_per_class', '-n', type=int, default=2000)  
# parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)  
# parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)  
parser.add_argument('--task_target', type=str, default="")  
parser.add_argument('--resume_checkpoint', type=str, default=None)  
parser.add_argument('--log_dir', type=str, default="./logs")  
parser.add_argument('--task_id', type=int, default=1, help='Task ID, used to select different semantic splits, such as 1, 2, 3, etc.' )
parser.add_argument('--save_freq', type=int, default=5, help='Checkpoint saving frequency (epochs)')  
parser.add_argument('--do_test', action='store_false', default=True, help='Whether to run test evaluation after training')
parser.add_argument('--pretrained_path', type=str, default=None, help="Override config['pretrained_path'] (DNA stage 2 loads the stage-1 checkpoint)")
parser.add_argument("--dataset", choices=("iab", "iabench"), default="iab")
parser.add_argument("--split_manifest", help="Existing shared IABench split manifest")


def init_seed(seed, use_cuda=True):  
    import random  
    import numpy as np  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if use_cuda:  
        torch.cuda.manual_seed_all(seed)  


def create_training_logger(config, args, use_semantic_split=False):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  
    task_str = f"_{config['task_target']}" if config.get('task_target') else ""  
    base_folder = config['log_dir']  

    sub_folder = f"semantic_split_{args.task_id}" if use_semantic_split else "default_split"  
    log_dir = os.path.join(base_folder, sub_folder, config['model_name'], task_str + '_' + now)  
    if args.dataset == "iabench":
        log_dir = (os.path.dirname(os.path.abspath(args.resume_checkpoint)) if args.resume_checkpoint else
                   os.path.join(base_folder, "iabench", config["model_name"], task_str + "_" + now))
    os.makedirs(log_dir, exist_ok=True)  
    logger = create_logger(os.path.join(log_dir, 'training.log'))  
    logger.info(f"Training logs saved to {log_dir}")  
    return logger, log_dir, now   


def choose_optimizer(model, config):  
    opt_name = config['optimizer']['type']  
    if opt_name == 'adam':  
        optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer'][opt_name]['lr'])  
    elif opt_name == 'sgd':  
        optimizer = torch.optim.SGD(model.parameters(), lr=config['optimizer'][opt_name]['lr'], momentum=0.9)  
    else:  
        raise NotImplementedError(f"Optimizer {opt_name} not supported")  
    return optimizer  


def choose_scheduler(config, optimizer):  
    if config.get('lr_scheduler', None) == 'step':  
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])  
    elif config.get('lr_scheduler', None) == 'cosine':  
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_T_max'], eta_min=config['lr_eta_min'])  
    else:  
        return None  


def choose_metric(config):  
    metric = config.get('metric_scoring', 'auc')  
    if metric not in ['auc', 'acc', 'ap']:  
        raise NotImplementedError(f"Metric {metric} not implemented")  
    return metric  


def main(argv=None):
    args = parser.parse_args(argv)
    # loading config  
    with open(args.config, 'r') as f:  
        config = yaml.safe_load(f)  

    # config['save_ckpt'] = args.save_ckpt  
    # config['save_feat'] = args.save_feat  
    # config['task_target'] = args.task_target  
    config['log_dir'] = args.log_dir
    if args.pretrained_path:
        config['pretrained_path'] = args.pretrained_path

    if args.dataset == "iabench":
        from comparison.training.iabench import (
            prepare_iabench, get_iabench_dataloaders, validate_checkpoint_metadata,
            validate_dna_stage_one)
        if args.use_semantic_split:
            raise ValueError("IABench has no semantic split; use its shared manifest")
        train_source, val_source = prepare_iabench(
            args.root_dir, args.split_manifest, config, args.batch_size)
        if args.resume_checkpoint:
            checkpoint = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
            validate_checkpoint_metadata(checkpoint, config['checkpoint_metadata'])
            del checkpoint
        elif config['model_name'] == 'dna' and config['train_stage'] == 2:
            validate_dna_stage_one(config.get('pretrained_path'), config['checkpoint_metadata'])
        print(f"IABench classes ({len(config['class_names'])}): {config['class_names']}")
        print(f"Shared split counts: {config['checkpoint_metadata']['split_counts']}")
        print(f"Output label space: {config['checkpoint_metadata']['output_label_space']} ({config['num_classes']} classes)")

    use_cuda = torch.cuda.is_available()  
    init_seed(config.get('manualSeed', 42), use_cuda)  

    if use_cuda and config.get('cudnn', True):  
        cudnn.benchmark = True  

    # loading logger  
    logger, log_dir, now = create_training_logger(config, args, use_semantic_split=args.use_semantic_split)
    logger.info("Config:\n" + "\n".join(f"{k}: {v}" for k, v in config.items()))  

    # loading dataloader  
    model_name = config['model_name']  
    model = None
    if args.dataset == 'iabench':
        # DEFL owns the CLIP model; reuse its preprocessing when creating split views.
        model_config = dict(config)
        if args.resume_checkpoint and model_name == 'dna':
            model_config['load_param'] = False
        model = ATTRIBUTOR[model_name](model_config)
        preprocess = model.semantic_extractor.clip_preprocess if model_name == 'defl' else None
        train_loader, val_loader = get_iabench_dataloaders(
            train_source, val_source, config, args.batch_size, args.num_workers, preprocess)
        test_loader = None
        logger.info(f"Checkpoints: {log_dir}/ckpt_best.pth and {log_dir}/ckpt_last.pth")
    elif args.use_semantic_split:
        print("using semantic split...")  
        train_semantics, test_semantics = get_semantic(args.task_id)  
        print("training semantic:", train_semantics)  
        print("testing semantic:", test_semantics)  
        train_loader, val_loader, test_loader = get_dataloader(  
            root_dir=args.root_dir,  
            model_name=model_name,  
            num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,  
            train_semantics=train_semantics,  
            test_semantics=test_semantics,  
            batch_size=args.batch_size,  
            num_workers=args.num_workers,
            config=config,
            use_semantic_split=True,  
        )  
    else:  
        print("using normal split...")  
        train_loader, val_loader, test_loader = get_dataloader(  
            root_dir=args.root_dir,  
            model_name=model_name,  
            num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,  
            batch_size=args.batch_size,  
            num_workers=args.num_workers,
            config=config,
        )  
    logger.info(f"train_loader samples: {len(train_loader)}")  
    logger.info(f"val_loader samples: {len(val_loader)}")  
    if test_loader is not None:
        logger.info(f"test_loader samples: {len(test_loader)}")

    if model is None:
        model = ATTRIBUTOR[config['model_name']](config)
    logger.info(model.parameters())
    optimizer = choose_optimizer(model, config)  
    scheduler = choose_scheduler(config, optimizer)  
    metric_scoring = choose_metric(config)  

    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=now, log_dir=log_dir)  

    # resume training 
    start_epoch = config.get('start_epoch', 1)  
    best_val_metric = None  
    best_epoch = 0  

    resume_ckpt = args.resume_checkpoint
    if resume_ckpt:  
        best_metrics, ckpt_epoch = trainer.load_checkpoint(resume_ckpt)
        logger.info(f"Resumed checkpoint from {resume_ckpt}")
        if ckpt_epoch is not None:
            start_epoch = ckpt_epoch + 1
        if best_metrics is not None:
            best_val_metric = best_metrics.get('val_metric')
            best_epoch = best_metrics.get('best_epoch', 0)

    n_epochs = config.get('nEpochs', 10)  
    if hasattr(args, 'n_epoch') and args.n_epoch is not None:  
        n_epochs = args.n_epoch  
          
    save_freq = config.get('save_freq', 5)  

    for epoch in range(start_epoch, n_epochs + 1):  
        logger.info(f"#### Epoch {epoch} ####")  
        val_metric, val_metrics = trainer.train_epoch(epoch, train_loader, val_loader={'test': val_loader})  

        if val_metric is not None:  
            if (best_val_metric is None) or (val_metric > best_val_metric):  
                best_val_metric = val_metric  
                best_epoch = epoch  
                logger.info(f"Validation metric improved, saving best checkpoint at epoch {epoch}")  
                trainer.save_checkpoint(filename="ckpt_best.pth", best_metrics={'val_metric': best_val_metric, 'best_epoch': best_epoch}, epoch=epoch)

        if args.dataset == 'iabench':
            trainer.save_checkpoint(filename='ckpt_last.pth',
                                    best_metrics={'val_metric': best_val_metric, 'best_epoch': best_epoch}, epoch=epoch)
        elif epoch % save_freq == 0 or epoch == n_epochs:
            ckpt_name = f"ckpt_epoch_{epoch}.pth"  
            logger.info(f"Saving checkpoint at epoch {epoch}: {ckpt_name}")  
            trainer.save_checkpoint(filename=ckpt_name, best_metrics={'val_metric': best_val_metric, 'best_epoch': best_epoch}, epoch=epoch)

    logger.info(f"Training complete. Best val at epoch {best_epoch}: {best_val_metric}")  

    for writer in trainer.writers.values():  
        writer.close()  

    if args.do_test and args.dataset == 'iab':
        degraded_levels = list(range(7))  

        for degraded in degraded_levels:  
            print(f"Testing with degraded level {degraded}...")  

            if args.use_semantic_split:  
                train_loader, val_loader, test_loader = get_dataloader(  
                    root_dir=args.root_dir,  
                    model_name=model_name,  
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
                    model_name=model_name,  
                    num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,  
                    batch_size=args.batch_size,  
                    degraded=degraded, 
                    config=config,
                    num_workers=args.num_workers
                )  

            test_metrics = trainer.test(test_loader)  

            # save results
            result_txt_path = os.path.join(log_dir, f"test_results_degraded_{degraded}.txt")  
            with open(result_txt_path, 'w') as f:  
                f.write(f"Test metrics for degraded={degraded} ({datetime.datetime.now()}):\n")  
                for metric_name, value in test_metrics.items():  
                    if metric_name == "conf_matrix":  
                        f.write(f"{metric_name}:\n{value}\n")  
                    else:  
                        f.write(f"{metric_name}: {value}\n")  

            print(f"Saved test results for degraded={degraded} to {result_txt_path}")  
if __name__ == '__main__':  
    main()  