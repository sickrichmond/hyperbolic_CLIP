"""TwinNeXt optimization with OmniDFA's stepwise learning-rate schedule."""
import csv
import math
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from comparison.training.trainer.trainer import Trainer


def omnidfa_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs=2):
    warmup = warmup_epochs * steps_per_epoch
    cycle = epochs * steps_per_epoch
    if cycle <= 0:
        raise ValueError('OmniDFA scheduler needs positive training steps')

    def factor(step):
        if step < warmup:
            return step / warmup
        # Preserve the released scheduler's post-warmup period, including its offset.
        progress = (step - warmup) % cycle / cycle
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class OmniDFATrainer(Trainer):
    def _setup_model(self):
        self.model.to(self.device)
        # Gather logits with gradients before computing cross-entropy.
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    @property
    def attributor(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def save_checkpoint(self, *args, **kwargs):
        # The base checkpoint writer stores unwrapped model keys.
        model = self.model
        self.model = self.attributor
        try:
            return super().save_checkpoint(*args, **kwargs)
        finally:
            self.model = model

    def load_checkpoint(self, path):
        from comparison.training.omnidfa import validate_omnidfa_checkpoint
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        validate_omnidfa_checkpoint(checkpoint, for_training=True)
        epoch = checkpoint.get('epoch')
        if type(epoch) is not int or (checkpoint.get('scheduler_state_dict') or {}).get('last_epoch') != (
                epoch * self.config['steps_per_epoch']):
            raise ValueError('OmniDFA resume scheduler does not match the saved epoch')
        model = self.model
        self.model = self.attributor
        try:
            return super().load_checkpoint(path)
        finally:
            self.model = model

    def train_epoch(self, epoch, train_loader, val_loader=None):
        train_loader.batch_sampler.set_epoch(epoch)
        self.set_train()
        use_bfloat16 = self.device.type == 'cuda' and self.config['precision'] == 'bfloat16'
        if use_bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError('OmniDFA bfloat16 training requires a supported GPU; select float32 otherwise')
        for step, batch in enumerate(tqdm(train_loader, desc=f'OmniDFA epoch {epoch}')):
            batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                predictions = self.model(batch)
            losses = self.attributor.compute_losses(batch, predictions)
            if not torch.isfinite(losses['overall']):
                raise ValueError('Nonfinite OmniDFA training loss')
            losses['overall'].backward()
            self.optimizer.step()
            self.scheduler.step()
            global_step = (epoch - 1) * len(train_loader) + step
            for key, value in losses.items():
                self.get_writer('train', self.dataset_name, key).add_scalar(key, value.item(), global_step)
            with (Path(self.log_dir) / 'steps.csv').open('a', newline='') as stream:
                writer = csv.writer(stream)
                if stream.tell() == 0:
                    writer.writerow(['epoch', 'step', 'global_step', *losses, 'lr'])
                writer.writerow([epoch, step, global_step, *(value.item() for value in losses.values()),
                                 self.scheduler.get_last_lr()[0]])
            if step % 100 == 0:
                self.logger.info(f'OmniDFA epoch={epoch} step={step} '
                                 f'loss={losses["overall"].item():.6f} '
                                 f'lr={self.scheduler.get_last_lr()[0]:.8g}')
        if val_loader is None:
            return None, None
        metrics = self.evaluate(val_loader['test'])
        return metrics['balanced_accuracy'], metrics

    @torch.no_grad()
    def evaluate(self, val_loader):
        # The base implementation recognizes DDP but not DataParallel.
        model = self.model
        self.model = self.attributor
        try:
            return super().evaluate(val_loader)
        finally:
            self.model = model

    def test(self, test_loader):
        model = self.model
        self.model = self.attributor
        try:
            return super().test(test_loader)
        finally:
            self.model = model
