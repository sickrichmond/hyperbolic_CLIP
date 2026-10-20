import numpy as np
import torch
import torch.nn.functional as F

from comparison.training.backbones.pose import TripletLoss
from comparison.training.trainer.trainer import Trainer


class POSETrainer(Trainer):
    """Implements POSE's alternating augmenter / feature-extractor updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_loss = TripletLoss(
            margin=self.config.get("triplet_margin", 0.3)
        ).to(self.device)
        self.reconstruction_loss = torch.nn.MSELoss()
        self.augmenter_optimizers = {}
        self.augmenter_schedulers = {}
        self.current_epoch = None

    def checkpoint_state(self):
        return {
            "augmenter_optimizer_state_dicts": {
                index: optimizer.state_dict()
                for index, optimizer in self.augmenter_optimizers.items()
            },
            "augmenter_scheduler_state_dicts": {
                index: scheduler.state_dict()
                for index, scheduler in self.augmenter_schedulers.items()
            },
        }

    def restore_checkpoint_state(self, checkpoint):
        optimizers = checkpoint.get("augmenter_optimizer_state_dicts", {})
        schedulers = checkpoint.get("augmenter_scheduler_state_dicts", {})
        if self.config.get("dataset") == "iabench":
            expected = set(range(checkpoint["epoch"] - self.config.get("start_epoch", 1) + 1))
            if set(optimizers) != expected or set(schedulers) != expected:
                raise ValueError("POSE resume requires optimizer and scheduler state for each trained augmenter")
        for index, state in optimizers.items():
            optimizer, scheduler = self._augmenter_optimizer(index, self.model.augmenters[index])
            optimizer.load_state_dict(state)
            scheduler.load_state_dict(schedulers[index])

    def train_epoch(self, epoch, train_loader, val_loader=None):
        self.current_epoch = epoch
        augmenter_index = epoch - self.config.get("start_epoch", 1)
        if augmenter_index < 0 or augmenter_index >= len(self.model.augmenters):
            raise ValueError(
                f"POSE epoch {epoch} requires augmenter index {augmenter_index}, "
                f"but only {len(self.model.augmenters)} are configured"
            )
        return super().train_epoch(epoch, train_loader, val_loader)

    def _set_grad(self, module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _augmenter_optimizer(self, index, augmenter):
        if index not in self.augmenter_optimizers:
            self.augmenter_optimizers[index] = torch.optim.Adam(
                augmenter.parameters(),
                lr=self.config.get("augnet_lr", 1e-2),
            )
            self.augmenter_schedulers[index] = torch.optim.lr_scheduler.StepLR(
                self.augmenter_optimizers[index],
                step_size=self.config.get("lr_step", 500),
                gamma=self.config.get("lr_gamma", 0.9),
            )
        return (
            self.augmenter_optimizers[index],
            self.augmenter_schedulers[index],
        )

    def train_step(self, data_dict):
        model = self.model
        images = data_dict["image"]
        labels = data_dict["label"].long()
        epoch_start = self.config.get("start_epoch", 1)
        current_index = self.current_epoch - epoch_start
        augmenter = model.augmenters[current_index]
        augmenter_optimizer, augmenter_scheduler = self._augmenter_optimizer(
            current_index, augmenter
        )

        self._set_grad(model.feature_extractor, False)
        for previous in model.augmenters:
            self._set_grad(previous, False)
        self._set_grad(augmenter, True)
        augmenter_optimizer.zero_grad(set_to_none=True)

        augmented = augmenter(images.detach())
        loss_augmenter = self.reconstruction_loss(augmented, images)
        loss_values = {"loss_mse": loss_augmenter.detach().item()}
        with torch.no_grad():
            known_features = model.extract_features({"image": images})
        augmented_features = model.feature_extractor.extract_features(augmented)

        if current_index > 0:
            cosine_similarity = F.cosine_similarity(
                known_features, augmented_features, dim=1
            ).mean()
            close_loss = torch.clamp(
                1 - cosine_similarity,
                min=1 - self.config.get("known_sim_limit", 0.95),
            )
            previous_index = np.random.randint(0, current_index)
            with torch.no_grad():
                previous_images = model.augmenters[previous_index](images)
                previous_features = model.feature_extractor.extract_features(
                    previous_images
                )
            distant_loss = F.cosine_similarity(
                augmented_features, previous_features, dim=1
            ).mean()
            loss_augmenter = (
                loss_augmenter
                + self.config.get("w_close_known", 1e-2) * close_loss
                + self.config.get("w_dist_pre", 1e-4) * distant_loss
            )
            loss_values.update(
                {
                    "loss_close_known": close_loss.detach().item(),
                    "loss_distant_preaug": distant_loss.detach().item(),
                }
            )

        loss_augmenter.backward()
        augmenter_optimizer.step()
        augmenter_scheduler.step()

        self._set_grad(model.feature_extractor, True)
        for previous in model.augmenters:
            self._set_grad(previous, False)
        self.optimizer.zero_grad(set_to_none=True)

        logits, known_features = model.feature_extractor(images)
        classification_loss = model.classification_loss(logits, labels)
        augmented_images = augmenter(images)
        augmented_features = model.feature_extractor.extract_features(
            augmented_images
        )
        augmented_labels = labels + model.num_classes
        combined_features = torch.cat((known_features, augmented_features), dim=0)
        combined_labels = torch.cat((labels, augmented_labels), dim=0)
        metric_loss = self.metric_loss(combined_features, combined_labels)
        total_loss = classification_loss + metric_loss

        if current_index > 0 and self.config.get("cls_pre", True):
            previous_index = np.random.randint(0, current_index)
            previous_images = model.augmenters[previous_index](images)
            previous_features = model.feature_extractor.extract_features(
                previous_images
            )
            previous_metric_loss = self.metric_loss(
                torch.cat((known_features, previous_features), dim=0),
                combined_labels,
            )
            total_loss = total_loss + previous_metric_loss
            loss_values["loss_metric_pre"] = previous_metric_loss.detach().item()

        total_loss.backward()
        self.optimizer.step()
        if (
            self.scheduler is not None
            and self.config.get("scheduler_step_per_batch", False)
        ):
            self.scheduler.step()

        loss_values.update(
            {
                "loss_cls": classification_loss.detach().item(),
                "loss_metric": metric_loss.detach().item(),
                "overall": total_loss.detach().item(),
            }
        )
        return loss_values, {
            "logits": logits.detach(),
            "features": known_features.detach(),
        }
