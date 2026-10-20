import torch
import torch.nn as nn
import torch.nn.functional as F

from comparison.dataset.ImageAttributionDataset.dataset import model_class_to_label
from comparison.training.attributors import ATTRIBUTOR
from comparison.training.attributors.attributor_base import AbstractAttributor
from comparison.training.backbones.pose import POSEFeatureExtractor, SingleLayerAugmenter
from comparison.training.metrics.base_metrics_class import (
    calculate_metrics_for_test,
    calculate_metrics_for_train,
)


@ATTRIBUTOR.register_module(module_name="pose")
class POSEAttributor(AbstractAttributor):
    """Progressive Open Space Expansion model for image attribution."""

    def __init__(self, config=None, load_param=False):
        super().__init__(config, load_param)
        self.config = config or {}
        self.num_classes = self.config.get("num_classes", len(model_class_to_label))
        if self.num_classes != len(model_class_to_label):
            raise ValueError(
                f"POSE config has {self.num_classes} classes, but the active IAB "
                f"label map has {len(model_class_to_label)}. Check "
                "IAB_EXCLUDE_GENERATORS and the POSE config."
            )
        self.num_augmenters = self.config.get("num_augmenters", 30)
        self.build_model(self.config)
        self.build_loss(self.config)
        if load_param:
            self.load_parameters(load_param)

    def build_model(self, config):
        self.feature_extractor = POSEFeatureExtractor(
            self.num_classes,
            input_data=config.get("input_data", "dct"),
            feature_dim=config.get("feature_dim", 512),
        )
        self.augmenters = nn.ModuleList(
            SingleLayerAugmenter(
                hidden_channels=config.get("inc", 32),
                kernel_size=config.get("kernel_size", 3),
            )
            for _ in range(self.num_augmenters)
        )

    def build_loss(self, config):
        self.loss_fn = nn.CrossEntropyLoss()
        self.temperature = config.get("temp", 0.1)

    def extract_features(self, input_data):
        return self.feature_extractor.extract_features(input_data["image"])

    def classifier(self, features):
        return self.feature_extractor.classify(features)

    def forward(self, input_data, inference=False):
        features = self.extract_features(input_data)
        logits = self.classifier(features)
        output = {"logits": logits, "features": features}
        if inference:
            output["probs"] = F.softmax(logits, dim=1)
        return output

    def compute_losses(self, input_data, pred_dict):
        return {
            "overall": self.classification_loss(
                pred_dict["logits"], input_data["label"].long()
            )
        }

    def classification_loss(self, logits, labels):
        probabilities = F.softmax(logits, dim=1)
        return self.loss_fn(probabilities / self.temperature, labels)

    def compute_metrics(self, input_data, pred_dict, test=False):
        labels = input_data["label"]
        logits = pred_dict["logits"]
        if test:
            auc, acc, ap, conf_matrix, semantic_acc, extra_metrics = (
                calculate_metrics_for_test(
                    labels.detach(),
                    logits.detach(),
                    input_data.get("semantic_label"),
                )
            )
            return {
                "acc": float(acc),
                "auc": float(auc),
                "ap": float(ap),
                "conf_matrix": conf_matrix,
                "semantic_acc": semantic_acc,
                **extra_metrics,
            }
        auc, acc, ap = calculate_metrics_for_train(labels.detach(), logits.detach())
        return {"acc": float(acc), "auc": float(auc), "ap": float(ap)}

    def load_parameters(self, load_param):
        path = (
            load_param
            if isinstance(load_param, str)
            else self.config.get("pretrained_path")
        )
        if not path:
            raise ValueError("No pretrained_path configured for POSE checkpoint loading")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.load_state_dict(state_dict)
