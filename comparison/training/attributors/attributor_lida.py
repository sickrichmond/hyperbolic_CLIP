import torch
import torch.nn as nn
import torch.nn.functional as F

from comparison.dataset.ImageAttributionDataset.dataset import model_class_to_label
from comparison.training.attributors import ATTRIBUTOR
from comparison.training.backbones.lida import LIDABackbone
from comparison.training.attributors.attributor_base import AbstractAttributor
from comparison.training.metrics.base_metrics_class import (
    calculate_metrics_for_test,
    calculate_metrics_for_train,
)


@ATTRIBUTOR.register_module(module_name="lida")
class LIDAAttributor(AbstractAttributor):
    def __init__(self, config=None, load_param=False):
        super().__init__(config, load_param)
        self.config = config or {}
        self.build_model(self.config)
        self.build_loss(self.config)

        if load_param:
            self.load_parameters(load_param)

    def build_model(self, config):
        self.backbone = LIDABackbone(pretrained=config.get("pretrained", True))
        self.fc = nn.Linear(self.backbone.output_dim, config.get("num_classes", len(model_class_to_label)))

    def build_loss(self, config):
        self.loss_fn = nn.CrossEntropyLoss()

    def extract_features(self, input_data):
        return self.backbone(input_data["image"])

    def classifier(self, features):
        return self.fc(features)

    def forward(self, input_data, inference=False):
        features = self.extract_features(input_data)
        logits = self.classifier(features)
        output = {
            "logits": logits,
            "features": features,
            "cls": torch.argmax(logits, dim=1),
        }
        if inference:
            output["probs"] = F.softmax(logits, dim=1)
        return output

    def compute_losses(self, input_data, pred_dict):
        return {"overall": self.loss_fn(pred_dict["logits"], input_data["label"])}

    def compute_metrics(self, input_data, pred_dict, test=False):
        label = input_data["label"]
        pred = pred_dict["logits"]
        if test:
            auc, acc, ap, conf_matrix, semantic_acc, extra_metrics = calculate_metrics_for_test(
                label.detach(), pred.detach(), input_data.get("semantic_label")
            )
            return {
                "acc": float(acc),
                "auc": float(auc),
                "ap": float(ap),
                "conf_matrix": conf_matrix,
                "semantic_acc": semantic_acc,
                **extra_metrics,
            }
        auc, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {"acc": float(acc), "auc": float(auc), "ap": float(ap)}

    def load_parameters(self, load_param):
        path = load_param if isinstance(load_param, str) else self.config.get("pretrained_path")
        if not path:
            raise ValueError("No pretrained_path configured for LIDA checkpoint loading")
        self.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded parameters from {path}")