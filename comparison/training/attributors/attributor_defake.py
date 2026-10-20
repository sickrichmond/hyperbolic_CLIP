import torch
import torch.nn as nn
import torch.nn.functional as F

from comparison.dataset.ImageAttributionDataset.dataset import model_class_to_label
from comparison.training.attributors import ATTRIBUTOR
from comparison.training.attributors.attributor_base import AbstractAttributor
from comparison.training.backbones.defake import DeFakeImageEncoder
from comparison.training.metrics.base_metrics_class import (
    calculate_metrics_for_test,
    calculate_metrics_for_train,
)


@ATTRIBUTOR.register_module(module_name="defake")
class DeFakeAttributor(AbstractAttributor):
    """Image-only De-Fake: frozen CLIP image features and a learned MLP head."""

    def __init__(self, config=None, load_param=False):
        super().__init__(config, load_param)
        self.config = config or {}
        self.build_model(self.config)
        self.build_loss(self.config)
        if load_param:
            self.load_parameters(load_param)

    def build_model(self, config):
        self.backbone = DeFakeImageEncoder(
            model_name=config.get("clip_model", "ViT-B/32")
        )
        hidden_dims = config.get("hidden_dims", [512, 256])
        if len(hidden_dims) != 2:
            raise ValueError("De-Fake hidden_dims must contain exactly two sizes")
        num_classes = config.get("num_classes", len(model_class_to_label))
        self.fc1 = nn.Linear(self.backbone.output_dim, hidden_dims[0])
        self.dropout = nn.Dropout(config.get("dropout", 0.5))
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def build_loss(self, config):
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, input_data):
        with torch.no_grad():
            return self.backbone(input_data["image"])

    def classifier(self, features):
        features = self.dropout(F.relu(self.fc1(features)))
        features = F.relu(self.fc2(features))
        return self.fc3(features)

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
        labels = input_data["label"]
        logits = pred_dict["logits"]
        if test:
            auc, acc, ap, conf_matrix, semantic_acc, extra_metrics = calculate_metrics_for_test(
                labels.detach(), logits.detach(), input_data.get("semantic_label")
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
        path = load_param if isinstance(load_param, str) else self.config.get("pretrained_path")
        if not path:
            raise ValueError("No pretrained_path configured for De-Fake checkpoint loading")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.load_state_dict(state_dict)