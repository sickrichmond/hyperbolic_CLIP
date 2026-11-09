"""TwinNeXt classification with evaluation support for legacy prototype banks."""
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score

from comparison.training.attributors import ATTRIBUTOR
from comparison.training.attributors.attributor_base import AbstractAttributor
from comparison.training.backbones.omnidfa import TwinNeXt
from comparison.training.metrics.base_metrics_class import (
    calculate_metrics_for_train, calculate_metrics_for_test,
)
from comparison.training.omnidfa import PROTOTYPE_PROTOCOL


@ATTRIBUTOR.register_module(module_name='omnidfa')
class OmniDFAAttributor(AbstractAttributor):
    def __init__(self, config=None, load_param=False):
        super().__init__(config, load_param)
        self.config = config or {}
        self.num_classes = self.config['num_classes']
        self.use_prototypes = self.config.get('attribution_protocol') == PROTOTYPE_PROTOCOL
        self.build_model(self.config)
        self.build_loss(self.config)
        if self.use_prototypes:
            self.register_buffer('prototypes', torch.zeros(self.num_classes, 128))
            self.register_buffer('prototype_counts', torch.zeros(self.num_classes, dtype=torch.long))
            # Keep the legacy state keys for strict loading. Cosine attribution only uses the bank.
            self.center_loss = nn.Module()
            self.center_loss.register_buffer('center_real', torch.zeros(128))
            self.center_loss.register_buffer('cosine_threshold', torch.tensor(1.0))
        if load_param:
            raise ValueError('Use --resume_checkpoint to load a complete OmniDFA checkpoint')

    def build_model(self, config):
        self.backbone = TwinNeXt(pretrained=config.get('pretrained', True),
                                 out_features=128 if self.use_prototypes else self.num_classes)

    def build_loss(self, config):
        self.loss_fn = nn.CrossEntropyLoss()

    def extract_features(self, input_data):
        return self.backbone.forward_features(input_data['image'], input_data['global_image'])

    def classifier(self, features):
        logits = self.backbone.mlp(features).float()
        if self.use_prototypes:
            return F.normalize(logits, dim=1) @ self.prototypes.T
        return logits

    def forward(self, input_data, inference=False):
        if self.use_prototypes and (self.prototype_counts <= 0).any():
            raise ValueError('OmniDFA inference requires a complete saved training prototype bank')
        features = self.extract_features(input_data)
        logits = self.classifier(features)
        out = {'logits': logits, 'cls': logits.argmax(dim=1)}
        if inference:
            out['probs'] = logits.softmax(dim=1)
        return out

    def compute_losses(self, input_data, pred_dict):
        if self.use_prototypes:
            raise ValueError('OmniDFA prototype checkpoints are supported for evaluation only')
        return {'overall': self.loss_fn(pred_dict['logits'], input_data['label'].long())}

    def compute_metrics(self, input_data, pred_dict, test=False):
        labels = input_data['label'].detach().cpu()
        logits = pred_dict['logits'].detach().cpu()
        balanced = balanced_accuracy_score(labels.numpy(), logits.argmax(dim=1).numpy())
        if test:
            auc, acc, ap, matrix, semantic, extra = calculate_metrics_for_test(
                labels, logits, input_data.get('semantic_label'))
            return {'auc': float(auc), 'acc': float(acc), 'ap': float(ap),
                    'conf_matrix': matrix, 'semantic_acc': semantic, **extra}
        auc, acc, ap = calculate_metrics_for_train(labels, logits)
        return {'auc': float(auc), 'acc': float(acc), 'ap': float(ap),
                'balanced_accuracy': float(balanced)}
