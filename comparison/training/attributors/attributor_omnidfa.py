"""Closed-set OmniDFA: native embedding loss, clean training-class prototypes."""
import torch
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score

from comparison.training.attributors import ATTRIBUTOR
from comparison.training.attributors.attributor_base import AbstractAttributor
from comparison.training.backbones.omnidfa import (
    TwinNeXt, SphereCenterLoss, supervised_contrastive_loss,
)
from comparison.training.metrics.base_metrics_class import (
    calculate_metrics_for_train, calculate_metrics_for_test,
)


@ATTRIBUTOR.register_module(module_name='omnidfa')
class OmniDFAAttributor(AbstractAttributor):
    def __init__(self, config=None, load_param=False):
        super().__init__(config, load_param)
        self.config = config or {}
        self.num_classes = self.config['num_classes']
        self.real_class_index = self.config['real_class_index']
        self.build_model(self.config)
        self.build_loss(self.config)
        self.register_buffer('prototypes', torch.zeros(self.num_classes, 128))
        self.register_buffer('prototype_counts', torch.zeros(self.num_classes, dtype=torch.long))
        if load_param:
            raise ValueError('Use --resume_checkpoint to load a complete OmniDFA checkpoint')

    def build_model(self, config):
        self.backbone = TwinNeXt(pretrained=config.get('pretrained', True))

    def build_loss(self, config):
        self.center_loss = SphereCenterLoss()

    def extract_features(self, input_data):
        return F.normalize(self.backbone(input_data['image'], input_data['global_image']).float(), dim=1)

    def classifier(self, features):
        return features @ self.prototypes.T

    def forward(self, input_data, inference=False):
        if inference and (self.prototype_counts <= 0).any():
            raise ValueError('OmniDFA inference requires a complete saved training prototype bank')
        features = self.extract_features(input_data)
        logits = self.classifier(features)
        out = {'features': features, 'logits': logits, 'cls': logits.argmax(dim=1)}
        if inference:
            out['probs'] = logits.softmax(dim=1)
        return out

    def compute_losses(self, input_data, pred_dict):
        labels = input_data['label'].long()
        features = pred_dict['features']
        # DataParallel gathers the entire differentiable batch before this loss.
        contrastive = supervised_contrastive_loss(features, labels)
        center = self.center_loss(features[labels == self.real_class_index])
        return {'overall': contrastive + 0.01 * center,
                'contrastive': contrastive.detach(), 'center': center.detach()}

    @torch.no_grad()
    def set_prototypes(self, sums, counts):
        if (counts <= 0).any() or not torch.isfinite(sums).all() or (sums.norm(dim=1) <= 1e-8).any():
            raise ValueError('OmniDFA needs a finite, nonzero training prototype for every class')
        self.prototypes.copy_(F.normalize(sums, dim=1))
        self.prototype_counts.copy_(counts)

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
