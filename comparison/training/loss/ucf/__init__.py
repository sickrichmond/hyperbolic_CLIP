import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from comparison.training.metrics.registry import LOSSFUNC

from .cross_entropy_loss import CrossEntropyLoss
from .contrastive_regularization import ContrastiveLoss
from .l1_loss import L1Loss
# `pose` is not among the ported methods and the UCF config only uses
# cross_entropy / contrastive_regularization / l1loss, so its Softmax loss is dropped.

