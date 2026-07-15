import os
import sys

current_file_path = os.path.abspath(__file__)  
parent_dir = os.path.dirname(current_file_path)          # ImageAttributionDataset 
dataset_root_dir = os.path.dirname(parent_dir)           # dataset
project_root_dir = os.path.dirname(dataset_root_dir)     # project-root
sys.path.append(parent_dir)
sys.path.append(dataset_root_dir)
sys.path.append(project_root_dir)


import importlib
import warnings

from comparison.training.metrics.registry import DATASET

# Each dataset module self-registers into DATASET on import. Some methods pull
# optional third-party deps (cv2, torch_dct, albumentations, openai-CLIP, ...).
# Import defensively so a missing dep for an UNUSED method does not prevent
# running the others (e.g. resnet50). A skipped method just won't be in DATASET.
_DATASET_MODULES = [
    "dataset_resnet50",
    "dataset_dct",
    "dataset_hifi_net",
    "dataset_defl",
    "dataset_hypclip",
]
for _m in _DATASET_MODULES:
    try:
        importlib.import_module(f"{__name__}.{_m}")
    except Exception as _e:
        warnings.warn(f"[dataset] skipped {_m}: {type(_e).__name__}: {_e}")
