import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import importlib
import warnings

from comparison.training.metrics.registry import ATTRIBUTOR

# Each attributor self-registers into ATTRIBUTOR on import. Some methods pull
# optional third-party deps (cv2, yacs, torch_dct, seaborn, openai-CLIP, ...).
# Import defensively so a missing dep for an UNUSED method does not prevent
# running the others (e.g. resnet50). A skipped method just won't be in
# ATTRIBUTOR, raising a clear KeyError only if you actually select it.
_ATTRIBUTOR_MODULES = [
    "attributor_resnet50",
    "attributor_clip_lr",
    "attributor_repmix",
    "attributor_hifi_net",
    "attributor_defl",
    "attributor_ssp",
    "attributor_patchcraft",
    "attributor_dct",
    "attributor_dna",
    "attributor_ucf",
    "attributor_patch",
    "attributor_gfd",
    "attributor_pose",
]
for _m in _ATTRIBUTOR_MODULES:
    try:
        importlib.import_module(f"{__name__}.{_m}")
    except Exception as _e:
        warnings.warn(f"[attributors] skipped {_m}: {type(_e).__name__}: {_e}")
