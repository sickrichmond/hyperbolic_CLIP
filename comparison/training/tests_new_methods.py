"""Self-check for the class-count wiring of every ported attributor.

Runs on CPU, no dataset, no framework:

    python -m comparison.training.tests_new_methods

Catches the class of bug that invalidated the first 22-class runs: a head sized
from a hardcoded 23, or `real` identified by a hardcoded index, both of which go
wrong the moment IAB_EXCLUDE_GENERATORS re-indexes the label map. Each case is
re-imported in a subprocess because the label map is read at import time.
"""
import os
import subprocess
import sys

# The DEFL hierarchy as it was hardcoded in dataset_defl.py before it was derived
# from _HIFI_HIERARCHY. Kept here as the regression oracle: at 23 classes the
# derived mapping must reproduce it exactly, i.e. the fix changed nothing for the
# already-published 23-class numbers.
_DEFL_23 = [
    (0,0,0,0), (0,1,3,1), (0,1,2,2), (0,1,2,3), (0,1,2,4), (0,1,2,5),
    (0,1,1,6), (0,1,1,7), (0,1,1,8), (0,1,1,9), (0,1,1,10), (0,0,0,11),
    (0,0,0,12), (0,0,0,13), (0,1,3,14), (0,1,3,15), (0,0,0,16), (0,1,4,17),
    (0,1,4,18), (0,0,0,19), (0,0,0,20), (0,0,0,21), (1,2,5,22),
]

CHILD = r'''
import torch
from comparison.dataset.ImageAttributionDataset.dataset import (
    model_class_to_label, hifi_label_mapping)
from comparison.training.attributors import ATTRIBUTOR
from comparison.dataset.ImageAttributionDataset import DATASET

n = int({n})
assert len(model_class_to_label) == n, (len(model_class_to_label), n)

# every method must be registered: a defensive import that swallowed one would
# only surface much later, as a KeyError inside a slurm job.
for m in ["resnet50", "dct", "hifi_net", "defl", "dna", "repmix", "patch", "ucf"]:
    assert m in ATTRIBUTOR.data, "attributor missing: " + m
    assert m in DATASET.data, "dataset missing: " + m

# `real` is the LAST class in both label spaces; index-free code must follow it.
assert model_class_to_label["real"] == n - 1, model_class_to_label["real"]

# hierarchical mappings track the active map
assert len(hifi_label_mapping()) == n
assert hifi_label_mapping()[-1][:3] == (1, 2, 5)   # real: generated=1, real, real

# attribution heads sized from the active map, not from a literal 23
from comparison.training.attributors.attributor_repmix import RepmixAttributor
r = RepmixAttributor({{"device": "cpu"}})
assert r.attribution.out_features == n, r.attribution.out_features
assert r.detection.out_features == 2

# RepMix gating: the real column must be gated by P(real) (detection col 0) and
# every other column by P(generated) (col 1). Feed a feature batch and check the
# sign pattern by driving the detection head to a known state.
with torch.no_grad():
    r.detection.weight.zero_(); r.detection.bias.copy_(torch.tensor([50.0, -50.0]))
    r.attribution.weight.zero_(); r.attribution.bias.fill_(1.0)
    out, _ = r.classifier(torch.zeros(2, r.config["d_embed"], device=r.device))
# P(real) ~ 1, P(generated) ~ 0  =>  only the real column survives
assert out[0, n - 1] > 0.99, out[0, n - 1]
assert out[0, :n - 1].abs().max() < 0.01, out[0, :n - 1].abs().max()

print("  ok: {n} classes")
'''


def run_case(n, env_exclude):
    env = dict(os.environ)
    if env_exclude:
        env["IAB_EXCLUDE_GENERATORS"] = env_exclude
    else:
        env.pop("IAB_EXCLUDE_GENERATORS", None)
    print(f"[case] IAB_EXCLUDE_GENERATORS={env_exclude or '<unset>'} -> {n} classes")
    r = subprocess.run([sys.executable, "-c", CHILD.format(n=n)], env=env)
    return r.returncode == 0


def main():
    ok = run_case(23, None) & run_case(22, "dalle3")

    # DEFL regression: derived == hardcoded at 23 classes.
    env = dict(os.environ)
    env.pop("IAB_EXCLUDE_GENERATORS", None)
    src = (
        "from comparison.dataset.ImageAttributionDataset.dataset import hifi_label_mapping;"
        f"assert hifi_label_mapping() == {_DEFL_23!r}, 'DEFL mapping drifted at 23 classes';"
        "print('  ok: DEFL 23-class mapping unchanged')"
    )
    print("[case] DEFL hierarchy regression")
    ok &= subprocess.run([sys.executable, "-c", src], env=env).returncode == 0

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
