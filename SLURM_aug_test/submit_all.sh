#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
LAUNCHERS=(
    eval_resnet50.sbatch
    eval_dct.sbatch
    eval_hifi_net.sbatch
    eval_defl.sbatch
    eval_dna.sbatch
    eval_repmix.sbatch
    eval_patch.sbatch
    eval_ucf.sbatch
    eval_hypclip.sbatch
)

cd "$REPO_ROOT"
for launcher in "${LAUNCHERS[@]}"; do
    sbatch "SLURM_aug_test/$launcher"
done
