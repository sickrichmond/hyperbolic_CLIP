#!/bin/bash
#SBATCH --account=EUHPC_D35_189
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=aug_ucf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=SLURM_aug_test/logs/%x_%A_%a.out
#SBATCH --error=SLURM_aug_test/logs/%x_%A_%a.err

bash SLURM_aug_test/run_array_task.sh ucf
