# Evaluate baseline checkpoints on IABench

One shared launcher supports ResNet50, DCT-CNN, HiFi-Net, DEFL, DNA-Det stage two,
RepMix, PatchForensics, UCF, LIDA, DeFake and POSE. Submit one job per explicitly
selected checkpoint:

```bash
cd "$WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo"
sbatch comparison/training/scripts/test_iabench.sh \
  /path/to/ckpt_best.pth
```

The checkpoint supplies the baseline, saved model configuration, class order,
head dimensions and hierarchy. No YAML selection or class exclusion setting is
needed. Both `ckpt_best.pth` and `ckpt_last.pth` work; use the best checkpoint for
the validation-selected comparison. DNA transformation/pretraining checkpoints
are rejected. DNA stage-two evaluation does not need the stage-one file.

Evaluation validates dataset/manifest digests, class order, split settings and
membership before loading the model. Only the manifest's test rows are decoded;
training and validation loaders are not constructed. All levels use the same
rows, without shuffling, mixing or dropped batches. `IAB_EXCLUDE_GENERATORS` does
not affect IABench. Metadata-free/legacy IAB checkpoints are not accepted here;
the existing IAB evaluation CLI and scripts retain their original behavior.

Each job requests one GPU, eight CPUs and `4-00:00:00` with `boost_qos_lprod`.
The environment/cache setup matches training, including the DNA dependency
overlay. Existing cached backbone dependencies still need to be available.

Defaults and environment overrides:

| Variable | Default |
| --- | --- |
| `CKPT` | Explicit positional checkpoint; alternatively set this variable |
| `DATA` | `$SCRATCH/datasets/IABench_images` |
| `SPLIT_MANIFEST` | `$WORK/hyp_fine_tuning/checkpoints/attribution_iabench_random_vitl14.splits.json` |
| `BATCH_SIZE` | `8` |
| `NUM_WORKERS` | `8` |
| `LOGDIR` | `$WORK/outputs/baselines_iabench_JOBID` |

The positional checkpoint takes precedence over `CKPT`. `DATA` also accepts the
original Arrow snapshot or its `data/` directory. The split manifest must match
the selected checkpoint; the exporter preserves this compatibility. For example:

```bash
DATA=/path/to/IABench/data NUM_WORKERS=2 BATCH_SIZE=32 \
  sbatch comparison/training/scripts/test_iabench.sh \
    /path/to/resnet50/ckpt_best.pth
```

The seven ImageAttributionBench degradation levels are applied to the decoded RGB
image before each baseline's native preprocessing:

| Level | Operation |
| --- | --- |
| 0 | Clean |
| 1 | Nearest-neighbor downsample/upscale at 0.5 scale |
| 2 | Nearest-neighbor downsample/upscale at 0.25 scale |
| 3 | JPEG quality 65 |
| 4 | JPEG quality 30 |
| 5 | Gaussian blur radius 3 |
| 6 | Gaussian blur radius 5 |

Results are `LOGDIR/test_results_degraded_LEVEL.txt` plus `test.log`. Each result
includes accuracy, balanced accuracy, macro AUC/AP, macro/weighted precision,
recall and F1, per-class metrics/support and the confusion matrix. Class names
follow checkpoint order. Checkpoint path, baseline, dataset and manifest digests,
split and test count are recorded for provenance. IABench has no semantic labels,
so no semantic accuracy is invented.

RepMix retains its native detection gate. PatchForensics retains probability
averaging and raw/voted/averaged patch diagnostics, while storing only image
probabilities and running patch counts. DEFL reuses its loaded CLIP preprocessing.
The existing hyperbolic launcher remains `slurm/slurm_eval_iabench.sh`.

For direct CLI use or a subset of levels:

```bash
python -m comparison.training.test --dataset iabench \
  --resume_checkpoint /path/to/ckpt_best.pth \
  --root_dir /path/to/IABench_images \
  --split_manifest /path/to/shared.splits.json \
  --batch_size 8 --num_workers 8 --level_start 0 --level_end 7 \
  --log_dir /path/to/results
```

`level_end` is exclusive. Reusing a results directory replaces files for the
requested levels; the default job-ID directory keeps separate submissions apart.

CPU checks: `python -m tests.test_iabench_baseline_evaluation` and
`python -m tests.test_iabench_baseline_eval_slurm`. They cover Arrow/export parity,
all eleven baseline formats, exact test membership at all seven levels, singleton
final batches, metadata/weight rejection and native PatchForensics metric parity.
Full GPU execution and throughput require verification on Leonardo.

LIDA, DeFake and POSE training/resume commands are in
[their IABench training guide](README_iabench_lida_defake_pose.md).

## Train with our corruption policy

HypCLIP already has an augmentation flag; ResNet50 now uses the same JPEG, blur
and downsampling implementation before its native preprocessing. After copying
the updated source to Leonardo, submit from the repository:

```bash
export DATA="$SCRATCH/datasets/IABench_images"
sbatch --time=4-00:00:00 --export=ALL,AUGMENT=1,AUG_POLICY=corruption slurm/slurm_train_iabench.sh
sbatch --export=ALL,AUGMENT=1,AUG_POLICY=corruption comparison/training/scripts/cineca_resnet50_train_iabench.sh
```

Both modes share the existing manifest; validation stays clean. HypCLIP saves
`attribution_iabench_random_anchor0p5_z4_ep30_corruption_vitl14.pt`. ResNet50 saves
best/last checkpoints under `comparison/training/logs/iabench/resnet50/_aug_TIMESTAMP/`. Clean defaults
and the original clean checkpoints are preserved. The new ResNet50 flag supports
our corruption policy; an explicit incompatible `AUG_POLICY` is rejected.

Use the existing hyperbolic evaluator for the HypCLIP checkpoint and the shared
baseline evaluator for ResNet50. Augmentation policy is included in ResNet50's
checkpoint metadata and evaluation provenance. Test images receive only the
requested fixed degradation, without training augmentation.

To resume augmented ResNet50 training, retain the mode flag:

```bash
sbatch --export=ALL,AUGMENT=1,RESUME_CHECKPOINT=/path/to/_aug_RUN/ckpt_last.pth \
  comparison/training/scripts/cineca_resnet50_train_iabench.sh
```

CPU verification: `python -m tests.test_iabench_resnet_augmentation`.
