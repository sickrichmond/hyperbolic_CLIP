# Augmented-split baseline evaluation

Each `eval_*.sbatch` file is a four-task array that reserves one node, four
GPUs, and 32 GB of memory. Array indices map to the four dataset names in
`datasets.txt`; each task evaluates every valid image in its dataset and writes
`metrics.json` plus `summary.csv` below:

```text
/leonardo_work/EUHPC_D35_189/hyp_fine_tuning/aug_test_results/<model>/<dataset>/
```

The three complete datasets contain 80,000 valid images each. The
photorealistic dataset contains 78,125. Temporary partial files are excluded
and counted in the JSON provenance (`ignored_non_image_files`). The optional
`real` symlink in the standard recap directory is excluded because this suite
scores the four requested generators while preserving their 22-way labels.

Run submission commands from the repository root. Submit all 32
model/dataset evaluations with:

```bash
bash SLURM_aug_test/submit_all.sh
```

Submit one model with, for example:

```bash
sbatch SLURM_aug_test/eval_resnet50.sbatch
```

The reviewed canonical 22-class checkpoint paths, model configs, batch sizes,
and worker counts are in `manifest.tsv`. The following environment variables
override runtime defaults:

- `AUG_TEST_CHECKPOINT`: checkpoint for the selected model.
- `AUG_TEST_DATA_ROOT`: parent directory containing the four `*_v2` datasets.
- `AUG_TEST_OUTPUT_ROOT`: result root.
- `AUG_TEST_BATCH_SIZE` and `AUG_TEST_NUM_WORKERS`: loader tuning.
- `AUG_TEST_NUM_GPUS`: local distributed process count from 1 to 4 (default 4).
- `AUG_TEST_MAX_IMAGES`: balanced smoke-test cap.
- `AUG_TEST_VALIDATE_ONLY=1`: validate construction and strict checkpoint load.
- `AUG_TEST_UCF_PRETRAINED`: UCF's required ImageNet Xception initialization.
- `IAB_PYDEPS`: DNA dependency overlay (defaults to the existing
  `$HOME/iab_pydeps` overlay when present). Stage-two inference loads the
  stage-one-only `albumentations` dependency lazily.

Example smoke submission:

```bash
AUG_TEST_MAX_IMAGES=64 sbatch SLURM_aug_test/eval_resnet50.sbatch
```

SLURM stdout/stderr files are written to `SLURM_aug_test/logs/`.
The evaluator rejects a run before inference if its paths, registry entries,
target labels, UCF initialization, or canonical 22-way checkpoint head are
invalid; checkpoint loading itself uses `strict=True`.

Inference uses one `torchrun` process per GPU. `AUG_TEST_BATCH_SIZE` is per GPU,
while `AUG_TEST_NUM_WORKERS` is a job-wide budget divided across ranks. Dataset
indices are sharded without padding, so every image is evaluated exactly once
even when the dataset size is not divisible by four. Only rank zero writes the
merged metrics files.

DeFL is also safe on offline compute nodes: its official RN50x16 preprocessing
and model are reconstructed from the complete CLIP state embedded in the
trained checkpoint, rather than attempting a network download.
