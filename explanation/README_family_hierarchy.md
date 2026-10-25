# IABench family-cone training

Enable the hierarchy explicitly; existing launch commands keep their behavior:

```bash
sbatch --export=ALL,HIERARCHY=increfa slurm/slurm_train_iabench.sh
# The same hierarchy supports either existing augmentation policy:
sbatch --export=ALL,HIERARCHY=increfa,AUGMENT=1 slurm/slurm_train_iabench.sh
```

The clean checkpoint is
`$WORK/hyp_fine_tuning/checkpoints/attribution_iabench_random_hierarchy_increfa_vitl14.pt`.
Augmented names insert `_aug` or `_omniaug` before `_hierarchy_increfa`.
Diagnostics use the same suffix and the job ID. All modes reuse the existing
split manifest; validation selects checkpoints and the comparison harness alone
evaluates held-out test rows. Concurrent runs still need distinct output paths.

```bash
CKPT="$WORK/hyp_fine_tuning/checkpoints/attribution_iabench_random_hierarchy_increfa_vitl14.pt" \
  sbatch slurm/slurm_eval_iabench.sh
```

## Taxonomy and objective

`training/families.py` stores the
[released IncreFA IABench mapping](https://github.com/Ant0ny44/IncreFA/blob/main/configs/iabench_families.json).
It assigns four GANs, two autoregressive models (`cogview4`, `gpt4o`), and
22 diffusion models, plus a separate real family. These are the paper's
experimental assignments, not independently verified architectural ancestry.
The comparison baselines retain their separate taxonomy. Active family order
is real, GAN, autoregressive, diffusion, omitting absent families. Unknown class
names fail rather than acquiring a guessed family. Indices follow the exact
checkpoint class order.

Hierarchy mode currently requires random, learnable class anchors on IABench.
Family directions are independently random and learnable; they use a separate
seeded generator and the existing geometric learning rate with zero weight decay.

Let `xi(f, c)` be the exterior angle at family anchor `f` toward class anchor
`c`, and `psi(f)` its depth-dependent half-aperture. The additional family loss is
the mean positive hinge `max(0, xi(parent(c), c) - psi(parent(c)))`, plus
`lambda_neg` times the mean wrong-family hinge
`max(0, psi(f) + margin - xi(f, c))`. It evaluates every active class/family pair
each update, without negative subsampling; class anchors receive equal weight.

The depth loss is `mean(max(0, family_depth[parent(c)] + gap - class_depth[c])**2)`.
Depth here is tangent norm, equivalently geodesic distance from the origin.
Both anchor levels receive gradients. The existing class norm regularizer
continues to use Lorentz spatial norms. Anchor separation is the mean hinge
`max(0, geodesic_margin - Lorentz_distance)` over unique pairs. It acts **only
on family pairs** in hierarchy mode, and on class pairs otherwise. All family
hinges and the relative depth loss remain unchanged. For a single active family,
wrong-family and anchor-separation losses are zero.

`--lambda_anchor` weights the new term (CLI default 0.2), and
`--geodesic_margin` sets its distance threshold (default 3). The threshold is
independent of the angular margin and spatial norm target. `--lambda_cosine`
remains a legacy CLI alias for the new weight; it no longer enables cosine loss.
The geodesic hinge encourages distance, not guaranteed directional separation
or disjoint cones. Increasing depth can also satisfy it.

There is no image-to-family training term. Class prediction still minimizes
the class exterior angle. Checkpoints store learned family tangent vectors,
ordered family mapping, weights and cosine scope; the family diagnostic reader
prefers these vectors and retains its text fallback for legacy checkpoints.
The legacy diagnostic CLI still uses the original IAB dataset workflow.

## Depth settings

| Setting | Initial value / weight |
| --- | --- |
| `--anchor_init_norm` | 2 (class tangent depth) |
| `--init_depth` | 3 in the launcher (one-time image calibration) |
| `--family_init_norm` | 1.2 |
| `--family_depth_margin` | 0.5 |
| `--lambda_family` | 1 |
| `--lambda_family_depth` | 0.5 |
| `--lambda_anchor` | Existing CLI/launcher weight (0.2 / 0.5) |
| `--geodesic_margin` | 3 (geodesic distance, independent of the norm target) |

With curvature 1 and `min_radius=0.5`, family/class depths 1.2/2 correspond
to spatial norms 1.509/3.627 and half-apertures 41.49/16.00 degrees.
For curvature `c`, spatial norm is `sinh(sqrt(c)*depth)/sqrt(c)`.
The aperture requires tangent depth above `asinh(2*min_radius)/sqrt(c)`.
Hierarchy training projects both anchor levels to at least this boundary plus
0.001 after each update; configured class norm bounds must also respect it.
This projection preserves the aperture domain, not family containment.

The values above are starting settings, not established optima. Image depths
remain learnable after calibration. Relative depth is a soft penalty; it does
not guarantee ordering, containment or non-overlapping cones. Transitivity
applies when the actual nested-cone containment relations hold.

For a first validation comparison, use family initial depths `{1.0, 1.2, 1.4}`
crossed with gaps `{0.25, 0.5}`, keeping class/image initialization at 2/3,
weights, manifest and seed 42 fixed. Use the Python CLI to vary these values;
give all six runs distinct `--output` and `--diag_plot_dir` paths. For example:

```bash
python train_iabench.py \
  --dataset_path "$SCRATCH/datasets/IABench_images" \
  --split_manifest "$WORK/hyp_fine_tuning/checkpoints/attribution_iabench_random_vitl14.splits.json" \
  --hierarchy increfa --anchor_init random --anchor_init_norm 2 --init_depth 3 \
  --family_init_norm 1.2 --family_depth_margin 0.5 \
  --curv 1 --min_radius 0.5 --margin 0.3 --lambda_neg 1 \
  --lambda_norm 0.5 --target_norm 4 --lambda_anchor 0.5 --geodesic_margin 3 \
  --clip_name openai/clip-vit-large-patch14 \
  --lora_target 'vision_model\.encoder\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj)' \
  --lora_r 16 --lora_alpha 32 --hyperbolic_dim 128 \
  --batch_size 256 --num_epochs 20 --lr 3e-4 --weight_decay 0.01 \
  --num_workers 8 --seed 42 --log_every 10 \
  --output /path/to/family_1.2_gap_0.5.pt \
  --diag_plot_dir /path/to/family_1.2_gap_0.5
```

Rank runs by clean balanced validation accuracy. Inspect class/image containment,
wrong-family violations, family cosine, aperture ranges and radial gaps before
interpreting improvements. Evaluate the selected configuration on the harness
test split only after choosing it.

## Diagnostics and checks

`stats.csv` retains existing class statistics and adds raw family positive,
negative and depth losses, class-in-parent and image-in-parent fractions,
wrong-family margin violations, family/class/image tangent depths, minimum and
mean parent-child gaps, depth-margin violations, image/class ordering violations,
family aperture extrema, family cosine mean/max and exterior-angle saturation.
Family containment diagnostics do not contribute gradients. Epoch values average
batch statistics: reported minima/maxima are averages of batch extrema.
`loss_anchor` refers to the family penalty in hierarchy mode; `mean_anc_cos_sim`
and `max_anc_cos_sim` continue to describe classes. New family cosine columns
are `mean_fam_cos_sim` and `max_fam_cos_sim`.

```bash
python -m tests.test_family_hierarchy
python -m tests.test_phase_b
python -m tests.test_sep_stats
python -m tests.test_checkpoint_io
python -m tests.test_attribution_training
python -m tests.test_iabench_training
python -m tests.test_iabench_slurm
```

These checks use CPU synthetic features/images, without downloading CLIP weights.
GPU training and the depth comparison require the Leonardo runtime and dataset.
