#!/usr/bin/env bash
# Run from any directory. The pilot logs report times for choosing array walltimes.
set -euo pipefail

stage=${1:-}
if [[ $# != 2 || ( $stage != pilot && $stage != full ) ]]; then
    echo "Usage: bash $0 pilot|full RUN_DIRECTORY" >&2
    exit 2
fi
export REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
export OUT=$(realpath -m "$2")
launcher="$REPO/slurm/slurm_incremental_tsne.sh"
account=${ACCOUNT:-EUHPC_D33_218}
common=(--parsable --account="$account" --nodes=1 --ntasks=1 --chdir="$REPO")
gpu=(--partition=boost_usr_prod --gpus-per-node=1 --cpus-per-task=8 --mem=64G)
cpu=(--partition=lrd_all_serial --cpus-per-task=4 --mem=16G)

if [[ $stage == pilot ]]; then
    mkdir "$OUT"
    mkdir "$OUT/logs"
    bash "$launcher" preflight
    pilot=$(sbatch "${common[@]}" "${gpu[@]}" --job-name=iab_tsne_pilot \
        --time="${PILOT_TIME:-01:00:00}" --output="$OUT/logs/%x_%j.log" "$launcher" pilot)
    printf '%s\n' "$pilot" > "$OUT/logs/pilot.job"
    printf 'pilot=%s\noutput=%s\n' "$pilot" "$OUT"
    exit
fi

# Reuse the full-sized T7 benchmark. The separate smoke directory cannot be mixed in.
tasks=0-47
fit_tasks=0-5
benchmark="$OUT/split14/hypclip/T7"
if [[ ${TSNE_METHODS:-} == 'lida increfa' ]]; then
    tasks=16-47
    fit_tasks=2-5
    benchmark="$OUT/split14/increfa/T7"
elif [[ -n ${TSNE_METHODS:-} ]]; then
    echo "Use all methods or TSNE_METHODS='lida increfa' for a partial run." >&2
    exit 2
fi
test -s "$benchmark.features.npz"
test -s "$benchmark.points.npz"
# Hold the lock while submitting and recording the job IDs so a retry can continue.
exec 9>"$OUT/logs/submit.lock"
flock -n 9
logs=(--output="$OUT/logs/%x_%A_%a.log")
if [[ -s $OUT/logs/extract.job ]]; then
    read -r extract < "$OUT/logs/extract.job"
else
    extract=$(sbatch "${common[@]}" "${gpu[@]}" "${logs[@]}" --job-name=iab_tsne_extract \
        --time="${EXTRACT_TIME:-01:00:00}" --array="$tasks%${GPU_CONCURRENCY:-4}" "$launcher" extract)
    printf '%s\n' "$extract" > "$OUT/logs/extract.job"
fi
if [[ -s $OUT/logs/fit.job ]]; then
    read -r fit < "$OUT/logs/fit.job"
else
    fit=$(sbatch "${common[@]}" "${cpu[@]}" "${logs[@]}" --job-name=iab_tsne_fit \
        --time="${FIT_TIME:-04:00:00}" --array="$fit_tasks%${CPU_CONCURRENCY:-2}" \
        --dependency="afterok:${extract%%;*}" --kill-on-invalid-dep=yes "$launcher" fit)
    printf '%s\n' "$fit" > "$OUT/logs/fit.job"
fi
if [[ -s $OUT/logs/render.job ]]; then
    read -r render < "$OUT/logs/render.job"
else
    render=$(sbatch "${common[@]}" "${cpu[@]}" --output="$OUT/logs/%x_%j.log" \
        --job-name=iab_tsne_render --time=00:30:00 --dependency="afterok:${fit%%;*}" \
        --kill-on-invalid-dep=yes "$launcher" render)
    printf '%s\n' "$render" > "$OUT/logs/render.job"
fi
printf 'extract=%s\nfit=%s\nrender=%s\noutput=%s\n' "$extract" "$fit" "$render" "$OUT"
