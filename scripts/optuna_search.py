"""
Optuna hyperparameter search for the 22-class hyperbolic attribution model.

Replaces the hand-written OAT sweeps (slurm/sweep_configs_22cls*.txt). Two things
the grid could not do:
  - it searches continuously, instead of one-factor-at-a-time around a base point
    (sweep 1 found lr at the BOUNDARY of its range, which a grid cannot fix);
  - it PRUNES: a config that collapses to chance in epoch 1 (lambda_neg 2.0 did
    exactly that) is killed instead of burning the whole task budget.

Each trial is a SUBPROCESS of train_attribution.py, not an in-process call: the
trainer builds ViT-L/14 + LoRA + DataParallel inside main(), and rebuilding that
ten times in one process leaks CUDA state. The subprocess also means this file
never has to touch the trainer.

Pruning reads the trainer's existing per-epoch validation line
    "  val: overall=X%  balanced=Y%  (N samples)"
with the same regex scripts/plot_training_log.py already uses. Nothing new is
logged by the trainer.

Trials do NOT keep their checkpoint (--output goes to $TMPDIR): lora_state is the
whole PEFT-wrapped CLIP, ~1.7 GB, and the winner gets retrained anyway. Optuna
keeps the hyperparameters, which is all that is needed.

Storage is a journal FILE with symlink locking, not SQLite: SQLite's locking is
unreliable on Lustre.

Usage:
    # one worker (the slurm array runs four against the same journal)
    python -m scripts.optuna_search --storage $WORK/hyp_fine_tuning/optuna/hypclip_22cls.log

    # ranking, any time, from the login node
    python -m scripts.optuna_search --storage ... --report
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Same line the trainer prints at the end of every epoch.
VAL_RE = re.compile(r"^\s*val:\s*overall=([0-9.]+)%\s+balanced=([0-9.]+)%")

GENERATORS = [
    "real", "4o", "CogView3_PLUS", "FLUX", "KANDINSKY", "PIXART", "PLAYGROUND_2_5",
    "SD1_5", "SD2_1", "SD3", "SD3_5", "SDXL", "gemini", "grok3", "hidream", "hunyuan",
    "ideogram", "infinity", "janus-pro", "kling", "mid-5.2", "mid-6.0",
]
SEMANTICS = ["COCO", "cat", "dog", "wild", "FFHQ", "celebahq",
             "bedroom", "church", "classroom", "ImageNet-1k"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--storage", required=True,
                   help="Path to the Optuna journal file (shared by all workers).")
    p.add_argument("--study_name", default="hypclip_22cls")
    p.add_argument("--report", action="store_true",
                   help="Print the ranking and exit — no training.")
    p.add_argument("--n_trials", type=int, default=100,
                   help="Upper bound per worker; --timeout is what actually stops it.")
    p.add_argument("--timeout", type=float, default=None,
                   help="Seconds. No NEW trial starts after this; running ones finish.")
    # Everything below is fixed across trials (parity with the baselines).
    p.add_argument("--dataset_path", default=None)
    p.add_argument("--captions_dir", default=None)
    p.add_argument("--split_manifest", default=None)
    p.add_argument("--clip_name", default="openai/clip-vit-large-patch14")
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--trial_ckpt", default=None,
                   help="Scratch path for the throwaway per-trial checkpoint "
                        "(default $TMPDIR/optuna_trial.pt).")
    return p.parse_args()


def suggest(trial):
    """The search space. Ranges are informed by sweeps 1 and 2 — see the plan."""
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lambda_norm = trial.suggest_categorical("lambda_norm", [0.0, 0.25, 0.5, 1.0])
    cfg = [
        "--lr",             f"{trial.suggest_float('lr', 5e-5, 1e-3, log=True):.6g}",
        "--lora_r",         str(lora_r),
        "--lora_alpha",     str(2 * lora_r),
        "--hyperbolic_dim", str(trial.suggest_categorical("hyperbolic_dim", [64, 128, 256])),
        "--curv",           f"{trial.suggest_float('curv', 0.5, 2.0, log=True):.4g}",
        "--min_radius",     f"{trial.suggest_float('min_radius', 0.1, 1.0):.4g}",
        "--margin",         f"{trial.suggest_float('margin', 0.05, 0.5):.4g}",
        # lambda_neg 2.0 collapses training to chance (measured in sweep 1) — excluded.
        "--lambda_neg",     f"{trial.suggest_float('lambda_neg', 0.5, 1.5):.4g}",
        "--lambda_norm",    str(lambda_norm),
        "--weight_decay",   f"{trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True):.4g}",
    ]
    # target_norm only exists when the regulariser is on.
    if lambda_norm > 0:
        cfg += ["--target_norm", f"{trial.suggest_float('target_norm', 2.0, 6.0):.4g}"]
    return cfg


def run_trial(trial, args):
    """Train one config in a subprocess, reporting val_balanced after every epoch."""
    import optuna

    cmd = [
        sys.executable, "train_attribution.py",
        "--dataset_path", args.dataset_path,
        "--captions_dir", args.captions_dir,
        "--generators", *GENERATORS,
        "--semantics", *SEMANTICS,
        "--clip_name", args.clip_name,
        "--anchor_init", "text",
        "--no_captions",
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
        "--num_workers", str(args.num_workers),
        "--split_manifest", args.split_manifest,
        "--output", args.trial_ckpt,
        *suggest(trial),
    ]
    print(f"\n=== trial {trial.number} ===\n{' '.join(cmd)}\n", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    best, epoch = -1.0, 0
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)      # keep the SLURM .out readable
            m = VAL_RE.match(line)
            if not m:
                continue
            epoch += 1
            balanced = float(m.group(2)) / 100.0
            best = max(best, balanced)
            trial.report(balanced, epoch)
            if trial.should_prune():
                proc.terminate()
                proc.wait(timeout=60)
                raise optuna.TrialPruned(
                    f"pruned at epoch {epoch}, balanced={balanced:.4f}")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"train_attribution.py exited {proc.returncode}")
    if best < 0:
        raise RuntimeError("no validation line was ever printed")
    return best


def report(study):
    """Ranking without pandas — trials_dataframe() needs it and the venv may not."""
    from collections import Counter

    trials = study.get_trials(deepcopy=False)
    if not trials:
        print("no trials yet")
        return
    print("  ".join(f"{k.name}={v}" for k, v in
                    sorted(Counter(t.state for t in trials).items(),
                           key=lambda kv: kv[0].name)))

    done = sorted((t for t in trials if t.value is not None),
                  key=lambda t: t.value, reverse=True)
    print(f"\n--- {len(done)} trials with a value, best first ---")
    for t in done:
        mins = (t.duration.total_seconds() / 60) if t.duration else 0
        params = "  ".join(f"{k}={v}" for k, v in sorted(t.params.items()))
        print(f"  {t.value:.4f}  #{t.number:<3d} {t.state.name:<9s} {mins:5.0f}m  {params}")
    if study.best_trial.value is not None:
        print(f"\nBEST value={study.best_value:.4f}  trial={study.best_trial.number}")
        for k, v in sorted(study.best_params.items()):
            print(f"  {k:16s} {v}")


def main():
    args = parse_args()
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock

    path = Path(args.storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Symlink locking: works on Lustre/NFS, unlike SQLite's byte-range locks.
    storage = JournalStorage(JournalFileBackend(
        str(path), lock_obj=JournalFileSymlinkLock(str(path))))

    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    if args.report:
        report(study)
        return

    for name in ("dataset_path", "captions_dir", "split_manifest"):
        if getattr(args, name) is None:
            sys.exit(f"--{name} is required when training")
    if args.trial_ckpt is None:
        args.trial_ckpt = str(Path(os.environ.get("TMPDIR", "/tmp")) / "optuna_trial.pt")

    t0 = time.time()
    study.optimize(lambda t: run_trial(t, args), n_trials=args.n_trials,
                   timeout=args.timeout, catch=(RuntimeError,))
    print(f"\nworker done after {(time.time() - t0) / 3600:.1f}h")
    report(study)


if __name__ == "__main__":
    main()
