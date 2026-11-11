"""CPU and shell checks: python -m tests.test_iab_anchor_sweep."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from comparison.training.scripts.collect_iab_anchor_sweep import CONFIGS, collect, main
from training.attribution_args import parse_args, validate_args


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "slurm/slurm_sweep_iab_omnidfa.sh"
SHELL = r'''
module() { :; }
python() { "$SWEEP_TEST_PYTHON" "$SWEEP_TEST_STUB" "$@"; }
export -f module python
exec bash "$1"
'''
STUB = '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ["SWEEP_TEST_CAPTURE"]).open("a") as stream:
    stream.write(json.dumps({"argv": args, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                            "excluded": os.environ["IAB_EXCLUDE_GENERATORS"]}) + "\\n")
stage = "train" if args[0] == "train_attribution.py" else args[args.index("--level_start") + 1]
if stage == os.environ.get("SWEEP_TEST_FAIL"):
    sys.exit(7)
if stage == "train":
    Path(args[args.index("--output") + 1]).touch()
'''


class IABAnchorSweepTests(unittest.TestCase):
    def test_all_array_tasks_and_failure_propagation(self):
        self.assertEqual(CONFIGS, ((.5, 1.5, 1.5), (.5, 2, 2), (.5, 2.5, 2.5)))
        with tempfile.TemporaryDirectory(prefix="iab sweep ") as tmp:
            root = Path(tmp)
            base = root / "hyp_fine_tuning"
            (base / "bin").mkdir(parents=True)
            (base / "bin/activate").touch()
            (base / "split_manifest_22cls.json").touch()
            stub = root / "stub.py"
            stub.write_text(STUB)
            capture = root / "calls.jsonl"
            env = dict(os.environ, WORK=tmp, FAST=str(root / "fast cache"), REPO=str(REPO),
                       SLURM_ARRAY_JOB_ID="100", SLURM_JOB_ID="101", CUDA_VISIBLE_DEVICES="4,5",
                       SWEEP_TEST_PYTHON=sys.executable, SWEEP_TEST_STUB=str(stub),
                       SWEEP_TEST_CAPTURE=str(capture), CKPT="must-not-be-used.pt",
                       NUM_EPOCHS="1", AUGMENT="0", AUG_POLICY="corruption", OPTIMIZER="sgd",
                       ANCHOR_INIT="text", HIERARCHY="increfa", LAMBDA_NORM="99",
                       LAMBDA_ANCHOR="99", TARGET_NORM="99", GEODESIC_MARGIN="99")
            for key in ("DATA", "CAPS", "OUT", "MANIFEST", "DIAG_PLOT_DIR", "SWEEP_TEST_FAIL"):
                env.pop(key, None)
            outputs = set()
            for index, (weight, z, tau) in enumerate(CONFIGS):
                with self.subTest(index=index):
                    capture.unlink(missing_ok=True)
                    subprocess.run(["bash", "-c", SHELL, "_", str(SCRIPT)], check=True,
                                   env=dict(env, SLURM_ARRAY_TASK_ID=str(index)), capture_output=True, text=True)
                    calls = [json.loads(line) for line in capture.read_text().splitlines()]
                    self.assertEqual(len(calls), 4)
                    args = parse_args(calls[0]["argv"][1:])
                    validate_args(args)
                    self.assertEqual((args.lambda_anchor, args.target_norm), (weight, tau))
                    self.assertEqual(args.hierarchy, "none")
                    self.assertNotIn("--geodesic_margin", calls[0]["argv"])
                    self.assertEqual((args.num_epochs, args.lambda_neg, args.lambda_norm, args.norm_mode),
                                     (30, 1, .5, "floor"))
                    self.assertEqual((args.batch_size, args.seed, args.optimizer, args.lr, args.lr_schedule),
                                     (256, 42, "adamw", 3e-4, "cosine"))
                    self.assertEqual((args.anchor_init, args.train_augment, args.aug_policy),
                                     ("random", True, "omnidfa"))
                    self.assertEqual(args.generators[0], "real")
                    self.assertEqual(len(args.generators), 22)
                    self.assertNotIn("dalle3", args.generators)
                    folder = base / f"checkpoints/sweep/iab_omnidfa_100/task_{index:02d}"
                    self.assertEqual(Path(args.output).parent, folder)
                    tau_tag = str(tau).replace(".", "p")
                    z_tag = format(z, "g").replace(".", "p")
                    self.assertIn(f"_z{z_tag}_tau{tau_tag}_", Path(args.output).name)
                    self.assertEqual(args.diag_plot_dir, str(folder / "diagnostics"))
                    self.assertEqual(args.split_manifest, str(base / "split_manifest_22cls.json"))
                    self.assertEqual((folder / "status.txt").read_text().strip(), "complete")
                    outputs.add(args.output)
                    for call, level in zip(calls[1:], (0, 4, 6)):
                        self.assertEqual(call["argv"][:2], ["-m", "comparison.training.test_hypclip"])
                        flags = dict(zip(call["argv"][2::2], call["argv"][3::2]))
                        self.assertEqual(flags["--checkpoint"], args.output)
                        self.assertEqual(flags["--root_dir"], args.dataset_path)
                        self.assertEqual((flags["--level_start"], flags["--level_end"]),
                                         (str(level), str(level + 1)))
                        self.assertEqual(flags["--pre_resize"], "0")
                        self.assertEqual(flags["--log_dir"], str(folder / "evaluation"))
                    self.assertTrue(all(call["gpu"] == "4,5" and call["excluded"] == "dalle3"
                                        for call in calls))
            self.assertEqual(len(outputs), 3)
            # A second submission has its own outputs, even for the same configuration.
            for job, failure, count in (("200", "train", 1), ("300", "4", 3)):
                capture.unlink()
                result = subprocess.run(["bash", "-c", SHELL, "_", str(SCRIPT)],
                    env=dict(env, SLURM_ARRAY_TASK_ID="0", SLURM_ARRAY_JOB_ID=job,
                             SWEEP_TEST_FAIL=failure), capture_output=True, text=True)
                self.assertEqual(result.returncode, 7)
                self.assertEqual(len(capture.read_text().splitlines()), count)
                folder = base / f"checkpoints/sweep/iab_omnidfa_{job}/task_00"
                self.assertEqual((folder / "status.txt").read_text().strip(), "failed (exit 7)")
            capture.unlink()
            duplicate = subprocess.run(["bash", "-c", SHELL, "_", str(SCRIPT)],
                env=dict(env, SLURM_ARRAY_TASK_ID="0"), capture_output=True, text=True)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertFalse(capture.exists())
            folder = base / "checkpoints/sweep/iab_omnidfa_100/task_00"
            self.assertEqual((folder / "status.txt").read_text().strip(), "complete")
            for task in ("-1", "3", "oops", "01"):
                invalid = subprocess.run(["bash", "-c", SHELL, "_", str(SCRIPT)],
                    env=dict(env, SLURM_ARRAY_TASK_ID=task), capture_output=True, text=True)
                self.assertEqual(invalid.returncode, 2)
                self.assertFalse(capture.exists())

    def make_results(self, root):
        for index, (weight, z, tau) in enumerate(CONFIGS):
            folder = root / f"task_{index:02d}"
            (folder / "evaluation").mkdir(parents=True)
            (folder / "status.txt").write_text("complete\n")
            checkpoint = dict(lambda_anchor=weight, geodesic_margin=z, target_norm=tau,
                              num_epochs=30, train_augment=True, aug_policy="omnidfa",
                              anchor_init="random", hierarchy="none", lambda_norm=.5,
                              norm_mode="floor", norm_space="hyperbolic_depth",
                              anchor_penalty="geodesic", anchor_reduction="mean",
                              epoch=4, val_balanced=.9 - index / 100)
            torch.save(checkpoint, folder / "attribution_22cls_fixture.pt")
            for level in (0, 4, 6):
                # Test ranking is deliberately opposite to validation ranking.
                (folder / f"evaluation/test_results_degraded_{level}.txt").write_text(
                    f"acc: {index / 100}\nbalanced_accuracy: {index / 100 + .01}\n")

    def test_complete_report_ranks_validation_and_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_results(root)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--dir", tmp]), 0)
            with (root / "summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual([int(row["task"]) for row in rows], list(range(3)))
            self.assertTrue(all(row["status"] == "complete" for row in rows))
            self.assertEqual(float(rows[0]["clean_acc"]), 0)
            self.assertEqual(float(rows[0]["blur5_balanced_accuracy"]), .01)
            self.assertIn("Best clean validation configuration: task 0", output.getvalue())
            self.assertIn("overall / balanced", output.getvalue())

    def test_missing_failed_and_malformed_runs_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_results(root)
            (root / "task_00/status.txt").write_text("failed (exit 7)")
            (root / "task_01").rename(root / "removed_task")
            (root / "task_02/evaluation/test_results_degraded_0.txt").write_text(
                "acc: .5\nacc: .6\nbalanced_accuracy: .5\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--dir", tmp]), 1)
            rows = {row["task"]: row for row in collect(root)}
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(rows[1]["status"], "missing")
            self.assertEqual(rows[2]["status"], "invalid")
            self.assertTrue(all(row["error"] for row in rows.values()))
            self.assertIn("Incomplete sweep: 3/3", output.getvalue())
            self.assertNotIn("Best clean validation configuration", output.getvalue())

    def test_rejects_checkpoints_from_older_grids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_results(root)
            for index, (key, value) in enumerate((
                    ("target_norm", 2), ("norm_space", "lorentz_spatial"), ("geodesic_margin", 4))):
                path = root / f"task_{index:02d}/attribution_22cls_fixture.pt"
                checkpoint = torch.load(path, weights_only=False)
                torch.save(dict(checkpoint, **{key: value}), path)
            rows = collect(root)
            self.assertTrue(all(row["status"] == "invalid" for row in rows))


if __name__ == "__main__":
    unittest.main()
