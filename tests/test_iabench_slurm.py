"""Launcher smoke check without Slurm, GPUs or model downloads.

Run: python -m tests.test_iabench_slurm
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from training.attribution_args import parse_args, validate_args


class IABenchSlurmTests(unittest.TestCase):
    def test_training_modes_and_test_evaluation(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="iabench slurm ") as directory:
            work = Path(directory)
            base = work / "hyp_fine_tuning"
            (base / "bin").mkdir(parents=True)
            (base / "bin/activate").touch()
            (base / "hyperbolic_CLIP_riccardo").mkdir()
            checkpoints = base / "checkpoints"
            checkpoints.mkdir()
            checkpoint = checkpoints / "attribution_iabench_random_anchor0p5_z4_ep30_aug_vitl14.pt"
            manifest = checkpoints / "attribution_iabench_random_vitl14.splits.json"
            checkpoint.touch()
            manifest.touch()
            capture = work / "command.json"
            scratch = work / "scratch storage"
            env = dict(os.environ, WORK=str(work), SCRATCH=str(scratch), SLURM_JOB_ID="123",
                       CUDA_VISIBLE_DEVICES="4", SLURM_TEST_PYTHON=sys.executable,
                       SLURM_TEST_CAPTURE=str(capture))
            for name in ("CKPT", "LOGDIR", "SPLIT_MANIFEST", "NUM_WORKERS", "PROFILE_STEPS",
                         "AUGMENT", "AUG_POLICY", "DATA", "HIERARCHY",
                         "LAMBDA_ANCHOR", "TARGET_NORM", "GEODESIC_MARGIN", "NUM_EPOCHS"):
                env.pop(name, None)
            stub = r"""
module() { :; }
python() {
    "$SLURM_TEST_PYTHON" -c 'import json, os, sys; from pathlib import Path; Path(os.environ["SLURM_TEST_CAPTURE"]).write_text(json.dumps({"argv": sys.argv[1:], "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}))' "$@"
}
export -f module python
exec bash "$1"
"""
            train_script = repo / "slurm/slurm_train_iabench.sh"
            for settings, suffix, policy in (({}, "", "corruption"),
                    ({"AUGMENT": "1"}, "_aug", "corruption"),
                    ({"AUGMENT": "1", "AUG_POLICY": "omnidfa", "DATA": str(work / "image cache")},
                     "_omniaug", "omnidfa"),
                    ({"HIERARCHY": "increfa"}, "_hierarchy_increfa", "corruption"),
                    ({"HIERARCHY": "increfa", "AUGMENT": "1"},
                     "_aug_hierarchy_increfa", "corruption"),
                    ({"TARGET_NORM": "2.5", "LAMBDA_ANCHOR": "0.3", "NUM_EPOCHS": "20"},
                     "", "corruption"),
                    ({"HIERARCHY": "increfa", "AUGMENT": "1", "TARGET_NORM": "1.5",
                      "LAMBDA_ANCHOR": "0", "NUM_EPOCHS": "40"},
                     "_aug_hierarchy_increfa", "corruption"),
                    ({"HIERARCHY": "increfa", "TARGET_NORM": "2.1234567"},
                     "_hierarchy_increfa", "corruption"),
                    ({"CKPT": str(checkpoints / "mean checkpoint.pt"),
                      "HIERARCHY": "increfa", "AUGMENT": "1"},
                     "_aug_hierarchy_increfa", "corruption")):
                with self.subTest(settings=settings):
                    completed = subprocess.run(["bash", "-c", stub, "_", str(train_script)],
                                   env=dict(env, **settings), check=True, capture_output=True, text=True)
                    command = json.loads(capture.read_text())
                    self.assertEqual(command["argv"][0], "train_iabench.py")
                    args = parse_args(command["argv"][1:], dataset="iabench")
                    validate_args(args)
                    self.assertIn("--lambda_anchor", command["argv"])
                    self.assertIn("--geodesic_margin", command["argv"])
                    self.assertNotIn("--lambda_cosine", command["argv"])
                    self.assertEqual(args.lambda_anchor, float(settings.get("LAMBDA_ANCHOR", .5)))
                    self.assertEqual(args.target_norm, float(settings.get("TARGET_NORM", 4)))
                    self.assertEqual(args.geodesic_margin,
                                     args.target_norm / (2 if args.hierarchy != "none" else 1))
                    self.assertEqual(args.num_epochs, int(settings.get("NUM_EPOCHS", 30)))
                    self.assertEqual(args.train_augment, settings.get("AUGMENT") == "1")
                    self.assertEqual(args.hierarchy, settings.get("HIERARCHY", "none"))
                    self.assertEqual(args.aug_policy, policy)
                    weight_tag = settings.get("LAMBDA_ANCHOR", "0.5").replace(".", "p")
                    margin_tag = format(args.geodesic_margin, ".17g").replace(".", "p")
                    self.assertEqual(args.output, settings.get("CKPT", str(checkpoints /
                        f"attribution_iabench_random_anchor{weight_tag}_z{margin_tag}_ep{args.num_epochs}{suffix}_vitl14.pt")))
                    self.assertEqual(args.split_manifest, str(manifest))
                    self.assertEqual(args.diag_plot_dir, str(
                        base / f"viz/iabench{suffix}_123"))
                    self.assertEqual(args.dataset_path, settings.get(
                        "DATA", str(scratch / "datasets/IABench_images")))
                    self.assertIn(f"Dataset: {args.dataset_path}", completed.stdout)
                    margin_line = next(line for line in completed.stdout.splitlines()
                                       if line.startswith("Geodesic anchor separation:"))
                    self.assertEqual(float(margin_line.split("geodesic_margin=")[1]),
                                     args.geodesic_margin)
                    self.assertEqual((args.val_frac, args.test_frac, args.seed), (0.1, 0.1, 42))
                    self.assertEqual(command["gpu"], "4")
            for settings in ({"AUGMENT": "2"}, {"AUG_POLICY": "unknown"},
                             {"HIERARCHY": "unknown"}, {"GEODESIC_MARGIN": "-1"},
                             {"GEODESIC_MARGIN": "nan"}, {"LAMBDA_ANCHOR": "-1"},
                             {"LAMBDA_ANCHOR": "oops"}, {"TARGET_NORM": "-1"},
                             {"TARGET_NORM": "nan"}, {"NUM_EPOCHS": "0"},
                             {"NUM_EPOCHS": "-1"}, {"NUM_EPOCHS": "2.5"}):
                capture.unlink(missing_ok=True)
                failed = subprocess.run(["bash", "-c", stub, "_", str(train_script)],
                                        env=dict(env, **settings), capture_output=True, text=True)
                self.assertEqual(failed.returncode, 2)
                self.assertFalse(capture.exists())

            eval_script = repo / "slurm/slurm_eval_iabench.sh"
            for override in (False, True):
                with self.subTest(override=override):
                    if override:
                        env.update(CKPT=str(checkpoints / "clean checkpoint.pt"),
                                   LOGDIR=str(work / "evaluation results"),
                                   SPLIT_MANIFEST=str(manifest), NUM_WORKERS="4",
                                   DATA=str(work / "image cache"))
                        Path(env["CKPT"]).touch()
                    completed = subprocess.run(["bash", "-c", stub, "_", str(eval_script)], env=env,
                                   check=True, capture_output=True, text=True)
                    command = json.loads(capture.read_text())
                    self.assertEqual(command["argv"][:2], ["-m", "comparison.training.test_hypclip"])
                    flags = dict(zip(command["argv"][2::2], command["argv"][3::2]))
                    self.assertEqual(flags["--dataset"], "iabench")
                    self.assertEqual(flags["--checkpoint"], env.get("CKPT", str(checkpoint)))
                    self.assertEqual(flags["--split_manifest"], str(manifest))
                    self.assertEqual((flags["--level_start"], flags["--level_end"]), ("0", "7"))
                    self.assertEqual(flags["--num_workers"], "4" if override else "8")
                    self.assertEqual(flags["--root_dir"], env.get(
                        "DATA", str(scratch / "datasets/IABench_images")))
                    self.assertIn(f"Dataset: {flags['--root_dir']}", completed.stdout)
                    self.assertEqual(flags["--log_dir"], env.get(
                        "LOGDIR", str(work / "outputs/hypclip_iabench_123")))
                    self.assertEqual(command["gpu"], "4")
            capture.unlink()
            Path(env["CKPT"]).unlink()
            failed = subprocess.run(["bash", "-c", stub, "_", str(eval_script)], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("checkpoint not found", failed.stdout)
            self.assertFalse(capture.exists())


    def test_iab_checkpoint_name_and_epochs(self):
        script = Path(__file__).resolve().parents[1] / "slurm/slurm_train_22cls_final_loss.sh"
        stub = r"""
module() { :; }
python() {
    "$SLURM_TEST_PYTHON" -c 'import json, os, sys; from pathlib import Path; Path(os.environ["SLURM_TEST_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))' "$@"
}
export -f module python
exec bash "$1"
"""
        with tempfile.TemporaryDirectory(prefix="iab_slurm_") as directory:
            root = Path(directory)
            base = root / "hyp_fine_tuning"
            (base / "bin").mkdir(parents=True)
            (base / "bin/activate").touch()
            (base / "hyperbolic_CLIP_riccardo").mkdir()
            (base / "split_manifest_22cls.json").touch()
            capture = root / "command.json"
            env = dict(os.environ, WORK=str(root), FAST=str(root / "fast"), SLURM_JOB_ID="123",
                       SLURM_TEST_PYTHON=sys.executable, SLURM_TEST_CAPTURE=str(capture))
            for key in ("CKPT", "AUGMENT", "AUG_POLICY", "OPTIMIZER", "ANCHOR_INIT",
                        "LAMBDA_ANCHOR", "GEODESIC_MARGIN", "TARGET_NORM", "NUM_EPOCHS"):
                env.pop(key, None)
            for settings, suffix in (({}, ""), ({"AUGMENT": "1"}, "_aug"),
                    ({"AUGMENT": "1", "AUG_POLICY": "omnidfa"}, "_omniaug"),
                    ({"LAMBDA_ANCHOR": "0.2", "TARGET_NORM": "3", "NUM_EPOCHS": "20"}, ""),
                    ({"CKPT": str(base / "mean checkpoint.pt")}, ""),
                    ({"CKPT": str(base / "mean aug checkpoint.pt"), "AUGMENT": "1"}, "_aug"),
                    ({"CKPT": str(base / "mean omni checkpoint.pt"), "AUGMENT": "1",
                      "AUG_POLICY": "omnidfa"}, "_omniaug")):
                with self.subTest(settings=settings):
                    subprocess.run(["bash", "-c", stub, "_", str(script)],
                        env=dict(env, **settings), check=True, capture_output=True, text=True)
                    argv = json.loads(capture.read_text())
                    self.assertEqual(argv[0], "train_attribution.py")
                    args = parse_args(argv[1:])
                    validate_args(args)
                    self.assertEqual(args.num_epochs, int(settings.get("NUM_EPOCHS", 30)))
                    weight = settings.get("LAMBDA_ANCHOR", "0.5").replace(".", "p")
                    margin = settings.get("TARGET_NORM", "4").replace(".", "p")
                    if "CKPT" in settings:
                        self.assertEqual(args.output, settings["CKPT"])
                    else:
                        self.assertEqual(Path(args.output).name,
                            f"attribution_22cls_anchor{weight}_z{margin}_random_adamw_ep{args.num_epochs}{suffix}_vitl14.pt")
            for epochs in ("0", "-1", "2.5", "oops"):
                with self.subTest(epochs=epochs):
                    capture.unlink(missing_ok=True)
                    result = subprocess.run(["bash", "-c", stub, "_", str(script)],
                        env=dict(env, NUM_EPOCHS=epochs), capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse(capture.exists())


if __name__ == "__main__":
    unittest.main()
