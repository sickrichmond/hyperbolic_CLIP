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
            checkpoint = checkpoints / "attribution_iabench_random_aug_vitl14.pt"
            manifest = checkpoints / "attribution_iabench_random_vitl14.splits.json"
            checkpoint.touch()
            manifest.touch()
            capture = work / "command.json"
            env = dict(os.environ, WORK=str(work), SLURM_JOB_ID="123",
                       CUDA_VISIBLE_DEVICES="4", SLURM_TEST_PYTHON=sys.executable,
                       SLURM_TEST_CAPTURE=str(capture))
            for name in ("CKPT", "LOGDIR", "SPLIT_MANIFEST", "NUM_WORKERS", "PROFILE_STEPS",
                         "AUGMENT", "AUG_POLICY", "DATA"):
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
                     "_omniaug", "omnidfa")):
                with self.subTest(settings=settings):
                    subprocess.run(["bash", "-c", stub, "_", str(train_script)],
                                   env=dict(env, **settings), check=True, capture_output=True, text=True)
                    command = json.loads(capture.read_text())
                    self.assertEqual(command["argv"][0], "train_iabench.py")
                    args = parse_args(command["argv"][1:], dataset="iabench")
                    validate_args(args)
                    self.assertEqual(args.train_augment, bool(suffix))
                    self.assertEqual(args.aug_policy, policy)
                    self.assertEqual(args.output, str(
                        checkpoints / f"attribution_iabench_random{suffix}_vitl14.pt"))
                    self.assertEqual(args.split_manifest, str(manifest))
                    self.assertEqual(args.dataset_path, settings.get(
                        "DATA", "/leonardo_scratch/large/userexternal/imaljkov/datasets/IABench/data"))
                    self.assertEqual((args.val_frac, args.test_frac, args.seed), (0.1, 0.1, 42))
                    self.assertEqual(command["gpu"], "4")
            for settings in ({"AUGMENT": "2"}, {"AUG_POLICY": "unknown"}):
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
                    subprocess.run(["bash", "-c", stub, "_", str(eval_script)], env=env,
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
                        "DATA", "/leonardo_scratch/large/userexternal/imaljkov/datasets/IABench/data"))
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


if __name__ == "__main__":
    unittest.main()
