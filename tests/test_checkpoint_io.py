"""Run with python -m unittest tests.test_checkpoint_io (CPU only)."""
import errno
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from checkpoint_io import atomic_torch_save, attribution_checkpoint_output


class CheckpointSaveTest(unittest.TestCase):
    def test_attribution_modes_cannot_overwrite_each_other(self):
        modes = [
            dict(train_augment=False, aug_policy="corruption", hierarchy="none"),
            dict(train_augment=True, aug_policy="corruption", hierarchy="none"),
            dict(train_augment=True, aug_policy="omnidfa", hierarchy="none"),
            dict(train_augment=False, aug_policy="corruption", hierarchy="increfa"),
            dict(train_augment=True, aug_policy="omnidfa", hierarchy="increfa"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "best.pt"
            for saved_mode in modes:
                atomic_torch_save(saved_mode, destination)
                original = destination.read_bytes()
                for requested_mode in modes:
                    with self.subTest(saved=saved_mode, requested=requested_mode):
                        if requested_mode == saved_mode:
                            with attribution_checkpoint_output(destination, **requested_mode):
                                pass
                        else:
                            with self.assertRaisesRegex(ValueError, "different CKPT/--output"):
                                with attribution_checkpoint_output(destination, **requested_mode):
                                    self.fail("Incompatible output was accepted")
                        self.assertEqual(destination.read_bytes(), original)

            atomic_torch_save({}, destination)
            with attribution_checkpoint_output(destination, **dict(modes[0], aug_policy="omnidfa")):
                pass  # An inactive policy does not change clean training.

    def test_attribution_output_lock_lifetime(self):
        mode = dict(train_augment=False, aug_policy="corruption", hierarchy="none")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "best.pt"
            lock_path = destination.with_name(f".{destination.name}.lock")
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with attribution_checkpoint_output(destination, **mode):
                    self.assertFalse(destination.exists())
                    # A separate process must be excluded before the first save, too.
                    probe = subprocess.run([
                        sys.executable, "-c",
                        "import fcntl, sys; "
                        "lock = open(sys.argv[1], 'a'); "
                        "fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)",
                        str(lock_path),
                    ], capture_output=True, text=True)
                    self.assertNotEqual(probe.returncode, 0)
                    self.assertIn("BlockingIOError", probe.stderr)
                    with self.assertRaisesRegex(RuntimeError, "Another training run"):
                        with attribution_checkpoint_output(destination, **mode):
                            self.fail("Concurrent output was accepted")
                    atomic_torch_save(mode, destination)
                    with self.assertRaisesRegex(RuntimeError, "Another training run"):
                        with attribution_checkpoint_output(destination, **mode):
                            self.fail("Atomic replacement released the run lock")
                    raise RuntimeError("training failed")
            with attribution_checkpoint_output(destination, **mode):
                pass  # A failed run releases the lock without deleting its file.

    def test_round_trip_and_failed_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "best.pt"
            payload = {"epoch": 1, "weights": torch.arange(8)}
            atomic_torch_save(payload, destination)
            loaded = torch.load(destination, weights_only=True)
            self.assertTrue(torch.equal(loaded["weights"], payload["weights"]))
            original = destination.read_bytes()

            def interrupted_save(payload, stream):
                stream.write(b"partial checkpoint")
                raise RuntimeError("PytorchStreamWriter failed writing file data/1")

            failures = [
                patch("checkpoint_io.torch.save", side_effect=interrupted_save),
                patch("checkpoint_io.os.fsync", side_effect=OSError(errno.ENOSPC, "full")),
                patch("checkpoint_io.os.replace", side_effect=OSError(errno.EIO, "I/O error")),
            ]
            for failure in failures:
                with failure, self.assertRaisesRegex(RuntimeError, "quota"):
                    atomic_torch_save({"epoch": 2}, destination)
                self.assertEqual(destination.read_bytes(), original)
                self.assertEqual(list(Path(directory).iterdir()), [destination])

            missing = Path(directory) / "first.pt"
            with patch("checkpoint_io.torch.save", side_effect=interrupted_save):
                with self.assertRaises(RuntimeError):
                    atomic_torch_save(payload, missing)
            self.assertFalse(missing.exists())
            self.assertEqual(list(Path(directory).iterdir()), [destination])

            atomic_torch_save({"epoch": 3}, destination)
            self.assertEqual(torch.load(destination, weights_only=True)["epoch"], 3)
            self.assertEqual(list(Path(directory).iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
