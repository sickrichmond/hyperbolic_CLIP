"""Run with python -m unittest tests.test_checkpoint_io (CPU only)."""
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from checkpoint_io import atomic_torch_save


class CheckpointSaveTest(unittest.TestCase):
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
