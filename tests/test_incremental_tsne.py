"""CPU checks for the incremental IAB t-SNE workflow."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from explanation import incremental_tsne as tsne


class IncrementalTSNETests(unittest.TestCase):
    def test_schedule_checkpoints_and_colors(self):
        self.assertEqual(len(tsne.PANELS), 48)
        fast, igor = Path("/fast"), Path("/igor")
        self.assertEqual(tsne.checkpoint_path("hypclip", "seq7", 0, fast, igor),
                         fast / "continual_runs_hyp/base7_hyp_clean/T0.pt")
        self.assertEqual(tsne.checkpoint_path("hypclip", "split14", 7, fast, igor),
                         fast / "continual_runs_hyp/split14_hyp_clean_r300_e10_replay/T7.pt")
        self.assertEqual(tsne.checkpoint_path("lida", "split14", 0, fast, igor),
                         fast / "lida_runs/lida_a10_db300_split14_base/T0.pt")
        self.assertEqual(tsne.checkpoint_path("increfa", "seq7", 3, fast, igor),
                         igor / "increfa_comparison/outputs/increfa_iab_seq7_buf300/task_3.pt")
        palette = tsne.permanent_colors()
        self.assertEqual(len(set(palette.values())), 21)
        for setting, base in tsne.BASE.items():
            for step in tsne.STEPS:
                names = tsne.seen_classes(setting, step)
                self.assertEqual(len(names), len(base) + step)
                colors = tsne.panel_colors(setting, step, palette)
                self.assertEqual(colors["real"], "#808080")
                self.assertEqual([name for name in colors if colors[name] == "#000000"],
                                 [tsne.INCREMENTS[step - 1]] if step else [])
                if step:
                    self.assertEqual(list(colors)[-1], tsne.INCREMENTS[step - 1])
                if step >= 2:
                    old = tsne.INCREMENTS[step - 2]
                    self.assertEqual(colors[old], palette[old])

    def test_metadata_uses_saved_indices_and_validates_the_stage(self):
        names = tsne.seen_classes("seq7", 1)
        reversed_names = names[::-1]
        self.assertEqual(tsne.checkpoint_classes({"class_names": reversed_names, "step": 1},
                                                "lida", "seq7", 1), reversed_names)
        with self.assertRaisesRegex(ValueError, "checkpoint step"):
            tsne.checkpoint_classes({"class_names": names, "cl_step": 2}, "hypclip", "seq7", 1)
        checkpoint = {"results": {"class_order": [list(names[:-1]), [names[-1]]]},
                      "task_id": 1, "num_learned_classes": len(names)}
        self.assertEqual(tsne.checkpoint_classes(checkpoint, "increfa", "seq7", 1), names)
        checkpoint["num_learned_classes"] -= 1
        with self.assertRaisesRegex(ValueError, "task/count"):
            tsne.checkpoint_classes(checkpoint, "increfa", "seq7", 1)

    def test_manifest_rows_and_atomic_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            rows = [f"{name}/{index}.png" for name in tsne.seen_classes("split14", 7)
                    for index in range(5)]
            manifest.write_text(json.dumps({"train": ["real/train.png"], "test": rows[::-1]}))
            ordered, digest = tsne.read_test_rows(manifest)
            selected = tsne.panel_rows(ordered, "seq7", 0)
            self.assertEqual(len(selected), 35)
            self.assertEqual(selected, sorted(selected))
            self.assertNotIn("real/train.png", selected)
            with self.assertRaisesRegex(ValueError, "zero|no test rows"):
                tsne.panel_rows([row for row in ordered if tsne.generator(row) != "real"], "seq7", 0)
            with self.assertRaises(FileNotFoundError):
                tsne.check_files(root, selected)
            panel = ("hypclip", "seq7", 0)
            artifact = tsne.panel_file(root, panel, "points.npz")
            metadata = tsne.identity(panel, digest, None)
            labels = list(map(tsne.generator, selected))
            features = np.random.default_rng(42).normal(size=(len(selected), 70))
            points, fit = tsne.fit_tsne(features)
            self.assertEqual(fit["pca_dimensions"], 35)
            tsne.save_arrays(artifact, {**metadata, "tsne": fit}, image_ids=selected,
                             labels=labels, coordinates=points)
            actual, actual_labels, _ = tsne.load_arrays(artifact, "coordinates", metadata, selected)
            np.testing.assert_array_equal(actual, points)
            self.assertEqual(actual_labels.tolist(), labels)
            with self.assertRaisesRegex(ValueError, "Test rows differ"):
                tsne.load_arrays(artifact, "coordinates", metadata, selected[::-1])
            with self.assertRaises(FileExistsError):
                tsne.save_arrays(artifact, metadata, coordinates=np.zeros_like(points))
            manifest.write_text(json.dumps({"test": [*rows, rows[0]]}))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                tsne.read_test_rows(manifest)

    def test_final_partial_batch_is_kept(self):
        data = torch.arange(15).reshape(5, 3)
        features, labels = tsne.collect_batches(
            data, lambda batch: (batch, batch[:, 0].tolist()), torch.device("cpu"), 2, 0)
        np.testing.assert_array_equal(features, data.numpy())
        self.assertEqual(labels, [0, 3, 6, 9, 12])


if __name__ == "__main__":
    unittest.main()
