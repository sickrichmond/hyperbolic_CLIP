import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from comparison.dataset.ImageAttributionDataset.dataset import build_label_map
from comparison.training.eval_aug_splits import (
    EXPECTED_NUM_CLASSES,
    TARGET_CLASSES,
    balanced_subset_indices,
    checkpoint_output_classes,
    count_candidate_files,
    count_valid_files,
    evaluate_protocols,
    exact_shard_indices,
    image_level_batch_field,
    probabilities_from_output,
    workers_for_rank,
)


class AugmentedEvaluationTests(unittest.TestCase):
    def test_target_indices_are_preserved_in_canonical_22_class_map(self):
        labels = build_label_map({"dalle3"})
        self.assertEqual(len(labels), EXPECTED_NUM_CLASSES)
        self.assertEqual(
            {name: labels[name] for name in TARGET_CLASSES},
            {"FLUX": 2, "SD3": 8, "SD3_5": 9, "SDXL": 10},
        )

    def test_patch_probabilities_are_spatially_averaged(self):
        logits = torch.tensor(
            [[[[4.0, 0.0]], [[0.0, 4.0]], [[-2.0, -2.0]]]], dtype=torch.float32
        )
        actual = probabilities_from_output("patch", {"logits": logits})
        expected = torch.softmax(logits, dim=1).mean(dim=(2, 3))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(dim=1), torch.ones(1))

    def test_non_patch_output_must_be_image_level(self):
        with self.assertRaisesRegex(ValueError, "must return"):
            probabilities_from_output("resnet50", {"logits": torch.zeros(2, 3, 1)})

    def test_repmix_singleton_mixup_axis_is_removed(self):
        logits = torch.randn(3, 1, EXPECTED_NUM_CLASSES)
        actual = probabilities_from_output("repmix", {"logits": logits})
        self.assertEqual(tuple(actual.shape), (3, EXPECTED_NUM_CLASSES))
        torch.testing.assert_close(actual, torch.softmax(logits[:, 0, :], dim=1))

    def test_repmix_singleton_label_axis_is_removed(self):
        labels = torch.tensor([[2], [8], [9]])
        actual = image_level_batch_field(labels, 3, "label")
        torch.testing.assert_close(actual, torch.tensor([2, 8, 9]))
        with self.assertRaisesRegex(ValueError, "image-level shape"):
            image_level_batch_field(torch.zeros(3, 2), 3, "label")

    def test_checkpoint_head_width_is_validated_by_model(self):
        state = {"fc.weight": torch.zeros(EXPECTED_NUM_CLASSES, 2048)}
        self.assertEqual(
            checkpoint_output_classes("resnet50", state), EXPECTED_NUM_CLASSES
        )
        with self.assertRaisesRegex(KeyError, "output head"):
            checkpoint_output_classes("resnet50", {})

    def test_full_and_restricted_protocols_keep_off_target_predictions(self):
        names = ["other", "FLUX", "SD3", "SD3_5", "SDXL"]
        labels = np.array([1, 2, 3, 4])
        semantics = np.array([0, 0, 1, 1])
        probabilities = np.array(
            [
                [0.60, 0.35, 0.02, 0.02, 0.01],
                [0.01, 0.05, 0.90, 0.02, 0.02],
                [0.01, 0.05, 0.02, 0.90, 0.02],
                [0.01, 0.05, 0.02, 0.02, 0.90],
            ]
        )
        result = evaluate_protocols(
            probabilities, labels, semantics, names, semantic_names={0: "a", 1: "b"}
        )
        self.assertAlmostEqual(result["full_22_way"]["accuracy"], 0.75)
        self.assertAlmostEqual(result["full_22_way"]["off_target_prediction_rate"], 0.25)
        self.assertEqual(result["full_22_way"]["off_target_predictions"], {"other": 1})
        self.assertAlmostEqual(result["restricted_4_way"]["accuracy"], 1.0)
        self.assertEqual(result["full_22_way"]["confusion_matrix"]["labels"], names)
        self.assertEqual(
            result["restricted_4_way"]["confusion_matrix"]["labels"], names[1:]
        )

    def test_confusion_matrices_keep_fixed_22_and_4_class_shapes(self):
        label_map = build_label_map({"dalle3"})
        names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
        target_indices = [label_map[name] for name in TARGET_CLASSES]
        probabilities = np.full((4, EXPECTED_NUM_CLASSES), 1e-6)
        for row, target_index in enumerate(target_indices):
            probabilities[row, target_index] = 1.0
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        result = evaluate_protocols(
            probabilities,
            np.asarray(target_indices),
            np.arange(4),
            names,
        )
        self.assertEqual(
            np.asarray(result["full_22_way"]["confusion_matrix"]["matrix"]).shape,
            (22, 22),
        )
        self.assertEqual(
            np.asarray(result["restricted_4_way"]["confusion_matrix"]["matrix"]).shape,
            (4, 4),
        )
        self.assertIsNone(result["full_22_way"]["per_class"]["4o"]["auc"])

    def test_balanced_subset_round_robins_classes(self):
        labels = [2, 2, 2, 8, 8, 8, 9, 9, 9, 10, 10, 10]
        selected = balanced_subset_indices(labels, 8)
        self.assertIsNotNone(selected)
        self.assertEqual(
            [labels[index] for index in selected], [2, 8, 9, 10, 2, 8, 9, 10]
        )
        self.assertIsNone(balanced_subset_indices(labels, None))
        with self.assertRaisesRegex(ValueError, "positive"):
            balanced_subset_indices(labels, 0)

    def test_distributed_shards_are_exact_without_padding(self):
        total = 78_125
        shards = [exact_shard_indices(total, rank, 4) for rank in range(4)]
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(len(flattened), total)
        self.assertEqual(len(set(flattened)), total)
        self.assertEqual(set(flattened), set(range(total)))
        self.assertEqual([len(shard) for shard in shards], [19532, 19531, 19531, 19531])
        with self.assertRaisesRegex(ValueError, "between 1 and 4"):
            exact_shard_indices(total, 0, 5)

    def test_loader_workers_are_a_job_wide_budget(self):
        self.assertEqual([workers_for_rank(8, rank, 4) for rank in range(4)], [2, 2, 2, 2])
        self.assertEqual([workers_for_rank(2, rank, 4) for rank in range(4)], [1, 1, 0, 0])

    def test_inventory_accepts_images_without_training_filename_filters(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "FLUX" / "AnimalFace" / "cat"
            cell.mkdir(parents=True)
            (cell / "AnimalFace_cat_p1_i0.png").touch()
            (cell / ".AnimalFace_cat_p1_i0.png.partial").touch()
            (cell / "AnimalFace_cat_p1000_i0.png").touch()
            (cell / "AnimalFace_cat_p1_i2.png").touch()
            (cell / "AnimalFace_cat_p2_i0.jpg").touch()
            (cell / "notes.txt").touch()
            self.assertEqual(count_valid_files(root, {"cat": "AnimalFace/cat"}), 4)
            self.assertEqual(count_candidate_files(root, {"cat": "AnimalFace/cat"}), 6)

            # The inventory is recursive and independent from the semantic map,
            # so an image in an unexpected path cannot be silently skipped.
            unexpected = root / "FLUX" / "unexpected" / "extra.png"
            unexpected.parent.mkdir()
            unexpected.touch()
            self.assertEqual(count_valid_files(root, {"cat": "AnimalFace/cat"}), 5)
            self.assertEqual(count_candidate_files(root, {"cat": "AnimalFace/cat"}), 7)


class SlurmAugmentedEvaluationTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    slurm_dir = root / "SLURM_aug_test"

    def test_array_indices_map_to_four_distinct_expected_datasets(self):
        datasets = (self.slurm_dir / "datasets.txt").read_text().splitlines()
        self.assertEqual(
            datasets,
            [
                "iab_recap_dataset_v2",
                "iab_recap_cartoon_v2",
                "iab_recap_clipart_v2",
                "iab_recap_photorealistic_v2",
            ],
        )
        self.assertEqual(len(datasets), len(set(datasets)))

    def test_eight_launchers_each_define_four_tasks(self):
        launchers = sorted(self.slurm_dir.glob("eval_*.sbatch"))
        self.assertEqual(len(launchers), 8)
        for launcher in launchers:
            contents = launcher.read_text()
            self.assertIn("#SBATCH --array=0-3", contents)
            self.assertIn("#SBATCH --nodes=1", contents)
            self.assertIn("#SBATCH --gpus-per-node=4", contents)
            self.assertIn("#SBATCH --mem=32G", contents)

        submit_script = (self.slurm_dir / "submit_all.sh").read_text()
        for launcher in launchers:
            self.assertEqual(submit_script.count(launcher.name), 1)
        self.assertEqual(len(launchers) * 4, 32)

    def test_manifest_pins_all_canonical_runs(self):
        rows = {}
        for line in (self.slurm_dir / "manifest.tsv").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            model, config, checkpoint, batch_size, workers = line.split("\t")
            rows[model] = {
                "config": config,
                "checkpoint": checkpoint,
                "batch_size": int(batch_size),
                "workers": int(workers),
            }
        self.assertEqual(
            set(rows),
            {"resnet50", "dct", "hifi_net", "defl", "dna", "repmix", "patch", "ucf"},
        )
        for model in ("resnet50", "dct", "hifi_net"):
            self.assertIn("_2026-07-19-18-53-50/ckpt_best.pth", rows[model]["checkpoint"])
        self.assertIn("_2026-07-21-01-11-04/ckpt_best.pth", rows["defl"]["checkpoint"])
        self.assertIn("_2026-07-21-07-31-06/ckpt_best.pth", rows["dna"]["checkpoint"])
        for model in ("repmix", "patch", "ucf"):
            self.assertIn("_2026-07-21-01-11-42/ckpt_best.pth", rows[model]["checkpoint"])
        self.assertLess(rows["defl"]["workers"], rows["resnet50"]["workers"])


if __name__ == "__main__":
    unittest.main()
