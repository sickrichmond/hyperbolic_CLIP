"""Saved-Arrow integration: python -m tests.test_iabench_training (no downloads)."""
from collections import Counter
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
import torch

from comparison.training import test_hypclip as evaluator
from comparison.training.metrics.base_metrics_class import calculate_metrics_for_test
from data.iabench_dataset import IABenchDataset, manifest_digest
from geometry.lorentz import exp_map0, oxy_angle
import train_attribution as trainer
import train_iabench
from tests.test_attribution_training import TinyModel, TinyTokenizer
from training.anchors import build_anchors


def save_arrow(root, counts=None, raw_shards=False):
    counts = counts or {"SDXL": 31, "Real": 11, "Flux.1": 21}
    rows = {"image": [], "label": [], "file_name": []}
    for j, (name, count) in enumerate(counts.items()):
        color = tuple(255 if channel == j % 3 else 0 for channel in range(3))
        for i in range(count):
            rows["image"].append(PILImage.new("RGB", (8, 8), color))
            rows["label"].append(name)
            rows["file_name"].append(f"{name}/{i}.png")
    data = Dataset.from_dict(rows, features=Features(
        {"image": Image(), "label": Value("string"), "file_name": Value("string")}))
    data_dir = Path(root) / "data"
    data.save_to_disk(str(data_dir), num_shards=3 if raw_shards else 1)
    if raw_shards:
        (data_dir / "state.json").unlink()
        (data_dir / "dataset_info.json").unlink()


class PixelProcessor:
    def __call__(self, images, **kwargs):
        r, g, b = images.getpixel((0, 0))
        return {"pixel_values": torch.tensor([[r / 255, g / 255, b / 255, 1.]])}


class IABenchIntegrationTests(unittest.TestCase):
    def test_prepare_only_without_images_or_clip(self):
        with tempfile.TemporaryDirectory() as root:
            save_arrow(root, raw_shards=True)
            path = Path(root) / "prepared.json"
            args = train_iabench.parse_args([
                "--dataset_path", root, "--split_manifest", str(path), "--prepare_only"])
            with patch.object(train_iabench, "parse_args", return_value=args), \
                    patch("data.iabench_dataset.CLIPImageProcessor.from_pretrained",
                          side_effect=AssertionError("processor loaded")), \
                    patch.object(Image, "decode_example", side_effect=AssertionError("image decoded")), \
                    patch.object(train_iabench, "run_training") as train, \
                    contextlib.redirect_stdout(io.StringIO()):
                train_iabench.main()
                original = path.read_bytes()
                train_iabench.main()
                self.assertEqual(path.read_bytes(), original)
                train.assert_not_called()
                manifest = json.loads(original)
                self.assertEqual([len(manifest[s]) for s in ("train", "val", "test")], [48, 6, 9])
                dataset = IABenchDataset(root, processor_name=None)
                self.assertEqual(dataset.make_split_manifest(), manifest)
                with self.assertRaisesRegex(RuntimeError, "Metadata-only"):
                    dataset[0]

    def test_split_manifest_and_views(self):
        with tempfile.TemporaryDirectory() as root, patch(
                "data.iabench_dataset.CLIPImageProcessor.from_pretrained",
                return_value=PixelProcessor()):
            save_arrow(root)
            dataset = IABenchDataset(root)
            # Splits use only cached metadata; row access would decode an image.
            with patch.object(Dataset, "__getitem__", side_effect=AssertionError("image read")):
                manifest = dataset.make_split_manifest()
                self.assertEqual(manifest, dataset.make_split_manifest())
            self.assertEqual(manifest["class_names"], ["Real", "Flux.1", "SDXL"])
            groups = [set(manifest[s]) for s in ("train", "val", "test")]
            self.assertEqual(set.union(*groups), set(range(len(dataset))))
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            self.assertEqual([len(g) for g in groups], [48, 6, 9])
            for split, expected in (("train", [8, 16, 24]), ("val", [1, 2, 3]),
                                    ("test", [2, 3, 4])):
                view = dataset.split_view(manifest, split)
                self.assertIs(view.data, dataset.data)
                self.assertIs(view.processor, dataset.processor)
                counts = Counter(g for _, g, _ in view.samples)
                self.assertEqual([counts[n] for n in manifest["class_names"]], expected)
            self.assertNotEqual(manifest["train"], dataset.make_split_manifest(seed=7)["train"])
            subset = dataset.make_split_manifest(generators=["SDXL", "Real"], max_per_class=10)
            self.assertEqual(subset["class_names"], ["Real", "SDXL"])
            self.assertEqual([len(subset[s]) for s in ("train", "val", "test")], [16, 2, 2])
            filtered = IABenchDataset(root, generators=["SDXL"], max_per_class=10)
            self.assertEqual(filtered.make_split_manifest()["class_names"], ["SDXL"])
            damaged = json.loads(json.dumps(manifest))
            damaged["test"][0] = damaged["train"][0]
            with self.assertRaisesRegex(ValueError, "does not match"):
                dataset.split_view(damaged, "test")
            with tempfile.TemporaryDirectory() as reordered:
                dataset.data.select(list(reversed(range(len(dataset))))).save_to_disk(
                    str(Path(reordered) / "data"))
                with self.assertRaisesRegex(ValueError, "does not match"):
                    IABenchDataset(reordered).split_view(manifest, "test")
            with self.assertRaisesRegex(ValueError, "too small"):
                dataset.make_split_manifest(max_per_class=9)
            with self.assertRaisesRegex(ValueError, "fractions"):
                dataset.make_split_manifest(val_frac=float("nan"))
            dataset.class_names.append("real")
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                dataset.make_split_manifest()

    def test_manifest_reuse_cli_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as root, patch(
                "data.iabench_dataset.CLIPImageProcessor.from_pretrained",
                return_value=PixelProcessor()):
            save_arrow(root)
            args = train_iabench.parse_args(["--dataset_path", root,
                                            "--output", str(Path(root) / "model.pt")])
            self.assertIsNone(args.generators)
            self.assertEqual((args.val_frac, args.test_frac, args.seed), (0.1, 0.1, 42))
            views, metadata = train_iabench.prepare_datasets(args)
            again, second = train_iabench.prepare_datasets(args)
            self.assertEqual(metadata, second)
            self.assertEqual(views[0].indices, again[0].indices)
            args.seed = 7
            with self.assertRaisesRegex(ValueError, "split settings"):
                train_iabench.prepare_datasets(args)
            for flag in ("--captions_dir", "--semantics", "--require_caption", "--split_scheme"):
                with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        train_iabench.parse_args(["--dataset_path", root, flag, "x"])
            train, val = views
            train.train_augment = True
            with patch.dict("data.iabench_dataset.AUG_POLICIES", {"corruption": lambda image: image}), \
                    patch("data.iabench_dataset.apply_degradation", side_effect=lambda image, _: image) as degrade:
                train[0]
                val[0]
                self.assertFalse(val.train_augment)
                train.degraded = 1
                with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                    train[0]
                val.degraded = 1
                before = list(val.indices)
                val[0]
                degrade.assert_called_once()
                self.assertEqual(val.indices, before)
            read = train.data.__getitem__
            row_index = val.indices[0]
            attempts = []

            def flaky(index):
                attempts.append(index)
                if len(attempts) == 1:
                    raise OSError("transient Arrow I/O")
                return read(index)

            with patch.object(Dataset, "__getitem__", side_effect=flaky), \
                    patch("data.image_io.time.sleep"):
                val[0]
            self.assertEqual(attempts, [row_index, row_index])

    def test_training_and_test_evaluation_all_anchor_modes(self):
        for mode in ("text", "text_free", "random", "image_centroid"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                save_arrow(root, raw_shards=mode == "random")
                output = Path(root) / "model.pt"
                args = train_iabench.parse_args([
                    "--dataset_path", root, "--output", str(output), "--anchor_init", mode,
                    "--hyperbolic_dim", "4", "--batch_size", "12", "--num_workers", "0",
                    "--num_epochs", "2", "--diag_plot_dir", root, "--log_every", "1",
                    *(["--plot_all_train", "--profile_steps", "3"] if mode == "random" else []),
                    *(["--profile_steps", "100"] if mode == "text" else [])])
                validation = trainer.run_validation
                frames = []

                def snapshot(images, labels, anchors, names, path, **kwargs):
                    frames.append((Path(path).name, len(images)))

                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch("data.iabench_dataset.CLIPImageProcessor.from_pretrained",
                                              return_value=PixelProcessor()))
                    stack.enter_context(patch.object(train_iabench, "parse_args", return_value=args))
                    stack.enter_context(patch.object(trainer, "AttributionCLIP", TinyModel))
                    stack.enter_context(patch("transformers.CLIPTokenizer.from_pretrained",
                                              return_value=TinyTokenizer()))
                    stack.enter_context(patch("torch.cuda.is_available", return_value=False))
                    stack.enter_context(patch("torch.cuda.device_count", return_value=0))
                    if mode == "random":
                        stack.enter_context(patch.dict("sys.modules", {
                            "training.poincare": SimpleNamespace(plot_epoch_snapshot=snapshot)}))
                        stack.enter_context(patch.object(
                            trainer, "perf_counter", side_effect=[0, 5, 7, 8, 11, 12, 13, 19, 20]))
                    val_spy = stack.enter_context(patch.object(trainer, "run_validation", wraps=validation))
                    epoch_collect = stack.enter_context(patch.object(
                        trainer, "collect_plot_embeddings",
                        side_effect=AssertionError("IABench must skip the per-epoch data pass")))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    train_iabench.main()
                    ckpt = torch.load(output, weights_only=False, map_location="cpu")
                    manifest = json.loads(Path(ckpt["split_manifest"]).read_text())
                    self.assertEqual(ckpt["dataset"], "iabench")
                    self.assertEqual(ckpt["class_names"], manifest["class_names"])
                    self.assertEqual(ckpt["anchor_texts"][0], "A real image")
                    self.assertEqual(ckpt["split_manifest_digest"], manifest_digest(manifest))
                    self.assertEqual(ckpt["loss"], "cone")
                    epoch_collect.assert_not_called()
                    self.assertEqual(val_spy.call_count, 2)
                    rows = (Path(root) / "stats.csv").read_text().splitlines()
                    self.assertEqual(len(rows), 9)  # Header + four steps in each epoch.
                    self.assertTrue(rows[0].startswith("step,epoch,lr,loss,"))
                    for call in val_spy.call_args_list:
                        self.assertEqual(call.args[1].dataset.split, "val")
                        self.assertEqual(call.args[1].dataset.indices, manifest["val"])
                    if mode == "random":
                        self.assertEqual(frames, [("train_all_final.png", 48)])
                        with (Path(root) / "timing.csv").open() as stream:
                            timing = list(csv.DictReader(stream))
                        self.assertEqual([int(r["step"]) for r in timing], [1, 2, 3])
                        self.assertEqual([float(r["data_wait_s"]) for r in timing], [5., 3., 6.])
                        self.assertEqual([float(r["train_step_s"]) for r in timing], [2., 1., 1.])
                    elif mode == "text":
                        with (Path(root) / "timing.csv").open() as stream:
                            timing = list(csv.DictReader(stream))
                        self.assertEqual([int(r["step"]) for r in timing], list(range(1, 9)))
                        self.assertEqual([int(r["epoch"]) for r in timing], [1]*4 + [2]*4)
                        self.assertTrue(all(float(r["data_wait_s"]) >= 0 for r in timing))
                    else:
                        self.assertFalse((Path(root) / "timing.csv").exists())
                    eval_args = evaluator.parse_args([
                        "--checkpoint", str(output), "--root_dir", root, "--dataset", "iabench",
                        "--num_workers", "0", "--batch_size", "12", "--level_end", "2",
                        "--log_dir", str(Path(root) / "metrics")])
                    score = evaluator.evaluate_loader
                    with patch.object(evaluator, "parse_args", return_value=eval_args), \
                            patch.object(evaluator.AttributionCLIP, "from_checkpoint",
                                         return_value=TinyModel()), \
                            patch.object(evaluator, "evaluate_loader", wraps=score) as test_spy:
                        evaluator.main()
                    for call in test_spy.call_args_list:
                        test = call.args[1].dataset
                        self.assertEqual(test.split, "test")
                        self.assertEqual(test.indices, manifest["test"])
                        self.assertFalse(test.train_augment)
                    self.assertEqual(test_spy.call_count, 2)
                    text = (Path(root) / "metrics/test_results_degraded_0.txt").read_text()
                    for key in ("balanced_accuracy", "precision_per_class", "support_per_class", "conf_matrix"):
                        self.assertIn(key, text)
                    self.assertIn(json.dumps(ckpt["class_names"]), text)
                    self.assertNotIn("semantic_acc:", text)
                    bad = dict(ckpt, split_manifest_digest="wrong")
                    with self.assertRaisesRegex(ValueError, "manifest digest"):
                        evaluator.load_iabench_test_dataset(eval_args, bad)

    def test_scoring_and_anchor_class_alignment(self):
        names, texts = build_anchors(["SDXL", "Real", "Flux.1"])
        self.assertEqual(names, ["Real", "SDXL", "Flux.1"])
        self.assertEqual(texts[0], "A real image")
        tangent = torch.eye(3) * 0.3
        checkpoint = {"class_names": names, "anchor_tangent": tangent}
        reverse = list(reversed(names))
        anchors = evaluator.load_anchors(checkpoint, None, 1., "cpu", class_names=reverse)
        self.assertTrue(torch.equal(anchors, exp_map0(tangent.flip(0))))
        with self.assertRaisesRegex(ValueError, "no anchor"):
            evaluator.load_anchors(checkpoint, None, 1., "cpu", class_names=["missing"])
        model = TinyModel()
        rows = [{"pixel_values": torch.eye(4)[i], "generator": names[i]} for i in range(3)]
        loader = torch.utils.data.DataLoader(rows, batch_size=3)
        x_anc = exp_map0(torch.eye(4)[:3] * 0.3)
        logits, labels, sem = evaluator.evaluate_loader(model, loader, x_anc, names, "cpu", 1.)
        x_img = model.encode_image(torch.eye(4)[:3])[0]
        expected = torch.stack([oxy_angle(a.expand_as(x_img), x_img) for a in x_anc], dim=1)
        self.assertTrue(torch.allclose(logits, -expected))
        self.assertEqual(labels.tolist(), [0, 1, 2])
        self.assertIsNone(sem)
        metrics = calculate_metrics_for_test(labels, logits)
        self.assertEqual(metrics[3].shape, (3, 3))
        self.assertEqual(metrics[4], {})
        self.assertEqual(metrics[5]["support_per_class"], [1, 1, 1])
        # Existing harness batches use image/label/semantic_label instead.
        old_rows = [{"image": row["pixel_values"], "label": i, "semantic_label": 0}
                    for i, row in enumerate(rows)]
        old_logits, old_labels, semantics = evaluator.evaluate_loader(
            model, torch.utils.data.DataLoader(old_rows, batch_size=3), x_anc, names, "cpu", 1.)
        self.assertTrue(torch.equal(old_logits, logits))
        self.assertTrue(torch.equal(old_labels, labels))
        self.assertEqual(semantics.tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
