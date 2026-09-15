"""CPU integration checks with synthetic features: python -m tests.test_attribution_training."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

import train_attribution as trainer
from geometry.lorentz import exp_map0
from models.attribution_clip import AttributionCLIP
from training.attribution_args import parse_args, validate_args


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        ids = torch.arange(len(texts)).unsqueeze(1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TinyDataset(Dataset):
    calls = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)
        self.samples = [(str(i), c, "COCO")
                        for c in kwargs["generators"] for i in range(4)]
        self.classes = kwargs["generators"]
        self.train_augment = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        _, name, _ = self.samples[i]
        label = self.classes.index(name)
        item = {"pixel_values": torch.eye(4)[label] + 0.01 * (i % 4),
                "generator": name}
        return item


class TinyModel(nn.Module):
    def __init__(self, curv=1.0, image_radius=0.0, **kwargs):
        super().__init__()
        self.curv, self.image_radius = curv, image_radius
        self.clip = nn.Linear(4, 4)
        self.clip.base_model = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(projection_dim=4)))
        self.projection = nn.Sequential(nn.Linear(4, 4))

    def _clip_image(self, pixel):
        return self.clip(pixel)

    def encode_image(self, pixel):
        tangent = F.normalize(self.projection(self._clip_image(pixel)), dim=-1)
        tangent = tangent * (self.image_radius or 3.0)
        return exp_map0(tangent, curv=self.curv), tangent

    def encode_text(self, ids, mask):
        tangent = self.projection(torch.eye(4)[ids[:, 0]]) * 0.3
        return exp_map0(tangent, curv=self.curv), tangent

    def forward(self, pixel):
        return self.encode_image(pixel)[0]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_summary(self):
        pass


def run_tiny_training(module, directory, options, manifest=False):
    """Exercise real optimization, validation, saving and final embedding collection."""
    directory = Path(directory)
    (directory / "split.json").write_text(json.dumps(
        {"train": ["train-row"], "val": ["val-row"]}))
    args = parse_args([
        "--dataset_path", "images", "--captions_dir", "captions",
        "--generators", "real", "FLUX", "SDXL", "--hyperbolic_dim", "4",
        "--num_epochs", "2", "--batch_size", "6", "--num_workers", "0",
        "--output", str(directory / "model.pt"),
        "--diag_plot_dir", str(directory), "--log_every", "1",
        *options,
        *(["--split_manifest", str(directory / "split.json")] if manifest else []),
    ])
    frames = []

    def snapshot(images, labels, anchors, names, path, **kwargs):
        frames.append((Path(path).name, np.array(images), list(labels)))
        return (None, None)

    plotter = SimpleNamespace(plot_epoch_snapshot=snapshot, _load_horopca=lambda: None)
    TinyDataset.calls = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(module, "AttributionCLIP", TinyModel))
        stack.enter_context(patch.object(module, "IABCLIPDataset", TinyDataset))
        stack.enter_context(patch.object(module, "parse_args", return_value=args))
        stack.enter_context(patch("transformers.CLIPTokenizer.from_pretrained",
                                  return_value=TinyTokenizer()))
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        stack.enter_context(patch("torch.cuda.device_count", return_value=0))
        stack.enter_context(patch.dict("sys.modules", {
            "training.poincare": plotter, "tests.visualize_horopca": plotter}))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        module.main()
    checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
    return checkpoint, frames, list(TinyDataset.calls)


class AttributionTrainingTests(unittest.TestCase):
    def test_checkpoint_loss_compatibility(self):
        # Reject incompatible scoring rules before downloading model weights.
        for loss in ("axis", "unknown"):
            with self.subTest(loss=loss), patch.object(
                    AttributionCLIP, "__init__", return_value=None) as construct:
                with self.assertRaisesRegex(ValueError, "Unsupported checkpoint loss"):
                    AttributionCLIP.from_checkpoint({"loss": loss})
                construct.assert_not_called()
        for metadata in ({}, {"loss": "cone"}):
            with self.subTest(metadata=metadata), patch.object(
                    AttributionCLIP, "__init__", return_value=None) as construct:
                AttributionCLIP.from_checkpoint({"clip_name": "test", **metadata})
                construct.assert_called_once()

    def test_cli_help_and_constraints(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
            parse_args(["--help"])
        self.assertEqual(cm.exception.code, 0)
        base = ["--dataset_path", "x", "--captions_dir", "y"]
        args = parse_args(base + ["--no_captions"])
        validate_args(args)
        for flag in ("--lambda_cap_in_class", "--lambda_img_in_cap",
                     "--lambda_axis", "--lambda_hinge", "--pos_mode",
                     "--lambda_family", "--hierarchy", "--theta_max",
                     "--loss", "--lambda_ce", "--ce_tau_init", "--lambda_sep",
                     "--lambda_cover", "--lambda_center", "--lambda_aperture",
                     "--inside_margin", "--separation_margin", "--nu",
                     "--psi_range", "--fixed_psi", "--calibrate_psi"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(base + [flag, "1"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(base + ["--anchor_init", "simplex"])
        args.fixed_image_radius, args.init_depth = 4, 3
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_epoch_pca_uses_all_fit_points(self):
        from training.poincare import run_pca_2d

        fit = np.array([[10, 0, 0], [-10, 0, 0], [0, 1, 0], [0, -1, 0],
                        [0, 0, 100]], dtype=np.float32)
        projected = run_pca_2d(fit, np.eye(3, dtype=np.float32))
        lengths = np.linalg.norm(projected, axis=1)
        self.assertGreater(lengths[0], 0.99)
        self.assertLess(lengths[1], 0.01)
        self.assertGreater(lengths[2], 0.99)

    def test_training_modes_and_dataset_paths(self):
        cases = [
            (["--anchor_init", "random", "--anchors_only",
              "--anchor_norm_range", "1", "2", "--fixed_image_radius", "4"], False),
            (["--anchor_init", "random", "--freeze_anchors"], False),
            (["--anchor_init", "text", "--lambda_norm", "0.5",
              "--target_norm", "4", "--norm_mode", "bilateral"], True),
            (["--anchor_init", "text_free", "--neg_samples", "1"], False),
            (["--anchor_init", "image_centroid", "--anchor_norm_range", "1", "2"], False),
            (["--anchor_init", "random", "--require_caption"], True),
        ]
        for options, manifest in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                checkpoint, frames, calls = run_tiny_training(
                    trainer, directory, options, manifest=manifest)
                self.assertEqual(checkpoint["class_names"], ["real", "FLUX", "SDXL"])
                self.assertGreaterEqual(checkpoint["val_balanced"], 0)
                self.assertEqual(checkpoint["hierarchy"], "none")
                self.assertEqual(checkpoint["family_names"], [])
                self.assertIsNone(checkpoint["family_of"])
                self.assertEqual(checkpoint["loss"], "cone")
                for key in ("anchor_sin_psi", "fixed_psi", "lambda_ce", "ce_tau",
                            "lambda_cover", "aperture_calibration"):
                    self.assertNotIn(key, checkpoint)
                self.assertEqual([f[0] for f in frames],
                                 ["epoch_01.png", "epoch_02.png"])
                for _, images, labels in frames:
                    self.assertEqual(len(images), 12)
                    self.assertEqual(set(labels), {0, 1, 2})
                    self.assertTrue(np.isfinite(images).all())
                self.assertEqual(calls[0]["require_caption"], "--require_caption" in options)
                self.assertEqual(calls[0]["seed"], calls[1]["seed"])
                self.assertEqual(calls[0]["split"], "all" if manifest else "train")
                self.assertEqual(calls[1]["split"], "all" if manifest else "val")
                if manifest:
                    self.assertEqual(calls[0]["include_paths"], {"train-row"})
                    self.assertEqual(calls[1]["include_paths"], {"val-row"})
                rows = (Path(directory) / "stats.csv").read_text().splitlines()
                self.assertEqual(len(rows), 5)
                self.assertTrue(rows[0].startswith("step,epoch,lr,loss,"))


if __name__ == "__main__":
    unittest.main()
