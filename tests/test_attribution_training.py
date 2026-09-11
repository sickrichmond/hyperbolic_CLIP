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
from losses.axis_cone_loss import regular_simplex
from training.attribution_args import parse_args, validate_args


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        ids = torch.arange(len(texts)).unsqueeze(1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TinyDataset(Dataset):
    captions = False
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
        if self.captions:
            item.update(input_ids=torch.tensor([label]),
                        attention_mask=torch.tensor([1]))
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

    def forward(self, pixel, ids=None, mask=None):
        image = self.encode_image(pixel)[0]
        return image if ids is None else (image, self.encode_text(ids, mask)[0])

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_summary(self):
        pass


def run_tiny_training(module, directory, options, captions=False, manifest=False):
    """Exercise real optimization, validation, saving and final embedding collection."""
    directory = Path(directory)
    (directory / "tree.json").write_text(json.dumps(
        {"real": "real", "FLUX": "diffusion", "SDXL": "diffusion"}))
    (directory / "split.json").write_text(json.dumps(
        {"train": ["train-row"], "val": ["val-row"]}))
    args = parse_args([
        "--dataset_path", "images", "--captions_dir", "captions",
        "--generators", "real", "FLUX", "SDXL", "--hyperbolic_dim", "4",
        "--num_epochs", "2", "--batch_size", "6", "--num_workers", "0",
        "--output", str(directory / "model.pt"),
        "--diag_plot_dir", str(directory), "--snapshot_every", "1",
        "--plot_all_train", "--log_every", "1",
        "--hierarchy_json", str(directory / "tree.json"),
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
        stack.enter_context(patch.object(TinyDataset, "captions", captions))
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
    def test_feasible_aperture_calibration(self):
        axes = regular_simplex(3, 4)

        def aligned_images(model, pixel):
            tangent = 3 * axes[pixel.argmax(1)] + 0 * model.projection[0].weight.sum()
            return exp_map0(tangent, curv=model.curv), tangent

        options = ["--loss", "axis", "--anchor_init", "simplex", "--freeze_anchors",
                   "--fixed_psi", "45", "--lambda_aperture", "0",
                   "--lambda_cover", "5", "--inside_margin", "2",
                   "--separation_margin", "2", "--calibrate_psi"]
        with tempfile.TemporaryDirectory() as directory, patch.object(
                TinyModel, "encode_image", aligned_images):
            checkpoint, _, _ = run_tiny_training(trainer, directory, options)
        self.assertTrue(checkpoint["aperture_calibration"]["applied"])
        self.assertEqual(checkpoint["val_balanced_calibrated"], 1.0)
        self.assertTrue(all(x == 1.0 for x in
                            checkpoint["aperture_calibration"]["train_coverage"]))

    def test_cli_help_and_constraints(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
            parse_args(["--help"])
        self.assertEqual(cm.exception.code, 0)
        args = parse_args(["--dataset_path", "x", "--captions_dir", "y",
                           "--no_captions", "--lambda_cap_in_class", "1"])
        validate_args(args)
        self.assertEqual(args.lambda_cap_in_class, 0)
        args.fixed_image_radius, args.init_depth = 4, 3
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_training_modes_and_dataset_paths(self):
        cases = [
            (["--loss", "axis", "--anchor_init", "simplex", "--freeze_anchors",
              "--fixed_psi", "45", "--lambda_aperture", "0", "--lambda_cover", "5",
              "--lambda_ce", "0.5", "--calibrate_psi"], False, True),
            (["--loss", "axis", "--anchor_init", "random", "--anchors_only"],
             False, False),
            (["--anchor_init", "text", "--lambda_cap_in_class", "0.5",
              "--lambda_img_in_cap", "0.5", "--hierarchy", "emergent",
              "--lambda_family", "0.2", "--lambda_sep", "0.3"], True, True),
            (["--anchor_init", "text_free", "--lambda_ce", "0.5"], False, False),
            (["--anchor_init", "image_centroid", "--anchor_norm_range", "1", "2"],
             False, False),
        ]
        for options, captions, manifest in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                checkpoint, frames, calls = run_tiny_training(
                    trainer, directory, options, captions, manifest)
                self.assertEqual(checkpoint["class_names"], ["real", "FLUX", "SDXL"])
                self.assertGreaterEqual(checkpoint["val_balanced"], 0)
                self.assertEqual(frames[-1][0], "train_all_final.png")
                self.assertEqual(len(frames[-1][1]), 12)
                self.assertEqual(set(frames[-1][2]), {0, 1, 2})
                self.assertTrue(np.isfinite(frames[-1][1]).all())
                self.assertEqual(calls[0]["require_caption"], captions)
                self.assertEqual(calls[0]["seed"], calls[1]["seed"])
                self.assertEqual(calls[0]["split"], "all" if manifest else "train")
                self.assertEqual(calls[1]["split"], "all" if manifest else "val")
                if manifest:
                    self.assertEqual(calls[0]["include_paths"], {"train-row"})
                    self.assertEqual(calls[1]["include_paths"], {"val-row"})
                if "--calibrate_psi" in options:
                    self.assertIn("aperture_calibration", checkpoint)
                rows = (Path(directory) / "stats.csv").read_text().splitlines()
                self.assertEqual(len(rows), 5)
                self.assertTrue(rows[0].startswith("step,epoch,lr,loss,"))


if __name__ == "__main__":
    unittest.main()
