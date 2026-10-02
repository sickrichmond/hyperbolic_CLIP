"""Small saved-Arrow check for data.iabench_dataset."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

from data.iabench_dataset import IABenchDataset


class IABenchDatasetTests(unittest.TestCase):
    def test_saved_arrow_loading_and_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = Dataset.from_dict(
                {
                    "image": [PILImage.new("RGB", (4, 4), color) for color in
                              ("red", "green", "blue")],
                    "label": ["Real", "SDXL", "SDXL"],
                    "file_name": ["real/a.jpg", "sdxl/b.png", "sdxl/c.png"],
                },
                features=Features({"image": Image(), "label": Value("string"),
                                   "file_name": Value("string")}),
            )
            data.save_to_disk(str(root / "data"))

            class Processor:
                def __call__(self, images, **kwargs):
                    return {"pixel_values": torch.ones(1, 3, 4, 4)}

            with patch("data.iabench_dataset.CLIPImageProcessor.from_pretrained",
                       return_value=Processor()):
                all_rows = IABenchDataset(str(root))
                self.assertEqual(len(all_rows), 3)
                self.assertEqual(all_rows[0]["file_name"], "real/a.jpg")
                self.assertTrue(all_rows[0]["is_real"].item())
                self.assertEqual(all_rows[0]["pixel_values"].shape, (3, 4, 4))

                selected = IABenchDataset(str(root / "data"),
                                         generators=["SDXL"], max_per_class=1)
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0]["generator"], "SDXL")
                self.assertFalse(selected[0]["is_real"].item())

                with self.assertRaisesRegex(ValueError, "no labels"):
                    IABenchDataset(str(root), generators=["missing"])


if __name__ == "__main__":
    unittest.main()
