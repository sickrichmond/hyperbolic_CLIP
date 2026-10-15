"""Lossless export and split compatibility: python -m tests.test_iabench_export."""
from io import BytesIO
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
import torch

from data.export_iabench import export_images
from data.iabench_dataset import IABenchDataset, IMAGE_INDEX


class IABenchExportTests(unittest.TestCase):
    def test_export_resume_pixels_and_manifest_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "source"
            output = Path(directory) / "exported"
            payloads = []
            for i in range(20):
                image = PILImage.new("RGB", (13, 9), (i * 10, 40, 80))
                image.putpixel((0, 0), (255, 0, 0))
                stream = BytesIO()
                exif = PILImage.Exif()
                exif[274] = 6
                image.save(stream, format="JPEG" if i % 2 else "PNG", exif=exif)
                payloads.append(stream.getvalue())
            labels = ["Real"] * 10 + ["Generator/with@slash"] * 10
            rows = {"image": [{"bytes": payload, "path": None} for payload in payloads],
                    "label": labels, "file_name": ["duplicate.png"] * 20}
            data = Dataset.from_dict(rows, features=Features(
                {"image": Image(), "label": Value("string"), "file_name": Value("string")}))
            data.save_to_disk(str(source_root / "data"), num_shards=3)
            (source_root / "data/state.json").unlink()
            (source_root / "data/dataset_info.json").unlink()
            with patch.object(Image, "decode_example", side_effect=AssertionError("image decoded")), \
                    patch("data.iabench_dataset.CLIPImageProcessor.from_pretrained",
                          side_effect=AssertionError("CLIP loaded")):
                index = export_images(str(source_root), output)
            paths = [output / row["image_path"] for row in index["rows"]]
            self.assertEqual(len(set(paths)), 20)
            self.assertEqual([path.read_bytes() for path in paths], payloads)

            def pixels(images, **kwargs):
                return {"pixel_values": torch.tensor(list(images.tobytes()), dtype=torch.uint8).reshape(
                    1, images.height, images.width, 3)}

            with patch("data.iabench_dataset.CLIPImageProcessor.from_pretrained", return_value=pixels):
                source = IABenchDataset(str(source_root))
                with patch("data.iabench_dataset.load_from_disk", side_effect=AssertionError("Arrow read")), \
                        patch("data.iabench_dataset.ArrowDataset.from_file",
                              side_effect=AssertionError("Arrow read")):
                    cached = IABenchDataset(str(output))
                self.assertEqual(cached.dataset_digest, source.dataset_digest)
                self.assertEqual(cached.samples, source.samples)
                manifest = source.make_split_manifest()
                self.assertEqual(cached.make_split_manifest(), manifest)
                for split in ("train", "val", "test"):
                    self.assertEqual(cached.split_view(manifest, split).indices, manifest[split])
                self.assertEqual(cached.make_split_manifest(max_per_class=10),
                                 source.make_split_manifest(max_per_class=10))
                for i in (0, 1):
                    self.assertEqual(cached._read_image(i).size, source._read_image(i).size)
                    for level in range(7):
                        cached.degraded = source.degraded = level
                        self.assertTrue(torch.equal(cached[i]["pixel_values"], source[i]["pixel_values"]))
                cached.degraded = source.degraded = 0
                cached.train_augment = source.train_augment = True
                random.seed(42)
                expected = source[1]["pixel_values"]
                random.seed(42)
                self.assertTrue(torch.equal(cached[1]["pixel_values"], expected))

            index_path = output / IMAGE_INDEX
            original_index = index_path.read_bytes()
            first_mtime = paths[0].stat().st_mtime_ns
            paths[-1].unlink()
            index_path.write_text(json.dumps(dict(index, complete=False)))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                IABenchDataset(str(output), processor_name=None)
            export_images(str(source_root), output)
            self.assertEqual(paths[0].stat().st_mtime_ns, first_mtime)
            self.assertEqual(paths[-1].read_bytes(), payloads[-1])
            for field, value, message in (("image_path", "../outside.png", "within"),
                                           ("label", "changed", "digest")):
                damaged = json.loads(original_index)
                damaged["rows"][0][field] = value
                index_path.write_text(json.dumps(damaged))
                with self.assertRaisesRegex(ValueError, message):
                    IABenchDataset(str(output), processor_name=None)
            index_path.write_bytes(original_index)
            changed_root = Path(directory) / "changed"
            source.data.select(list(reversed(range(20)))).save_to_disk(str(changed_root / "data"))
            with self.assertRaisesRegex(ValueError, "different dataset metadata"):
                export_images(str(changed_root), output)
            self.assertEqual(index_path.read_bytes(), original_index)
            paths[0].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs from source"):
                export_images(str(source_root), output)
            self.assertEqual(paths[0].read_bytes(), b"changed")
            with self.assertRaisesRegex(ValueError, "separate directory"):
                export_images(str(source_root), source_root)
            with self.assertRaisesRegex(ValueError, "already an exported"):
                export_images(str(output), Path(directory) / "second")


if __name__ == "__main__":
    unittest.main()
