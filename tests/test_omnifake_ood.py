import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from data import omnifake
from scripts import download_omnifake_test as download


class OmniFakeTest(unittest.TestCase):
    def test_selective_extraction_spans_volumes_and_validates_existing_images(self):
        expected = {folder: bytes(range(256)) + folder.encode()
                    for folder in omnifake.OMNIFAKE_TO_IAB}
        buffer = io.BytesIO()
        with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
            archive.writestr('train/GPT4_o/excluded.png', b'train')
            archive.writestr('val/real/excluded.png', b'real')
            for folder, payload in expected.items():
                archive.writestr(f'val/{folder}/test.png', payload)
                archive.writestr(f'val/other/{folder}.png', b'other')
        with ZipFile(buffer) as archive:
            entries = archive.infolist()
            directory_start = archive.start_dir

        data = buffer.getvalue()
        volume_size = 300
        for entry in entries:
            entry.volume, entry.header_offset = divmod(entry.header_offset, volume_size)
        final_disk, directory_offset = divmod(directory_start, volume_size)
        ranges = []

        def fetch_range(disk, start, end):
            absolute = disk * volume_size
            ranges.append((absolute + start, absolute + end))
            return data[absolute + start:absolute + end]

        with TemporaryDirectory() as temporary, patch.multiple(
            download, VOLUME_SIZE=volume_size, FINAL_DISK=final_disk,
            DIRECTORY_OFFSET=directory_offset, IMAGES_PER_GENERATOR=1,
        ), patch.object(download, 'fetch_range', side_effect=fetch_range), patch.object(
            omnifake, 'IMAGES_PER_GENERATOR', 1,
        ):
            root = Path(temporary)
            val = root / 'val'
            val.mkdir()
            selection = download.select_test_images(entries)
            for folder, members in selection.items():
                download.extract_generator(val / folder, members)

            samples = omnifake.discover_test_samples(root)
            self.assertEqual({path.parent.name: path.read_bytes() for path, _ in samples}, expected)
            self.assertEqual({path.parent.name: label for path, label in samples},
                             omnifake.OMNIFAKE_TO_IAB)
            selected_ranges = [entry[:2] for members in selection.values() for entry in members.values()]
            self.assertGreater(len(ranges), len(selected_ranges))
            self.assertEqual(sum(end - start for start, end in ranges),
                             sum(end - start for start, end in selected_ranges))
            self.assertTrue(all(any(lo <= start < end <= hi for lo, hi in selected_ranges)
                                for start, end in ranges))

            ranges.clear()
            download.extract_generator(val / 'GPT4_o', selection['GPT4_o'])
            self.assertEqual(ranges, [])
            (val / 'GPT4_o' / 'test.png').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'differs from the archive'):
                download.extract_generator(val / 'GPT4_o', selection['GPT4_o'])
            (val / 'GPT4_o' / 'test.png').unlink()
            with self.assertRaisesRegex(ValueError, 'expected 1 test images'):
                omnifake.discover_test_samples(root)


if __name__ == '__main__':
    unittest.main()
