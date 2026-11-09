"""Download the ten shared IAB classes from OmniFake's official val (test) ZIP.

Run as python -m scripts.download_omnifake_test --output DATA_ROOT.
"""

import argparse
from http.client import HTTPException
from io import BytesIO
from itertools import chain, pairwise
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import time
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile
import zlib

from data.omnifake import IMAGE_EXTENSIONS, IMAGES_PER_GENERATOR, OMNIFAKE_TO_IAB


# These offsets and sizes belong to this immutable Hub revision.
REVISION = '7b59ac271edaba6f0f93e1a4840979679b129332'
DATA_URL = f'https://huggingface.co/datasets/MoeNew/OmniFake/resolve/{REVISION}/data'
VOLUME_SIZE = 5 * 2**30
FINAL_DISK = 16
FINAL_SIZE = 2474618737
DIRECTORY_OFFSET = 2455231014
DIRECTORY_SIZE = 19387625
ENTRY_COUNT = 180047
CHUNK_SIZE = 32 * 2**20
LOCAL_HEADER = struct.Struct('<4s5H3I2H')


def fetch_range(disk, start, end):
    name = 'val.zip' if disk == FINAL_DISK else f'val.z{disk + 1:02d}'
    size = FINAL_SIZE if disk == FINAL_DISK else VOLUME_SIZE
    if not (0 <= disk <= FINAL_DISK and 0 <= start < end <= size):
        raise BadZipFile(f'Invalid byte range for {name}: {start}-{end}')
    expected_range = f'bytes {start}-{end - 1}/{size}'
    request = Request(f'{DATA_URL}/{name}', headers={'Range': f'bytes={start}-{end - 1}'})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90) as response:
                if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                    raise ValueError(f'{name}: server did not honor range {expected_range}')
                data = response.read(end - start + 1)
            if len(data) != end - start:
                raise OSError(f'{name}: incomplete range {expected_range}')
            return data
        except (OSError, HTTPException):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def read_span(start, end):
    data = bytearray()
    while start < end:
        disk, offset = divmod(start, VOLUME_SIZE)
        stop = min(VOLUME_SIZE, offset + end - start)
        data.extend(fetch_range(disk, offset, stop))
        start += stop - offset
    return data


def read_catalog():
    data = fetch_range(FINAL_DISK, DIRECTORY_OFFSET, DIRECTORY_OFFSET + DIRECTORY_SIZE)
    # Give zipfile a single-disk directory for metadata parsing. Entry offsets
    # and volume numbers still refer to the original data.
    footer = struct.pack('<4sQ2H2I4Q', b'PK\x06\x06', 44, 45, 45, 0, 0,
                         ENTRY_COUNT, ENTRY_COUNT, DIRECTORY_SIZE, 0)
    footer += struct.pack('<4sIQI', b'PK\x06\x07', 0, DIRECTORY_SIZE, 1)
    footer += struct.pack('<4s4H2IH', b'PK\x05\x06', 0, 0, 65535, 65535,
                          DIRECTORY_SIZE, 0, 0)
    with ZipFile(BytesIO(data + footer)) as archive:
        entries = archive.infolist()
    if len(entries) != ENTRY_COUNT:
        raise BadZipFile(f'Expected {ENTRY_COUNT} catalog entries, found {len(entries)}')
    return entries


def select_test_images(entries):
    images = {name: {} for name in OMNIFAKE_TO_IAB}
    directory_start = FINAL_DISK * VOLUME_SIZE + DIRECTORY_OFFSET
    ordered = sorted(entries, key=lambda entry: (entry.volume, entry.header_offset))
    for entry, next_entry in pairwise(chain(ordered, [None])):
        parts = entry.filename.split('/')
        if len(parts) != 3 or parts[0] != 'val' or parts[1] not in images:
            continue
        _, generator, name = parts
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        start = entry.volume * VOLUME_SIZE + entry.header_offset
        end = (next_entry.volume * VOLUME_SIZE + next_entry.header_offset
               if next_entry is not None else directory_start)
        if (name in images[generator] or '\\' in name or '\x00' in entry.orig_filename
                or not 0 <= entry.header_offset < VOLUME_SIZE
                or not 0 <= start < end <= directory_start
                or not entry.compress_size < end - start <= CHUNK_SIZE
                or not 0 < entry.file_size <= CHUNK_SIZE):
            raise BadZipFile(f'Invalid or duplicate member: {entry.filename}')
        images[generator][name] = (start, end, entry)
    for generator, members in images.items():
        if len(members) != IMAGES_PER_GENERATOR:
            raise ValueError(f'{generator}: expected {IMAGES_PER_GENERATOR} test images, '
                             f'found {len(members)}')
    return images


def adjacent_chunks(members):
    chunk = []
    for name, (start, end, entry) in members.items():
        if chunk and (start != chunk[-1][2] or end - chunk[0][1] > CHUNK_SIZE):
            yield chunk
            chunk = []
        chunk.append((name, start, end, entry))
    if chunk:
        yield chunk


def read_image(data, entry):
    sig, _, flags, method, _, _, _, _, _, name_len, extra_len = LOCAL_HEADER.unpack_from(data)
    name_start = LOCAL_HEADER.size
    name_end = name_start + name_len
    image_start = name_end + extra_len
    image_end = image_start + entry.compress_size
    encoding = 'utf-8' if flags & (1 << 11) else 'cp437'
    if (sig != b'PK\x03\x04' or flags & 1 or method != ZIP_DEFLATED
            or method != entry.compress_type or image_end > len(data)
            or bytes(data[name_start:name_end]).decode(encoding) != entry.orig_filename):
        raise BadZipFile(f'Invalid local header for {entry.filename}')
    inflater = zlib.decompressobj(-15)
    payload = inflater.decompress(data[image_start:image_end], entry.file_size + 1)
    if (not inflater.eof or inflater.unused_data or len(payload) != entry.file_size
            or zlib.crc32(payload) != entry.CRC):
        raise BadZipFile(f'Image did not pass ZIP verification: {entry.filename}')
    return payload


def extract_generator(destination, members):
    if destination.is_symlink():
        raise ValueError(f'Expected a real directory: {destination}')
    if destination.exists():
        if set(members) != {path.name for path in destination.iterdir()}:
            raise ValueError(f'Existing directory has different test membership: {destination}')
        for name, (_, _, entry) in members.items():
            path = destination / name
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry.file_size
                    or zlib.crc32(path.read_bytes()) != entry.CRC):
                raise ValueError(f'Existing test image differs from the archive: {path}')
        print(f'{destination.name}: verified {len(members)} existing images', flush=True)
        return

    with TemporaryDirectory(prefix=f'.{destination.name}-', dir=destination.parent) as staging:
        pending = Path(staging) / destination.name
        pending.mkdir()
        for chunk in adjacent_chunks(members):
            base = chunk[0][1]
            data = memoryview(read_span(base, chunk[-1][2]))
            for name, start, end, entry in chunk:
                (pending / name).write_bytes(read_image(data[start - base:end - base], entry))
        pending.rename(destination)
    print(f'{destination.name}: downloaded {len(members)} test images', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Directory to receive val/')
    args = parser.parse_args()
    images = select_test_images(read_catalog())
    val = args.output / 'val'
    val.mkdir(parents=True, exist_ok=True)
    print(f'OmniFake {REVISION} → {val}', flush=True)
    for generator, members in images.items():
        extract_generator(val / generator, members)


if __name__ == '__main__':
    main()
