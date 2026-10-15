"""Export IABench's original encoded images without resizing or recompression.

Run on an allocated CINECA node:
    python -m data.export_iabench --root ARROW_ROOT --output IMAGE_ROOT

Rerunning verifies and reuses matching files. The image index is published as
complete only after every row has been exported. Existing split manifests remain
valid because label/filename order and the metadata digest are preserved.
"""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

from datasets import Image as ArrowImage
from PIL import Image
from tqdm import tqdm

from data.iabench_dataset import IABenchDataset, IMAGE_INDEX
from data.image_io import retry_image_read


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_images(root, output):
    output = Path(output)
    if output.resolve() in {Path(root).resolve(), (Path(root) / "data").resolve()}:
        raise ValueError("Export to a separate directory from the Arrow dataset")
    source = IABenchDataset(root, processor_name=None)
    if source.image_paths is not None:
        raise ValueError("The source is already an exported image dataset")
    raw = source.data.cast_column("image", ArrowImage(decode=False))
    index_path = output / IMAGE_INDEX
    header = {"format": "iabench-images-v1", "dataset_digest": source.dataset_digest,
              "image_feature": source.data.features.to_dict()["image"]}
    if index_path.exists():
        previous = json.loads(index_path.read_text())
        if any(previous.get(key) != value for key, value in header.items()):
            raise ValueError("Existing export belongs to different dataset metadata")
    else:
        atomic_write(index_path, json.dumps(dict(header, complete=False, rows=[])).encode())

    rows = []
    for i in tqdm(range(len(source)), desc="Export IABench"):
        def encoded_image():
            stored = raw[i]["image"]
            return (stored["bytes"] if stored["bytes"] is not None
                    else Path(stored["path"]).read_bytes())

        payload = retry_image_read(encoded_image)
        with Image.open(BytesIO(payload)) as image:
            suffix = ".jpg" if image.format == "JPEG" else f".{image.format.lower()}"
        relative = f"images/{i // 1000:06d}/{i:09d}{suffix}"
        path = output / relative
        if path.exists():
            if path.is_symlink() or path.read_bytes() != payload:
                raise ValueError(f"Existing image differs from source: {path}")
        else:
            atomic_write(path, payload)
        rows.append({"label": source.source_labels[i], "file_name": source.file_names[i],
                     "image_path": relative})

    index = dict(header, complete=True, rows=rows)
    atomic_write(index_path, json.dumps(index, ensure_ascii=False).encode())
    print(f"Exported {len(rows)} images → {output}")
    print(f"Dataset digest: {source.dataset_digest}; reuse the existing split manifest.")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Original IABench Arrow root or data directory")
    parser.add_argument("--output", required=True, help="Directory for image files and their index")
    args = parser.parse_args()
    export_images(args.root, args.output)


if __name__ == "__main__":
    main()
