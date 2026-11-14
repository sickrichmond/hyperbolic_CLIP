"""Adapt Igor's LIDA evaluate.py to score external queries against its saved IAB gallery.

Run with Igor's LIDA checkout on PYTHONPATH. The encoder, fingerprint, cosine
retrieval, and metric definitions come from that checkout.
"""

import argparse
import json
from pathlib import Path

from comparison.training.test_baselines_styled import collect_omnifake_samples, collect_samples
from comparison.training.test_hypclip_styled import discover_styles


def external_queries(dataset, root, class_names):
    if dataset == "omnifake":
        yield root, collect_omnifake_samples(root, class_names)
        return

    from comparison.dataset.ImageAttributionDataset.dataset import (
        semantic_label_map, semantic_to_relpath,
    )

    for style, _ in discover_styles(root, class_names):
        yield style, collect_samples(style, class_names, semantic_to_relpath, semantic_label_map)


class QueryFingerprints:
    def __init__(self, samples, size, mode, jpeg99):
        self.samples = samples
        self.size = size
        self.mode = mode
        self.jpeg99 = jpeg99

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from data.degradations import apply_jpeg99
        from data.image_io import open_image_retry
        path, label, *_ = self.samples[index]
        image = open_image_retry(path)
        if self.jpeg99:
            image = apply_jpeg99(image)
        return fingerprint_image(image, self.size, self.mode), label


def fingerprint_image(image, size, mode):
    from lida.fingerprint import NORMALIZE, low_bit_fingerprint

    return NORMALIZE(low_bit_fingerprint(image, size, mode))


class RetryGalleryReads:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from data.image_io import retry_image_read

        return retry_image_read(lambda: self.dataset[index])


def restore_gallery(checkpoint, gallery_root, class_names):
    from lida.data import FingerprintDataset, class_of
    from lida.model import LIDAEncoder

    database = checkpoint["database"]
    gallery_paths = []
    for name in class_names:
        paths = database.get(name)
        if not paths or any(class_of(path) != name or ".." in Path(path).parts for path in paths):
            raise ValueError(f"Invalid registered gallery for {name!r}")
        gallery_paths.extend(paths)
    fp = dict(size=checkpoint["image_size"], mode=checkpoint["fingerprint"])
    gallery = RetryGalleryReads(FingerprintDataset(str(gallery_root), gallery_paths, class_names, **fp))
    encoder = LIDAEncoder(imagenet_init=False)
    encoder.backbone.load_state_dict(checkpoint["backbone"], strict=True)
    return encoder, gallery, fp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gallery-root", required=True, type=Path)
    parser.add_argument("--query-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("styled", "omnifake"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers must be non-negative")
    if not args.gallery_root.is_dir():
        parser.error(f"Gallery root is not a directory: {args.gallery_root}")
    if args.gallery_root.resolve() == args.query_root.resolve():
        parser.error("Gallery and query roots must be separate")

    import numpy as np
    import torch
    from lida.retrieval import class_scores, extract_features, rank1_map, summarize

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = list(checkpoint["class_names"])
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Checkpoint requires a nonempty, unique class order")
    queries = list(external_queries(args.dataset, args.query_root, class_names))
    encoder, gallery, fp = restore_gallery(checkpoint, args.gallery_root, class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    run = dict(device=device, batch_size=args.batch_size, num_workers=args.num_workers)
    g_feat, g_lab = extract_features(encoder, gallery, desc="gallery", **run)
    gallery_labels = g_lab.numpy()

    jpeg99 = args.dataset == "styled"
    for root, samples in queries:
        q_feat, q_lab = extract_features(
            encoder, QueryFingerprints(samples, jpeg99=jpeg99, **fp), desc=root.name, **run)
        truth = q_lab.numpy()
        scores, rank1_sum, map_sum = [], 0.0, 0.0
        # Keep at most one query block's full gallery similarities in memory.
        for start in range(0, len(truth), 4096):
            query_slice = slice(start, start + 4096)
            block_scores, similarities = class_scores(
                q_feat[query_slice], g_feat, g_lab, len(class_names))
            labels = truth[query_slice]
            rank1, mean_ap = rank1_map(similarities, labels, gallery_labels)
            scores.append(block_scores)
            rank1_sum += rank1 * len(labels)
            map_sum += mean_ap * len(labels)
            del similarities

        result = summarize(truth, np.concatenate(scores), class_names,
                           checkpoint["old_classes"], checkpoint["new_classes"])
        result.update({
            "model_name": "lida_retrieval", "checkpoint": str(args.checkpoint.resolve()),
            "step": checkpoint["step"], "registered_only": checkpoint["registered_only"],
            "class_names": class_names, "old_classes": checkpoint["old_classes"],
            "new_classes": checkpoint["new_classes"], "gallery_root": str(args.gallery_root.resolve()),
            "gallery_size": len(gallery), "query_root": str(root.resolve()),
            "dataset": "styled_iab" if jpeg99 else "OmniFake",
            "evaluation_condition": "jpeg99" if jpeg99 else "clean",
            "image_size": fp["size"], "fingerprint": fp["mode"],
            "rank1": rank1_sum / len(truth), "mAP": map_sum / len(truth),
            "num_images": result["n_images"], "accuracy": result["overall_acc"],
            "balanced_accuracy": result["balanced_acc"], "auc": result["auc_macro_ovr"],
            "ap": result["ap_macro_ovr"], "confusion_matrix": result["confusion"],
            "support_per_class": [result["per_class_n"][name] for name in class_names],
            "recall_per_class": [result["per_class_acc"][name] for name in class_names],
        })
        if jpeg99:
            result.update(style=root.name, jpeg_quality=99)
        else:
            result["split"] = "val"
        output = args.output_dir / root.name if jpeg99 else args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        path = output / "metrics.json"
        with path.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(f"{root.name}: overall accuracy {result['overall_acc']:.2%} | "
              f"balanced accuracy {result['balanced_acc']:.2%} | "
              f"{len(truth)} queries | {path}", flush=True)


if __name__ == "__main__":
    main()
