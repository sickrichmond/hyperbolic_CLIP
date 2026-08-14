"""Exhaustively evaluate a registered attribution baseline on one augmented split.

Unlike ``comparison.training.test``, this entrypoint does not create a new
train/validation/test split.  Every valid image below ``--dataset-root`` is
evaluated exactly once with the preprocessing registered for ``--model``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from sklearn import metrics
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


TARGET_CLASSES = ("FLUX", "SD3", "SD3_5", "SDXL")
EXPECTED_NUM_CLASSES = 22
VALID_MODELS = (
    "resnet50",
    "dct",
    "hifi_net",
    "defl",
    "dna",
    "repmix",
    "patch",
    "ucf",
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_CHECKPOINT_HEAD_KEYS = {
    "resnet50": "fc.weight",
    "dct": "feature_extractor.fc.weight",
    "hifi_net": "SegNet.branch_cls_level_4.fc.weight",
    "defl": "nn_classifier.fc3.weight",
    "dna": "backbone.classification_head.1.weight",
    "repmix": "attribution.weight",
    "patch": "net_D.module.convout.weight",
    "ucf": "head_spe.mlp.2.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Balanced total-image cap for smoke tests; omit for exhaustive evaluation.",
    )
    parser.add_argument(
        "--ucf-pretrained",
        type=Path,
        default=None,
        help="ImageNet Xception initialization required while constructing UCF.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build the dataset/model and strictly load the checkpoint, then exit.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def probabilities_from_output(model_name: str, output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Convert every attributor's output contract to image-level probabilities."""
    if "logits" not in output:
        raise KeyError(f"{model_name} inference returned no 'logits' value")
    scores = output["logits"]
    if model_name == "patch":
        if scores.ndim != 4:
            raise ValueError(f"PatchForensics must return (B,C,H,W), got {tuple(scores.shape)}")
        return torch.softmax(scores, dim=1).mean(dim=(2, 3))
    if model_name == "repmix" and scores.ndim == 3 and scores.shape[1] == 1:
        # RepMix represents the number of mixed examples as a separate axis.
        # Test mode has exactly one source image, so remove only that known axis
        # (never an indiscriminate squeeze, which would break batch size one).
        scores = scores[:, 0, :]
    if scores.ndim != 2:
        raise ValueError(f"{model_name} must return (B,C) logits, got {tuple(scores.shape)}")
    return torch.softmax(scores, dim=1)


def image_level_batch_field(
    value: torch.Tensor, batch_size: int, field_name: str
) -> torch.Tensor:
    """Normalize ordinary labels and RepMix's ``(B, 1)`` labels to ``(B,)``."""
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(
            f"{field_name} must have image-level shape ({batch_size},) or "
            f"({batch_size}, 1), got {tuple(value.shape)}"
        )
    return value


def checkpoint_output_classes(
    model_name: str, state_dict: dict[str, torch.Tensor]
) -> int:
    """Read the trained output-head width before constructing or running a model."""
    key = _CHECKPOINT_HEAD_KEYS[model_name]
    if key not in state_dict:
        raise KeyError(
            f"{model_name} checkpoint is missing its output head {key!r}; "
            "it is not a compatible canonical checkpoint"
        )
    weight = state_dict[key]
    if not isinstance(weight, torch.Tensor) or weight.ndim < 2:
        raise ValueError(f"checkpoint output head {key!r} is not a weight tensor")
    return int(weight.shape[0])


@contextmanager
def defl_offline_clip_loader(state_dict: dict[str, torch.Tensor]):
    """Construct DeFL's RN50x16 from its checkpoint on offline compute nodes.

    DeFL calls ``clip.load`` once in its dataset merely to obtain preprocessing
    and once in its model to obtain RN50x16. The canonical trained checkpoint
    contains the complete CLIP state, so the first call can return a parameter-
    free placeholder and the second can build the exact model without a cache
    lookup or download. The monkeypatch is scoped to model/dataset construction.
    """
    prefix = "semantic_extractor.clip_model."
    clip_state = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    required = {
        "visual.conv1.weight",
        "visual.attnpool.positional_embedding",
        "text_projection",
        "token_embedding.weight",
    }
    missing = sorted(required - clip_state.keys())
    if missing:
        raise KeyError(f"DeFL checkpoint is missing embedded CLIP tensors: {missing}")

    output_width = round(
        (clip_state["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
    )
    input_resolution = output_width * 32

    import clip
    from clip import clip as clip_impl
    from clip.model import build_model

    preprocess = clip_impl._transform(input_resolution)
    original_load = clip.load
    call_count = 0

    def offline_load(name: str, device: str = "cpu", **_: Any):
        nonlocal call_count
        if name != "RN50x16":
            raise ValueError(f"offline DeFL loader only supports RN50x16, got {name!r}")
        call_count += 1
        if call_count == 1:
            return torch.nn.Identity().to(device), preprocess
        # build_model deletes optional metadata entries, so pass a shallow copy.
        return build_model(dict(clip_state)).to(device), preprocess

    clip.load = offline_load
    try:
        yield {
            "source": "embedded_in_trained_checkpoint",
            "architecture": "RN50x16",
            "input_resolution": input_resolution,
        }
    finally:
        clip.load = original_load


def balanced_subset_indices(labels: Sequence[int], limit: int | None) -> list[int] | None:
    """Choose a deterministic, round-robin class-balanced subset for smoke tests."""
    if limit is None or limit >= len(labels):
        return None
    if limit <= 0:
        raise ValueError("--max-images must be positive")
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)
    selected: list[int] = []
    offsets = {label: 0 for label in by_label}
    ordered_labels = sorted(by_label)
    while len(selected) < limit:
        made_progress = False
        for label in ordered_labels:
            offset = offsets[label]
            if offset < len(by_label[label]):
                selected.append(by_label[label][offset])
                offsets[label] += 1
                made_progress = True
                if len(selected) == limit:
                    break
        if not made_progress:
            break
    return selected


def _auc_ap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_indices: Sequence[int],
    class_names: Sequence[str],
) -> tuple[float | None, float | None, dict[str, dict[str, float | None]]]:
    per_class: dict[str, dict[str, float | None]] = {}
    aucs: list[float | None] = []
    aps: list[float | None] = []
    for class_index, class_name in zip(class_indices, class_names):
        binary = (labels == class_index).astype(np.int64)
        if np.unique(binary).size < 2:
            auc = ap = None
        else:
            auc = float(metrics.roc_auc_score(binary, probabilities[:, class_index]))
            ap = float(metrics.average_precision_score(binary, probabilities[:, class_index]))
        per_class[class_name] = {"auc": auc, "average_precision": ap}
        aucs.append(auc)
        aps.append(ap)
    return _safe_mean(aucs), _safe_mean(aps), per_class


def _classification_summary(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_names: Sequence[str],
    target_names: Sequence[str],
    target_indices: Sequence[int],
    semantic_labels: np.ndarray,
    semantic_names: dict[int, str],
) -> dict[str, Any]:
    all_indices = list(range(len(output_names)))
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        labels,
        predictions,
        labels=all_indices,
        average=None,
        zero_division=0,
    )
    target_precision = [precision[i] for i in target_indices]
    target_recall = [recall[i] for i in target_indices]
    target_f1 = [f1[i] for i in target_indices]
    # Ranking metrics for classes absent from the augmented dataset are
    # undefined and explicitly reported as null. The macro therefore averages
    # the four finite target-class one-vs-rest scores.
    macro_auc, macro_ap, ranking = _auc_ap(
        labels, probabilities, all_indices, output_names
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(output_names)
    }
    for name in output_names:
        per_class[name].update(ranking[name])
    semantic_accuracy = {}
    for semantic_id in sorted(np.unique(semantic_labels).tolist()):
        mask = semantic_labels == semantic_id
        semantic_accuracy[semantic_names.get(int(semantic_id), str(semantic_id))] = {
            "accuracy": float(np.mean(predictions[mask] == labels[mask])),
            "support": int(mask.sum()),
        }
    return {
        "sample_count": int(labels.size),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean(target_recall)),
        "precision_macro": float(np.mean(target_precision)),
        "recall_macro": float(np.mean(target_recall)),
        "f1_macro": float(np.mean(target_f1)),
        "roc_auc_macro_ovr": macro_auc,
        "average_precision_macro_ovr": macro_ap,
        "macro_average_classes": list(target_names),
        "per_class": per_class,
        "semantic_group_accuracy": semantic_accuracy,
        "confusion_matrix": {
            "labels": list(output_names),
            "matrix": metrics.confusion_matrix(
                labels, predictions, labels=all_indices
            ).tolist(),
        },
    }


def evaluate_protocols(
    probabilities: np.ndarray,
    labels: np.ndarray,
    semantic_labels: np.ndarray,
    class_names: Sequence[str],
    target_names: Sequence[str] = TARGET_CLASSES,
    semantic_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Compute full-head and target-logit-restricted attribution metrics."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    semantic_labels = np.asarray(semantic_labels, dtype=np.int64).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.size:
        raise ValueError("probabilities must be (N,C) and align with labels")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("probability columns do not match class_names")
    if semantic_labels.size != labels.size:
        raise ValueError("semantic_labels do not align with labels")
    missing = [name for name in target_names if name not in class_names]
    if missing:
        raise ValueError(f"target classes missing from output space: {missing}")
    target_indices = [class_names.index(name) for name in target_names]
    unexpected = sorted(set(np.unique(labels).tolist()) - set(target_indices))
    if unexpected:
        raise ValueError(f"dataset contains non-target label indices: {unexpected}")

    semantic_names = semantic_names or {}
    full_predictions = probabilities.argmax(axis=1)
    full = _classification_summary(
        labels,
        full_predictions,
        probabilities,
        class_names,
        target_names,
        target_indices,
        semantic_labels,
        semantic_names,
    )
    off_target = ~np.isin(full_predictions, target_indices)
    off_counts = Counter(class_names[int(i)] for i in full_predictions[off_target])
    full["off_target_prediction_rate"] = float(np.mean(off_target))
    full["off_target_prediction_count"] = int(off_target.sum())
    full["off_target_predictions"] = dict(sorted(off_counts.items()))

    restricted = probabilities[:, target_indices]
    row_sums = restricted.sum(axis=1, keepdims=True)
    restricted = np.divide(
        restricted,
        row_sums,
        out=np.full_like(restricted, 1.0 / len(target_indices)),
        where=row_sums > 0,
    )
    local_labels = np.array([target_indices.index(int(label)) for label in labels])
    local_predictions = restricted.argmax(axis=1)
    restricted_summary = _classification_summary(
        local_labels,
        local_predictions,
        restricted,
        target_names,
        target_names,
        list(range(len(target_names))),
        semantic_labels,
        semantic_names,
    )
    return {"full_22_way": full, "restricted_4_way": restricted_summary}


def enumerate_target_samples(
    dataset_root: Path,
    class_to_label: dict[str, int],
    semantic_relpaths: dict[str, str],
    semantic_to_label: dict[str, int],
) -> list[tuple[str, int, int, str]]:
    """Enumerate every target image without the training loader's filename filter."""
    samples: list[tuple[str, int, int, str]] = []
    for generator in TARGET_CLASSES:
        if generator not in class_to_label:
            raise ValueError(f"target class is absent from the active label map: {generator}")
        for semantic, relpath in semantic_relpaths.items():
            directory = dataset_root / generator / relpath
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                    samples.append(
                        (
                            str(path),
                            class_to_label[generator],
                            semantic_to_label[semantic],
                            semantic,
                        )
                    )
    return samples


def count_valid_files(dataset_root: Path, semantic_relpaths: dict[str, str]) -> int:
    """Independently count images anywhere below the four target class roots."""
    total = 0
    for generator in TARGET_CLASSES:
        generator_root = dataset_root / generator
        if not generator_root.is_dir():
            continue
        for directory, _, filenames in os.walk(generator_root):
            total += sum(Path(name).suffix.lower() in _IMAGE_SUFFIXES for name in filenames)
    return total


def count_candidate_files(dataset_root: Path, semantic_relpaths: dict[str, str]) -> int:
    """Count every file below target roots, including non-image artifacts."""
    total = 0
    for generator in TARGET_CLASSES:
        generator_root = dataset_root / generator
        if not generator_root.is_dir():
            continue
        for _, _, filenames in os.walk(generator_root):
            total += len(filenames)
    return total


def _checkpoint_provenance(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    stat = path.stat()
    best_metrics = checkpoint.get("best_metrics")
    validation_metric = (
        best_metrics.get("val_metric") if isinstance(best_metrics, dict) else None
    )
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_time_utc": dt.datetime.fromtimestamp(
            stat.st_mtime, tz=dt.timezone.utc
        ).isoformat(),
        "epoch": checkpoint.get("epoch"),
        "validation_metric": _jsonable(validation_metric),
        "best_metrics": _jsonable(best_metrics),
        "strict_load": True,
    }


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_is_dirty(repo_root: Path) -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_results(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    json_tmp = output_dir / ".metrics.json.tmp"
    json_tmp.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    json_tmp.replace(json_path)

    full = report["metrics"]["full_22_way"]
    restricted = report["metrics"]["restricted_4_way"]
    summary = {
        "model": report["model"],
        "dataset": report["dataset"]["name"],
        "checkpoint": report["checkpoint"]["path"],
        "checkpoint_epoch": report["checkpoint"]["epoch"],
        "checkpoint_validation_metric": report["checkpoint"]["validation_metric"],
        "git_commit": report["runtime"]["git_commit"],
        "samples": report["dataset"]["evaluated_images"],
        "full_accuracy": full["accuracy"],
        "full_balanced_accuracy": full["balanced_accuracy"],
        "full_precision_macro": full["precision_macro"],
        "full_recall_macro": full["recall_macro"],
        "full_f1_macro": full["f1_macro"],
        "full_roc_auc_macro_ovr": full["roc_auc_macro_ovr"],
        "full_average_precision_macro_ovr": full["average_precision_macro_ovr"],
        "off_target_prediction_rate": full["off_target_prediction_rate"],
        "restricted_accuracy": restricted["accuracy"],
        "restricted_balanced_accuracy": restricted["balanced_accuracy"],
        "restricted_precision_macro": restricted["precision_macro"],
        "restricted_recall_macro": restricted["recall_macro"],
        "restricted_f1_macro": restricted["f1_macro"],
        "restricted_roc_auc_macro_ovr": restricted["roc_auc_macro_ovr"],
        "restricted_average_precision_macro_ovr": restricted[
            "average_precision_macro_ovr"
        ],
    }
    csv_path = output_dir / "summary.csv"
    csv_tmp = output_dir / ".summary.csv.tmp"
    with csv_tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    csv_tmp.replace(csv_path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")

    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.config.is_file():
        raise FileNotFoundError(f"config file does not exist: {args.config}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {args.checkpoint}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {args.dataset_root}")

    # This must happen before importing the registry: the active label map is
    # constructed at module-import time and the selected checkpoints are 22-way.
    os.environ.setdefault("IAB_EXCLUDE_GENERATORS", "dalle3")
    from comparison.dataset.ImageAttributionDataset import DATASET
    from comparison.dataset.ImageAttributionDataset.dataset import (
        model_class_to_label,
        semantic_label_map,
        semantic_to_relpath,
    )
    from comparison.training.attributors import ATTRIBUTOR

    if os.environ["IAB_EXCLUDE_GENERATORS"] != "dalle3":
        raise RuntimeError(
            "These checkpoints require IAB_EXCLUDE_GENERATORS=dalle3 exactly; got "
            f"{os.environ['IAB_EXCLUDE_GENERATORS']!r}"
        )
    if len(model_class_to_label) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"expected a {EXPECTED_NUM_CLASSES}-class label map, "
            f"found {len(model_class_to_label)}"
        )
    if args.model not in ATTRIBUTOR.data:
        raise RuntimeError(f"model {args.model!r} was not registered; check optional dependencies")
    if args.model not in DATASET.data:
        raise RuntimeError(f"dataset adapter {args.model!r} was not registered")

    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"model configuration must be a YAML mapping: {args.config}")
    source_config = dict(config)
    if config.get("model_name") != args.model:
        raise ValueError(
            f"config model_name={config.get('model_name')!r} does not match --model={args.model!r}"
        )
    config["num_classes"] = EXPECTED_NUM_CLASSES
    config["specific_task_number"] = EXPECTED_NUM_CLASSES
    if args.model == "repmix":
        config["inference"] = True
    if args.model == "ucf":
        pretrained = args.ucf_pretrained or Path(config["pretrained"])
        pretrained = pretrained.expanduser().resolve()
        if not pretrained.is_file():
            raise FileNotFoundError(f"UCF Xception initialization not found: {pretrained}")
        config["pretrained"] = str(pretrained)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError(f"checkpoint has no model_state_dict: {args.checkpoint}")
    checkpoint_provenance = _checkpoint_provenance(args.checkpoint, checkpoint)
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint model_state_dict is not a mapping: {args.checkpoint}")
    output_classes = checkpoint_output_classes(args.model, state_dict)
    if output_classes != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"{args.model} checkpoint output head has {output_classes} classes; "
            f"expected {EXPECTED_NUM_CLASSES}"
        )
    # Drop optimizer/scheduler tensors before constructing memory-heavy models.
    del checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("The registered baselines require a CUDA-enabled SLURM job")

    clip_context = (
        defl_offline_clip_loader(state_dict)
        if args.model == "defl"
        else nullcontext(None)
    )
    with clip_context as defl_clip_initialization:
        dataset_class = DATASET[args.model]
        dataset = dataset_class(
            root_dir=str(args.dataset_root),
            num_images_per_semantic_per_class=sys.maxsize,
            degraded=0,
            config=config,
        )
        model = ATTRIBUTOR[args.model](config)
    # The training dataset intentionally filters generated filenames by `_p/_i`
    # conventions and follows an optional `real` symlink. Augmented evaluation
    # instead uses every image in the four requested generator directories.
    dataset.samples = enumerate_target_samples(
        args.dataset_root,
        model_class_to_label,
        semantic_to_relpath,
        semantic_label_map,
    )
    if hasattr(dataset, "N"):
        dataset.N = len(dataset.samples)
    dataset.set_test()
    expected_count = count_valid_files(args.dataset_root, semantic_to_relpath)
    candidate_count = count_candidate_files(args.dataset_root, semantic_to_relpath)
    if len(dataset) != expected_count:
        raise RuntimeError(
            f"dataset adapter enumerated {len(dataset)} images in known semantic cells, "
            f"but a recursive inventory of the four target roots found {expected_count}; "
            "refusing a non-exhaustive evaluation"
        )
    labels = [int(sample[1]) for sample in dataset.samples]
    target_indices = {model_class_to_label[name] for name in TARGET_CLASSES}
    unexpected_labels = sorted(set(labels) - target_indices)
    if unexpected_labels:
        raise RuntimeError(f"augmented dataset contains unexpected labels: {unexpected_labels}")
    if set(labels) != target_indices:
        missing = target_indices - set(labels)
        raise RuntimeError(f"augmented dataset is missing target label indices: {sorted(missing)}")

    selected = balanced_subset_indices(labels, args.max_images)
    eval_dataset = Subset(dataset, selected) if selected is not None else dataset
    collate_fn = getattr(dataset, "collate_fn", None)

    model.load_state_dict(state_dict, strict=True)
    # Once strict loading succeeds the checkpoint-owned CPU weights are no
    # longer needed for inference.
    del state_dict
    device = torch.device("cuda")
    model.to(device)
    model.device = device
    model.eval()

    validation = {
        "model": args.model,
        "dataset": args.dataset_root.name,
        "full_dataset_images": len(dataset),
        "evaluated_images": len(eval_dataset),
        "checkpoint": str(args.checkpoint),
        "checkpoint_output_classes": output_classes,
        "target_class_indices": {
            name: model_class_to_label[name] for name in TARGET_CLASSES
        },
        "strict_checkpoint_load": True,
    }
    print(json.dumps(validation, indent=2))
    if args.validate_only:
        return

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    all_probabilities: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_semantics: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{args.model}:{args.dataset_root.name}"):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
            output = model(batch, inference=True)
            probabilities = probabilities_from_output(args.model, output)
            if probabilities.shape[1] != EXPECTED_NUM_CLASSES:
                raise RuntimeError(
                    f"{args.model} produced {probabilities.shape[1]} classes instead of "
                    f"{EXPECTED_NUM_CLASSES}"
                )
            if not torch.isfinite(probabilities).all():
                raise RuntimeError(f"{args.model} produced non-finite probabilities")
            batch_labels = image_level_batch_field(
                batch["label"], probabilities.shape[0], "label"
            )
            batch_semantics = image_level_batch_field(
                batch["semantic_label"], probabilities.shape[0], "semantic_label"
            )
            all_probabilities.append(probabilities.cpu())
            all_labels.append(batch_labels.cpu())
            all_semantics.append(batch_semantics.cpu())

    probabilities_np = torch.cat(all_probabilities).numpy()
    labels_np = torch.cat(all_labels).numpy()
    semantics_np = torch.cat(all_semantics).numpy()
    class_names = [
        name for name, _ in sorted(model_class_to_label.items(), key=lambda item: item[1])
    ]
    semantic_names = {index: name for name, index in semantic_label_map.items()}
    protocol_metrics = evaluate_protocols(
        probabilities_np,
        labels_np,
        semantics_np,
        class_names,
        semantic_names=semantic_names,
    )
    sample_counts = Counter(class_names[int(label)] for label in labels_np)
    semantic_counts = Counter(semantic_names[int(label)] for label in semantics_np)
    repo_root = Path(__file__).resolve().parents[2]
    report = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "model": args.model,
        "checkpoint": checkpoint_provenance,
        "configuration": {
            "path": str(args.config),
            "source_values": _jsonable(source_config),
            "effective_values": _jsonable(config),
        },
        "dataset": {
            "name": args.dataset_root.name,
            "root": str(args.dataset_root),
            "valid_images": len(dataset),
            "candidate_files": candidate_count,
            "ignored_non_image_files": candidate_count - len(dataset),
            "evaluated_images": len(eval_dataset),
            "max_images": args.max_images,
            "class_counts": dict(sorted(sample_counts.items())),
            "semantic_counts": dict(sorted(semantic_counts.items())),
            "degradation_level": 0,
            "split_policy": "all_valid_images",
        },
        "runtime": {
            "git_commit": _git_commit(repo_root),
            "git_dirty": _git_is_dirty(repo_root),
            "hostname": socket.gethostname(),
            "command": [
                sys.executable,
                "-m",
                "comparison.training.eval_aug_splits",
                *sys.argv[1:],
            ],
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "defl_clip_initialization": defl_clip_initialization,
        },
        "metrics": protocol_metrics,
    }
    _write_results(args.output_dir, report)
    print(f"Wrote {args.output_dir / 'metrics.json'}")
    print(f"Wrote {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
