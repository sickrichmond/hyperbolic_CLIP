"""Extract, fit, and render the 48 incremental IAB checkpoint panels."""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from itertools import chain, product
import json
import os
from pathlib import Path
from runpy import run_path
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np


WORK_ROOT = Path("/leonardo_work/EUHPC_D35_189/hyp_fine_tuning")
FAST_ROOT = Path("/leonardo_scratch/fast/EUHPC_D35_189/hyp_fine_tuning")
METHODS = ("hypclip", "lida", "increfa")
METHOD_LABELS = dict(zip(METHODS, (r"$\mathtt{Anchorage}$", "LIDA", "IncreFA")))
BASE = {
    "seq7": ("real", "SD1_5", "SD3", "PIXART", "FLUX", "mid-5.2", "ideogram"),
    "split14": ("real", "SD3", "PIXART", "CogView3_PLUS", "mid-5.2", "mid-6.0",
                "KANDINSKY", "gemini", "FLUX", "ideogram", "SD1_5", "SDXL",
                "infinity", "grok3"),
}
INCREMENTS = ("SD3_5", "4o", "hunyuan", "janus-pro", "PLAYGROUND_2_5", "SD2_1", "kling")
CLASS_LABELS = {
    "real": "Real", "SD1_5": "SD 1.5", "SD3": "SD 3", "PIXART": "PixArt", "FLUX": "FLUX",
    "mid-5.2": "Midjourney 5.2", "ideogram": "Ideogram", "CogView3_PLUS": "CogView3-Plus",
    "mid-6.0": "Midjourney 6.0", "KANDINSKY": "Kandinsky", "gemini": "Gemini",
    "SDXL": "SDXL", "infinity": "Infinity", "grok3": "Grok 3", "SD3_5": "SD 3.5",
    "4o": "GPT-4o", "hunyuan": "Hunyuan", "janus-pro": "Janus-Pro",
    "PLAYGROUND_2_5": "Playground 2.5", "SD2_1": "SD 2.1", "kling": "Kling",
}
STEPS = range(len(INCREMENTS) + 1)
PANELS = tuple(product(METHODS, BASE, STEPS))
FIT = dict(n_components=2, method="barnes_hut", metric="euclidean", random_state=42,
           perplexity=30, init="pca", learning_rate="auto", max_iter=1000)


def seen_classes(setting, step):
    return BASE[setting] + INCREMENTS[:step]


def checkpoint_path(method, setting, step, fast, igor):
    if method == "hypclip":
        base = {"seq7": "base7_hyp_clean", "split14": "split14_hyp_clean_base"}
        run = f"{setting}_hyp_clean_r300_e10_replay" if step else base[setting]
        return fast / "continual_runs_hyp" / run / f"T{step}.pt"
    if method == "lida":
        run = f"lida_a10_db300_{setting}_{'frozen' if step else 'base'}"
        return fast / "lida_runs" / run / f"T{step}.pt"
    return igor / "increfa_comparison" / "outputs" / f"increfa_iab_{setting}_buf300" / f"task_{step}.pt"


def read_test_rows(manifest):
    raw = manifest.read_bytes()
    rows = json.loads(raw)["test"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of manifest test paths")
    for row in rows:
        if (not isinstance(row, str) or "\\" in row
                or any(part in {"", ".", ".."} for part in row.split("/")) or "/" not in row):
            raise ValueError(f"Invalid relative manifest test path: {row!r}")
    if len(set(rows)) != len(rows):
        raise ValueError("Duplicate paths in manifest test split")
    return sorted(rows), sha256(raw).hexdigest()


def generator(row):
    return row.split("/", 1)[0]


def panel_rows(rows, setting, step, limit=None):
    names = seen_classes(setting, step)
    counts = dict.fromkeys(names, 0)
    selected = []
    for row in rows:
        name = generator(row)
        if name in counts and (limit is None or counts[name] < limit):
            counts[name] += 1
            selected.append(row)
    if missing := [name for name, count in counts.items() if not count]:
        raise ValueError(f"{setting} T{step} has no test rows for {missing}")
    return selected


def check_files(root, rows):
    for row in rows:
        if not (root / row).is_file():
            raise FileNotFoundError(root / row)


def checkpoint_classes(checkpoint, method, setting, step):
    expected = seen_classes(setting, step)
    if method == "increfa":
        order = checkpoint["results"]["class_order"]
        expected_order = [list(BASE[setting]), *([name] for name in INCREMENTS)]
        if order[:step + 1] != expected_order[:step + 1]:
            raise ValueError(f"IncreFA task order differs from {setting} at T{step}: {order}")
        names = tuple(chain.from_iterable(order[:step + 1]))
        if checkpoint["task_id"] != step or checkpoint["num_learned_classes"] != len(names):
            raise ValueError(f"IncreFA checkpoint task/count differs from {setting} T{step}")
    else:
        names = tuple(checkpoint["class_names"])
        step_key = "cl_step" if method == "hypclip" else "step"
        if checkpoint.get(step_key, 0) != step:
            raise ValueError(f"{method} checkpoint step differs from T{step}")
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError(f"{method} {setting} T{step} classes {names} differ from {expected}")
    return names


class RetryReads:
    def __init__(self, dataset, retry):
        self.dataset = dataset
        self.retry = retry

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.retry(lambda: self.dataset[index])


def collect_batches(dataset, encode, device, batch_size, workers):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False,
                        pin_memory=device.type == "cuda", drop_last=False)
    features, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="embedding", mininterval=20):
            vectors, truth = encode(batch)
            features.append(vectors.float().cpu())
            labels.extend(truth)
    return torch.cat(features).numpy(), labels


def native_features(args, checkpoint, names, rows, device):
    import torch
    from torch.utils.data import Subset

    method = args.method
    native = args.igor_root / {"hypclip": "continual_hyp", "lida": "LIDA",
                               "increfa": "increfa_comparison"}[method]
    retry = run_path(str(Path(__file__).resolve().parents[1] / "data/image_io.py"))["retry_image_read"]
    os.chdir(native)
    sys.path.insert(0, str(native))

    if method == "hypclip":
        from evaluate import load_model, manifest_dataset
        from training.poincare import lorentz_to_poincare

        options = argparse.Namespace(dataset_path=str(args.dataset_root),
                                     captions_dir=str(args.captions_dir), max_per_class=None)
        dataset = manifest_dataset(checkpoint, names, {"test": rows}, "test", options)
        by_path = {sample[0].relative_to(args.dataset_root).as_posix(): sample
                   for sample in dataset.samples}
        if len(dataset) != len(rows) or set(by_path) != set(rows):
            raise ValueError("HypCLIP native dataset does not exactly cover the requested test rows")
        dataset.samples = [by_path[row] for row in rows]
        model = load_model(checkpoint, device).eval()

        def encode(batch):
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                spatial, _ = model.encode_image(batch["pixel_values"].to(device, non_blocking=True))
            return spatial, batch["generator"]

        features, labels = collect_batches(dataset, encode, device, args.batch_size, args.num_workers)
        return lorentz_to_poincare(features, checkpoint["curv"]), labels

    if method == "lida":
        from lida.data import FingerprintDataset
        from lida.model import LIDAEncoder
        from lida.retrieval import extract_features

        dataset = FingerprintDataset(str(args.dataset_root), rows, names,
                                     size=checkpoint["image_size"], mode=checkpoint["fingerprint"])
        if [row for row, _ in dataset.items] != rows:
            raise ValueError("LIDA native dataset does not exactly cover the requested test rows")
        model = LIDAEncoder(imagenet_init=False)
        model.backbone.load_state_dict(checkpoint["backbone"], strict=True)
        features, labels = extract_features(model.to(device), RetryReads(dataset, retry), device,
                                            batch_size=args.batch_size, num_workers=args.num_workers)
        return features.numpy(), [names[index] for index in labels.tolist()]

    from models import load_model
    from train_iab import IABTrainer

    trainer = IABTrainer.__new__(IABTrainer)
    trainer.config = deepcopy(checkpoint["config"])
    trainer.config["device"] = str(device)
    trainer.config["data"].update(iab_root=str(args.dataset_root), split_manifest=str(args.manifest))
    trainer.setseed()
    dataset = trainer.dataset = trainer.load_dataset()
    native_order = [[dataset.idx_to_label_str[i] for i in task] for task in dataset.order]
    if native_order != checkpoint["results"]["class_order"]:
        raise ValueError("IncreFA native dataset order differs from checkpoint metadata")
    by_path = {dataset.paths[index]: int(index) for name in names
               for index in dataset.test_indices_by_class[dataset.label_str_to_idx[name]]}
    indices = [by_path[row] for row in rows]
    trainer.model = model = load_model(trainer.config)
    trainer.configure_predefined_families()
    if model.class_to_family != checkpoint["class_to_family"]:
        raise ValueError("IncreFA native family mapping differs from checkpoint metadata")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or any(not key.startswith("backbone.") for key in missing):
        raise ValueError(f"Invalid IncreFA state: missing={missing}, unexpected={unexpected}")
    model.orth_space.update_indicator(len(names))
    model.eval()

    def encode(batch):
        images, labels = batch
        features = model._extract_features(images.to(device, non_blocking=True))
        latent, _ = model.orth_space(features)
        mapped = dataset.apply_label_mapping_batch(labels)
        return latent, [names[index] for index in mapped.tolist()]

    subset = RetryReads(Subset(dataset, indices), retry)
    return collect_batches(subset, encode, device, args.batch_size, args.num_workers)


def panel_file(output, panel, suffix):
    method, setting, step = panel
    return output / setting / method / f"T{step}.{suffix}"


def save_arrays(path, metadata, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.") as stream:
        # Keep the temporary file on the destination filesystem for atomic publication.
        np.savez_compressed(stream, metadata=json.dumps(metadata, sort_keys=True), **arrays)
        stream.flush()
        # Publish a complete artifact without replacing another task's result.
        os.link(stream.name, path)


def load_arrays(path, value, expected, rows):
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(artifact["metadata"].item())
        if any(metadata[key] != expected[key] for key in expected):
            raise ValueError(f"Provenance mismatch in {path}. Use a fresh output directory.")
        if artifact["image_ids"].tolist() != rows:
            raise ValueError(f"Test rows differ in {path}")
        labels = artifact["labels"]
        if labels.tolist() != [generator(row) for row in rows]:
            raise ValueError(f"Ground-truth labels differ in {path}")
        array = artifact[value]
        if (array.ndim != 2 or array.shape[0] != len(rows) or not np.isfinite(array).all()
                or (value == "coordinates" and array.shape[1] != 2)):
            raise ValueError(f"Invalid {value} in {path}")
        return array, labels, metadata


def fit_tsne(features):
    import inspect
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if len(features) <= FIT["perplexity"]:
        raise ValueError(f"t-SNE needs more than {FIT['perplexity']} test rows")
    pca_dimensions = min(50, len(features)) if features.shape[1] > 50 else None
    if pca_dimensions is not None:
        features = PCA(n_components=pca_dimensions, svd_solver="randomized",
                       random_state=FIT["random_state"]).fit_transform(features)
    settings = FIT.copy()
    if "max_iter" not in inspect.signature(TSNE).parameters:
        settings["n_iter"] = settings.pop("max_iter")
    coordinates = TSNE(**settings).fit_transform(np.ascontiguousarray(features))
    return coordinates, {**FIT, "pca_dimensions": pca_dimensions, "sklearn_version": sklearn.__version__}


def permanent_colors():
    import colorcet

    names = seen_classes("split14", 7)
    return {"real": "#808080", **dict(zip(names[1:], colorcet.palette["glasbey_dark"]))}


def panel_colors(setting, step, colors):
    names = seen_classes(setting, step)
    result = {name: colors[name] for name in names}
    if step:
        result[names[-1]] = "#000000"
    return result


def draw_panel(axis, coordinates, labels, colors):
    for name, color in colors.items():
        points = coordinates[labels == name]
        opacity = 0.9 if color == "#000000" else 0.55
        axis.scatter(points[:, 0], points[:, 1], color=color, s=1.6, alpha=opacity,
                     linewidths=0, rasterized=True, label=CLASS_LABELS[name])
    minimum, maximum = coordinates.min(axis=0), coordinates.max(axis=0)
    midpoint = (minimum + maximum) / 2
    half_span = 0.55 * max(float((maximum - minimum).max()), 1e-6)
    axis.set(xlim=(midpoint[0] - half_span, midpoint[0] + half_span),
             ylim=(midpoint[1] - half_span, midpoint[1] + half_span),
             xticks=midpoint[0] + half_span * np.array([-0.5, 0, 0.5]),
             yticks=midpoint[1] + half_span * np.array([-0.5, 0, 0.5]))
    axis.tick_params(length=0, labelbottom=False, labelleft=False)
    axis.set_axisbelow(True)
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.6)


def save_figure(figure, path):
    for extension in ("png", "pdf"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")


def identity(panel, digest, limit):
    method, setting, step = panel
    return dict(method=method, setting=setting, step=step, manifest_sha256=digest,
                split="test", max_per_class=limit)


def render(args, panels, rows, digest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_theme(style="whitegrid", rc={"grid.color": "#e8e8e8", "grid.linewidth": 0.6})
    colors = permanent_colors()
    legend_box = dict(frameon=True, fancybox=False, framealpha=1, edgecolor="#777777")
    for setting in BASE:
        selected = [panel for panel in panels if panel[1] == setting]
        if not selected:
            continue
        full_grid = len(selected) == 24
        if full_grid:
            grid, axes = plt.subplots(3, 8, figsize=(18, 7.95))
            grid.subplots_adjust(left=0.041, right=0.994, top=0.927, bottom=0.117,
                                 wspace=0.025, hspace=0.028)
        for panel in selected:
            method, _, step = panel
            expected = identity(panel, digest, args.max_per_class)
            coordinates, labels, metadata = load_arrays(
                panel_file(args.output_dir, panel, "points.npz"), "coordinates", expected,
                panel_rows(rows, setting, step, args.max_per_class))
            if any(metadata["tsne"][key] != value for key, value in FIT.items()):
                raise ValueError(f"Unexpected t-SNE fit settings for {panel}")
            panel_palette = panel_colors(setting, step, colors)
            fig, axis = plt.subplots(figsize=(6.5, 6.2), layout="constrained")
            draw_panel(axis, coordinates, labels, panel_palette)
            axis.set_title(f"T{step}", fontsize=18, pad=6)
            axis.set_ylabel(METHOD_LABELS[method], fontsize=16, labelpad=7)
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=4,
                        fontsize=9, markerscale=3, **legend_box)
            save_figure(fig, panel_file(args.output_dir, panel, "plot"))
            plt.close(fig)
            if full_grid:
                axis = axes[METHODS.index(method), step]
                draw_panel(axis, coordinates, labels, panel_palette)
                if method == METHODS[0]:
                    axis.set_title(f"T{step}", fontsize=18, pad=6)
                if step == 0:
                    axis.set_ylabel(METHOD_LABELS[method], fontsize=16, labelpad=7)
        if full_grid:
            legend = {CLASS_LABELS[name]: colors[name] for name in seen_classes(setting, 7)}
            legend["New at this step"] = "#000000"
            handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                              markersize=4, label=name) for name, color in legend.items()]
            grid.legend(handles=handles, loc="lower left", mode="expand",
                        bbox_to_anchor=(0.041, 0.022, 0.953, 0),
                        ncols=(len(handles) + 1) // 2, fontsize=11, labelspacing=0.35,
                        borderaxespad=0, handlelength=1, handletextpad=0.4, **legend_box)
            save_figure(grid, args.output_dir / f"{setting}_grid")
            plt.close(grid)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "extract", "fit", "render"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fast-root", type=Path, default=FAST_ROOT)
    parser.add_argument("--igor-root", type=Path, default=WORK_ROOT / "hyperbolic_CLIP_igor")
    parser.add_argument("--dataset-root", type=Path, default=FAST_ROOT.parent / "datasets/iab_dataset")
    parser.add_argument("--manifest", type=Path, default=WORK_ROOT / "split_manifest_22cls.json")
    parser.add_argument("--captions-dir", type=Path, default=WORK_ROOT / "iab_captions")
    parser.add_argument("--checkpoint", type=Path, help="Override the checkpoint for a single panel")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        help="Limit preflight or standalone rendering to these methods")
    parser.add_argument("--setting", choices=BASE)
    parser.add_argument("--step", type=int, choices=STEPS)
    parser.add_argument("--task-id", type=int, choices=range(len(PANELS)), metavar="0..47")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-per-class", type=int, help="Smoke tests only")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    selection = (args.method, args.setting, args.step)
    if args.task_id is not None:
        if any(value is not None for value in selection):
            parser.error("--task-id and --method/--setting/--step are mutually exclusive")
        args.method, args.setting, args.step = PANELS[args.task_id]
        selection = (args.method, args.setting, args.step)
    single = all(value is not None for value in selection)
    if not single and (args.command in {"extract", "fit"} or
                       args.setting is not None or args.step is not None):
        parser.error("Select a panel with --task-id or --method, --setting and --step")
    if args.command == "preflight" and single:
        parser.error("Preflight always checks all panels")
    if args.methods and (any(value is not None for value in selection) or
                         args.command not in {"preflight", "render"}):
        parser.error("--methods applies only to preflight and render without a panel selection")
    if args.checkpoint and not single:
        parser.error("--checkpoint requires a single panel")
    if args.batch_size < 1 or args.num_workers < 0 or (args.max_per_class is not None
                                                     and args.max_per_class < 1):
        parser.error("Batch size and per-class cap must be positive, workers must be nonnegative")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    methods = args.methods or ((args.method,) if args.method else METHODS)
    panels = (selection,) if single else tuple(panel for panel in PANELS if panel[0] in methods)
    return args, panels


def main(argv=None):
    args, panels = parse_args(argv)
    rows, digest = read_test_rows(args.manifest)
    if args.command == "preflight":
        missing = [str(path) for panel in panels
                   if not os.access(path := checkpoint_path(*panel, args.fast_root, args.igor_root),
                                    os.R_OK) or not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing/unreadable checkpoints:\n" + "\n".join(missing))
        all_rows = set()
        for setting in BASE:
            selected = panel_rows(rows, setting, 7)
            all_rows.update(selected)
            print(f"{setting} T7: {len(selected)} test rows "
                  f"{dict(Counter(map(generator, selected)))}", flush=True)
        check_files(args.dataset_root, sorted(all_rows))
        print(f"Preflight passed: {len(panels)} checkpoints, {len(all_rows)} unique test rows, "
              f"manifest sha256 {digest}", flush=True)
        return
    if args.command == "render":
        render(args, panels, rows, digest)
        return

    start = perf_counter()
    panel, = panels
    method, setting, step = panel
    rows = panel_rows(rows, setting, step, args.max_per_class)
    checkpoint_pathname = args.checkpoint or checkpoint_path(*panel, args.fast_root, args.igor_root)
    stat = checkpoint_pathname.stat()
    metadata = {**identity(panel, digest, args.max_per_class), "checkpoint": str(checkpoint_pathname),
                "checkpoint_size": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns}
    feature_path = panel_file(args.output_dir, panel, "features.npz")
    points_path = panel_file(args.output_dir, panel, "points.npz")
    if args.command == "extract":
        if feature_path.exists():
            load_arrays(feature_path, "features", metadata, rows)
            print(f"Already extracted: {feature_path}", flush=True)
            return
        import torch

        check_files(args.dataset_root, rows)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA extraction requires an allocated GPU")
        checkpoint = torch.load(checkpoint_pathname, map_location="cpu", weights_only=False)
        names = checkpoint_classes(checkpoint, method, setting, step)
        metadata["checkpoint_classes"] = names
        features, labels = native_features(args, checkpoint, names, rows, device)
        if labels != list(map(generator, rows)) or len(features) != len(rows):
            raise ValueError("Extracted features and manifest rows are out of sync")
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError(f"Invalid extracted features for {panel}")
        save_arrays(feature_path, metadata, image_ids=rows, labels=labels, features=features)
        print(f"Extracted {features.shape} in {perf_counter() - start:.1f}s: {feature_path}", flush=True)
        return
    if points_path.exists():
        _, _, existing = load_arrays(points_path, "coordinates", metadata, rows)
        if any(existing["tsne"][key] != value for key, value in FIT.items()):
            raise ValueError(f"Unexpected t-SNE fit settings in {points_path}")
        print(f"Already fitted: {points_path}", flush=True)
        return
    features, labels, metadata = load_arrays(feature_path, "features", metadata, rows)
    coordinates, metadata["tsne"] = fit_tsne(features)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Nonfinite t-SNE coordinates for {panel}")
    save_arrays(points_path, metadata, image_ids=rows, labels=labels, coordinates=coordinates)
    print(f"Fitted {coordinates.shape} in {perf_counter() - start:.1f}s: {points_path}", flush=True)


if __name__ == "__main__":
    main()
