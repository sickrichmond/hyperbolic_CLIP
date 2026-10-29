"""Frozen two-source manifests and image-only views for the hierarchy experiment."""
from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from PIL import Image
from torch.utils.data import Dataset

from data.degradations import apply_degradation
from data.iabench_dataset import IABenchDataset, manifest_digest
from data.image_io import retry_image_read
from training.families import INCREFA_FAMILIES

METHODS = ('ours', 'resnet50', 'hifi_net')
IAB_GENERATORS = ('grok3', 'janus-pro', 'infinity')
NEARBY = ('CompVis@stable-diffusion-v1-4', 'stabilityai@stable-diffusion-2-1', 'stabilityai@sdxl-turbo')
UNSEEN_LINES = {
    'FLUX': ('black-forest-labs@FLUX.1-dev', 'black-forest-labs@FLUX.1-schnell'),
    'PixArt': ('PixArt-alpha@PixArt-XL-2-1024-MS',),
    'CogView': ('cogview4',),
    'Imagen': ('Imagen3',),
}
UNSEEN = (*NEARBY, *(name for group in UNSEEN_LINES.values() for name in group), 'StyleGAN-XL', 'janus-pro')
EXPLORATORY = ('midjourneyv6', 'nano-banana')
FAMILY_GROUPS = {family: list(names) for family, names in INCREFA_FAMILIES.items() if family != 'unknown'}
FAMILY_GROUPS['autoregressive'] += list(IAB_GENERATORS)
FORMAT = 'iab-hybrid-hierarchy-v1'


def protocol_digest(descriptor):
    return manifest_digest({key: value for key, value in descriptor.items() if key != 'roots'})


def write_json(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Refusing to replace a different experiment file: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def describe_sources(bench, bench_manifest, iab_root, iab_manifest):
    """Validate source membership and derive the fixed protocol without image decoding."""
    expected = {name for group in INCREFA_FAMILIES.values() for name in group}
    if set(bench_manifest['class_names']) != expected or bench_manifest['class_names'][0] != 'real':
        raise ValueError('The complete 29-class IABench manifest is required, real first')
    if (bench_manifest['seed'], bench_manifest['val_frac'], bench_manifest['test_frac']) != (42, .1, .1):
        raise ValueError('IABench requires the existing seed-42 80/10/10 manifest')
    bench.split_view(bench_manifest, 'test')
    if iab_manifest.get('seed') != 42 or iab_manifest.get('use_semantic_split') is not False:
        raise ValueError('IAB requires an existing seed-42 non-semantic harness manifest')
    names = [*bench_manifest['class_names'], *IAB_GENERATORS]
    seen = [name for name in names if name not in UNSEEN + EXPLORATORY]
    records, splits, used_paths, files = {}, {}, set(), []
    iab_root = Path(iab_root).resolve()
    for split in ('train', 'val', 'test'):
        ids = []
        for row in bench_manifest[split]:
            key = f'iabench:{row}'
            records[key] = dict(generator=bench.source_labels[row], source='iabench')
            ids.append(key)
        counts = Counter()
        for relative in iab_manifest[split]:
            if not isinstance(relative, str) or not relative:
                raise ValueError('IAB paths must be nonempty relative strings')
            path = PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts or str(path) != relative:
                raise ValueError(f'Unsafe or noncanonical IAB path: {relative}')
            if relative in used_paths:
                raise ValueError(f'Duplicate or overlapping IAB split membership: {relative}')
            used_paths.add(relative)
            generator = path.parts[0]
            if generator not in IAB_GENERATORS:
                continue
            absolute = (iab_root / relative).resolve()
            if not absolute.is_relative_to(iab_root) or not absolute.is_file():
                raise ValueError(f'Missing or escaped IAB image: {relative}')
            key = f'iab:{relative}'
            records[key] = dict(generator=generator, source='iab')
            ids.append(key)
            counts[generator] += 1
            files.append([relative, absolute.stat().st_size])
        if any(counts[name] == 0 for name in IAB_GENERATORS):
            raise ValueError(f'Every imported IAB generator needs existing {split} rows')
        splits[split] = ids if split == 'test' else [key for key in ids if records[key]['generator'] in seen]
    if len(records) != sum(len(bench_manifest[s]) for s in ('train', 'val', 'test')) + len(files):
        raise ValueError('Source sample identifiers must be unique across splits')
    descriptor = dict(
        format=FORMAT, seed=42, class_names=names, seen_classes=seen, unseen_classes=list(UNSEEN),
        exploratory_classes=list(EXPLORATORY), family_groups=FAMILY_GROUPS,
        nearby_diffusion=list(NEARBY), unseen_lines={key: list(value) for key, value in UNSEEN_LINES.items()},
        source_manifest_digests=dict(iabench=manifest_digest(bench_manifest), iab=manifest_digest(iab_manifest)),
        source_digests=dict(iabench=bench.dataset_digest, iab=manifest_digest(sorted(files))),
        splits=splits, routing='ground_truth_membership', train_augment=False,
        iab_grok_crop=[100, 50],
    )
    return descriptor, records


def prepare_experiment(iabench_root, iabench_manifest, iab_root, iab_manifest, output_dir):
    bench = IABenchDataset(iabench_root, processor_name=None)
    bm = json.loads(Path(iabench_manifest).read_text())
    im = json.loads(Path(iab_manifest).read_text())
    descriptor, _ = describe_sources(bench, bm, iab_root, im)
    descriptor['roots'] = dict(iabench=str(Path(iabench_root).resolve()), iab=str(Path(iab_root).resolve()))
    payloads = {'iabench.splits.json': bm, 'iab.splits.json': im, 'experiment.json': descriptor}
    output = Path(output_dir)
    for name, value in payloads.items():
        path = output / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'Refusing to replace a different experiment file: {path}')
    for name, value in payloads.items():
        write_json(output / name, value, immutable=True)
    print(f'Prepared {output / "experiment.json"}: 20 seen, 10 scored unseen, 2 exploratory')
    return output / 'experiment.json'


class ExperimentData:
    def __init__(self, experiment, iabench_root=None, iab_root=None):
        self.path = Path(experiment).resolve()
        self.descriptor = json.loads(self.path.read_text())
        if self.descriptor.get('format') != FORMAT:
            raise ValueError('Unsupported hierarchy experiment format')
        self.roots = dict(iabench=str(Path(iabench_root or self.descriptor['roots']['iabench']).resolve()),
                          iab=str(Path(iab_root or self.descriptor['roots']['iab']).resolve()))
        self.bench = IABenchDataset(self.roots['iabench'], processor_name=None)
        self.iab_root = Path(self.roots['iab'])
        bm = json.loads((self.path.parent / 'iabench.splits.json').read_text())
        im = json.loads((self.path.parent / 'iab.splits.json').read_text())
        current, self.records = describe_sources(self.bench, bm, self.iab_root, im)
        if protocol_digest(current) != protocol_digest(self.descriptor):
            raise ValueError('Experiment sources, membership, taxonomy or protocol differ')
        # Object-key sorting must not change family indices or seeded anchor initialization.
        self.descriptor['family_groups'] = {family: self.descriptor['family_groups'][family]
                                            for family in current['family_groups']}
        self.digest = protocol_digest(self.descriptor)

    def view(self, split, processor=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown experiment split')
        return ExperimentView(self, split, processor)

    def read_image(self, key):
        record = self.records[key]
        if record['source'] == 'iabench':
            return self.bench._read_image(int(key.split(':', 1)[1]))
        with Image.open(self.iab_root / key.split(':', 1)[1]) as image:
            image = image.convert('RGB')
        if record['generator'] == 'grok3':
            w, h = image.size
            if w <= 100 or h <= 50:
                raise ValueError(f'Grok image is too small for the benchmark crop: {key}')
            image = image.crop((0, 0, w - 100, h - 50))
        return image


class ExperimentView(Dataset):
    """Raw image source for native baseline adapters, or processed pixels for ours."""
    def __init__(self, data, split, processor):
        self.data, self.split, self.processor = data, split, processor
        self.class_names = data.descriptor['class_names' if split == 'test' else 'seen_classes']
        self.samples = [(key, data.records[key]['generator'], data.records[key]['source'])
                        for key in data.descriptor['splits'][split]]
        self.allowed = frozenset(key for key, _, _ in self.samples)
        self.train_augment = False
        self.aug_policy = 'corruption'
        self.degraded = 0

    def __len__(self):
        return len(self.samples)

    def _read_image(self, key):
        if key not in self.allowed:
            raise ValueError('Image is outside this experiment split')
        return self.data.read_image(key)

    def __getitem__(self, index):
        if self.processor is None:
            raise RuntimeError('Pixel processing requires a processor')
        if self.train_augment:
            raise ValueError('This experiment requires clean training')
        key, generator, source = self.samples[index]
        image = retry_image_read(lambda: self._read_image(key))
        image = apply_degradation(image, self.degraded)
        return dict(pixel_values=self.processor(images=image, return_tensors='pt')['pixel_values'][0],
                    generator=generator, source=source)
