"""Freeze the newest IAB baseline checkpoints from the 22-class Slurm log tree.

Run from the repository on Leonardo:
python -m comparison.training.scripts.styled_checkpoints --output /path/to/map.json
"""
import argparse
import json
from pathlib import Path

from comparison.training.test_baselines_styled import METHODS


def discover_checkpoints(log_root):
    mapping = {}
    missing = []
    for method in METHODS:
        candidates = [path for path in (Path(log_root) / method).glob('*/ckpt_best.pth')
                      if path.is_file() and not (
                          method == 'dna' and 'pretrain' in path.parent.name.casefold())]
        if not candidates:
            missing.append(method)
            continue
        newest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        mapping[method] = str(newest.resolve())
    if missing:
        raise FileNotFoundError(f'No ckpt_best.pth for {missing} under {log_root}; '
                                'DNA needs a stage-two run, not pretraining')
    return mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log_root', default='comparison/training/logs/default_split')
    parser.add_argument('--output', required=True, help='new JSON file; existing maps are preserved')
    args = parser.parse_args(argv)
    mapping = discover_checkpoints(args.log_root)
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(mapping, indent=2) + '\n')
    for method, path in mapping.items():
        print(f'{method:10s} {path}')
    print(f'Checkpoint map saved to {args.output}')


if __name__ == '__main__':
    main()
