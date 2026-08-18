"""The generator taxonomy the DATA shows, next to the one the metadata asserts.

`_HIFI_HIERARCHY` groups the 22 generators by provenance (`commercial` lumps 8
models together because they are products, not because they share an architecture).
Before we make the cone loss enforce that tree, it is worth asking what tree the
model's own errors and anchors imply — and whether the two agree.

Two independent sources, neither of which costs a job:

  --confmat   the 22x22 conf_matrix in every test_results_degraded_*.txt, read by
              nothing today. Distance = how often the model swaps the two classes.
  --angles    the anchor-cosine matrix dumped by `probe_anchor_spread --dump`.
              Distance = the angle between the two class directions, in degrees.

Average-linkage agglomerative clustering on either, cut at --cut clusters, then
Adjusted Rand Index against the HiFi-Net level-3 partition. Agreement means the
semantic taxonomy is confirmed by pixels; disagreement is the more interesting
result and says which tree Phase B should enforce.

The chosen partition is written with --out as {class: family} for `--hierarchy
emergent`, with each family named after its most central member so it still has a
usable text prompt.

    python -m comparison.training.scripts.extract_tree --selfcheck
    IAB_EXCLUDE_GENERATORS=dalle3 python -m comparison.training.scripts.extract_tree \\
        --confmat $WORK/outputs/hypclip_fair_22cls --cut 6 --out data/tree_emergent.json

Pure stdlib, login node. Run from the repo root.
"""
import argparse
import json
import sys
from pathlib import Path

from comparison.training.scripts.detection_from_confmat import parse_conf

# level3 of _HIFI_HIERARCHY, decoded (dataset.py:72-74).
L3_NAMES = ["commercial", "SD", "diffusers", "DiT", "AR", "real"]


def confusion_distance(conf):
    """d[i][j] = 1 - symmetric confusion rate. Identical classes -> 0, never swapped -> 1.

    Each direction is normalised by its own row before averaging, otherwise a class
    with more test images would look more confusable than it is.
    """
    k = len(conf)
    rows = [sum(r) or 1 for r in conf]
    return [[0.0 if i == j else
             1.0 - (conf[i][j] / rows[i] + conf[j][i] / rows[j]) / 2
             for j in range(k)] for i in range(k)]


def average_linkage(dist, cut):
    """Agglomerative clustering, average linkage. Returns a list of index lists.

    n=22, so the O(n^3) loop is free and a library would be a dependency for nothing.
    """
    clusters = [[i] for i in range(len(dist))]
    while len(clusters) > cut:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = sum(dist[i][j] for i in clusters[a] for j in clusters[b])
                d /= len(clusters[a]) * len(clusters[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters.pop(b)
    return clusters


def adjusted_rand(x, y):
    """ARI between two labellings given as lists of cluster ids. 1.0 = identical."""
    c2 = lambda n: n * (n - 1) / 2
    pairs = {}
    for a, b in zip(x, y):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    counts_x, counts_y = {}, {}
    for a in x:
        counts_x[a] = counts_x.get(a, 0) + 1
    for b in y:
        counts_y[b] = counts_y.get(b, 0) + 1

    index = sum(c2(n) for n in pairs.values())
    ex = sum(c2(n) for n in counts_x.values())
    ey = sum(c2(n) for n in counts_y.values())
    expected = ex * ey / c2(len(x))
    denom = (ex + ey) / 2 - expected
    return (index - expected) / denom if denom else 1.0


def centre_of(cluster, dist):
    """The member with the smallest mean distance to the rest — the family's name."""
    return min(cluster, key=lambda i: sum(dist[i][j] for j in cluster))


def report(title, dist, names, hifi_labels, cut):
    clusters = average_linkage(dist, cut)
    labels = [0] * len(names)
    for cid, members in enumerate(clusters):
        for i in members:
            labels[i] = cid

    print(f"\n### {title} — {cut} clusters\n")
    tree = {}
    for members in clusters:
        family = names[centre_of(members, dist)]
        print(f"  {family:16s} <- " + ", ".join(sorted(names[i] for i in members)))
        for i in members:
            tree[names[i]] = family
    print(f"\n  ARI vs HiFi-Net level 3: {adjusted_rand(labels, hifi_labels):.3f}")
    return tree, labels


def selfcheck():
    """Three synthetic blocks that are never confused across -> must be recovered."""
    names = [f"c{i}" for i in range(6)]
    conf = [[100 if i == j else (20 if i // 2 == j // 2 else 0)
             for j in range(6)] for i in range(6)]
    dist = confusion_distance(conf)
    clusters = average_linkage(dist, 3)
    got = sorted(sorted(c) for c in clusters)
    assert got == [[0, 1], [2, 3], [4, 5]], got
    labels = [0, 0, 1, 1, 2, 2]
    assert adjusted_rand(labels, labels) == 1.0
    assert adjusted_rand(labels, [0, 1, 2, 0, 1, 2]) < 0.1
    assert centre_of([0, 1], dist) in (0, 1)
    print(f"selfcheck ok ({names})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--confmat", metavar="DIR",
                   help="directory holding test_results_degraded_<level>.txt")
    p.add_argument("--level", type=int, default=0,
                   help="which degradation level's confusion to cluster. 0 (clean) is "
                        "usually USELESS here: a 99.3%% model leaves ~300 errors over 462 "
                        "off-diagonal cells, so the distances are ~1 everywhere and the "
                        "linkage chains into one blob. Try 2 (DS0.25) or 5 (Blur3), where "
                        "the model errs enough for the error structure to carry a taxonomy.")
    p.add_argument("--angles", metavar="JSON",
                   help="anchor cosine matrix from `probe_anchor_spread --dump`")
    p.add_argument("--cut", type=int, default=6,
                   help="number of families (6 = HiFi-Net level 3)")
    p.add_argument("--out", metavar="JSON", help="write {class: family} for --hierarchy emergent")
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()

    if args.selfcheck:
        return selfcheck()
    if not (args.confmat or args.angles):
        sys.exit("need --confmat and/or --angles")

    from comparison.dataset.ImageAttributionDataset.dataset import (
        _HIFI_HIERARCHY, model_class_to_label)
    idx_to_name = {v: k for k, v in model_class_to_label.items()}
    names = [idx_to_name[i] for i in range(len(idx_to_name))]
    hifi_labels = [_HIFI_HIERARCHY[n][2] for n in names]
    print(f"{len(names)} classes. HiFi-Net level 3 for reference:")
    for cid, fam in enumerate(L3_NAMES):
        members = [n for n in names if _HIFI_HIERARCHY[n][2] == cid]
        if members:
            print(f"  {fam:16s} <- " + ", ".join(sorted(members)))

    chosen = None
    if args.confmat:
        src = Path(args.confmat) / f"test_results_degraded_{args.level}.txt"
        conf = parse_conf(src)
        if conf is None:
            sys.exit(f"{src}: no conf_matrix in it")
        if len(conf) != len(names):
            sys.exit(f"conf_matrix is {len(conf)}x{len(conf)} but the active label map has "
                     f"{len(names)} classes — check IAB_EXCLUDE_GENERATORS")
        k = len(conf)
        cells = [conf[i][j] for i in range(k) for j in range(k) if i != j]
        filled = sum(1 for c in cells if c)
        print(f"\nlevel {args.level}: {sum(cells)} errors in {filled}/{len(cells)} off-diagonal "
              f"cells ({filled / len(cells):.0%} filled)")
        if filled < len(cells) // 4:
            print("  ⚠️  too sparse to carry a taxonomy — most distances are exactly 1 and the\n"
                  "     linkage will chain. Re-run with --level 2 or --level 5.")
        chosen, conf_labels = report(f"confusion matrix (level {args.level})",
                                     confusion_distance(conf), names, hifi_labels, args.cut)

    if args.angles:
        payload = json.loads(Path(args.angles).read_text())
        order = payload["class_names"]
        if order != names:
            sys.exit(f"dump was made with a different label map: {order}")
        dist = payload["angles_deg"]
        angle_tree, angle_labels = report("anchor angles", dist, names, hifi_labels, args.cut)
        if args.confmat:
            print(f"\n  ARI confusion vs anchors: {adjusted_rand(conf_labels, angle_labels):.3f}"
                  "  — the two sources must agree before either tree is trusted.")
        chosen = chosen or angle_tree

    if args.out:
        Path(args.out).write_text(json.dumps(chosen, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
