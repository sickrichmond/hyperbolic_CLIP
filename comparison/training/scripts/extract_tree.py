"""The generator taxonomy the DATA shows, next to the one the metadata asserts.

`_HIFI_HIERARCHY` groups the 22 generators by provenance (`commercial` lumps 8
models together because they are products, not because they share an architecture).
Before we make the cone loss enforce that tree, it is worth asking what tree the
model's own errors and anchors imply — and whether the two agree.

Three independent sources, none of which costs a job:

  --confmat   the 22x22 conf_matrix in every test_results_degraded_*.txt, read by
              nothing today. Distance = how often the model swaps the two classes.
              One source per --level.
  --angles    the anchor angle matrix dumped by `probe_anchor_spread --dump`.
              Weak by construction: all 22 anchors fit inside 9 degrees.
  --centroids the per-class MEAN feature from a scripts.extract_clip_features cache.
              The most direct: it owes nothing to what the model gets wrong.

Average-linkage agglomerative clustering on each, cut at --cut, then Adjusted Rand
Index against the HiFi-Net partition chosen by --against, AND — the part that
decides — between every pair of sources. A tree only exists if the sources agree
with EACH OTHER; if they do not, there is nothing stable to enforce and Phase B
should carry only the asserted taxonomy.

--out writes the FIRST source as {class: family} for `--hierarchy emergent`, each
family named after its most central member so it still has a usable text prompt.

    python -m comparison.training.scripts.extract_tree --selfcheck
    IAB_EXCLUDE_GENERATORS=dalle3 python -m comparison.training.scripts.extract_tree \\
        --confmat $WORK/outputs/hypclip_fair_22cls --level 1 2 5 \\
        --centroids $WORK/hyp_fine_tuning/clip_features_lora --cut 6

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


def centroid_distance(cache_dir, names):
    """Angular distance between per-class MEAN features.

    The third and most direct source: it asks "do these generators produce images CLIP
    places near each other", with no reference to what the model gets wrong. Reordered
    into the active label order via the cache's own `classes` field — extract_clip_features
    builds y from `enumerate(args.generators)`, which is exactly that list.
    """
    import torch
    blob = torch.load(Path(cache_dir) / "clip_features_train.pt", weights_only=False)
    X, y = blob["X"].float(), blob["y"].long()
    order = list(blob["classes"])
    c = torch.zeros(len(order), X.shape[1]).index_add_(0, y, X)
    c = c / torch.bincount(y, minlength=len(order)).clamp(min=1).unsqueeze(1)
    c = c[[order.index(n) for n in names]]
    c = c / c.norm(dim=-1, keepdim=True)
    return (c @ c.T).clamp(-1, 1).arccos().rad2deg().tolist()


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
    print(f"\n  ARI vs HiFi-Net: {adjusted_rand(labels, hifi_labels):.3f}")
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
    p.add_argument("--level", type=int, nargs="+", default=[0],
                   help="which degradation levels to cluster; one source each. 0 (clean) is "
                        "usually USELESS: a 99.3%% model leaves ~300 errors over 462 off-diagonal "
                        "cells, so nearly every distance is exactly 1 and the linkage chains into "
                        "one blob. 2 (DS0.25) and 5 (Blur3) have real error structure; the JPEG "
                        "levels do not, their errors are the drift toward `real`.")
    p.add_argument("--angles", metavar="JSON",
                   help="anchor angle matrix from `probe_anchor_spread --dump`")
    p.add_argument("--centroids", metavar="DIR",
                   help="feature cache from scripts.extract_clip_features — clusters the "
                        "per-class MEAN feature, which owes nothing to the model's errors")
    p.add_argument("--cut", type=int, default=6,
                   help="number of families (6 = HiFi-Net level 3)")
    p.add_argument("--against", type=int, choices=[1, 2, 3], default=3,
                   help="which HiFi-Net level to score against: 1 = generated/real (2 groups), "
                        "2 = commercial/open-source/real (3), 3 = the six families. The data may "
                        "support a coarse split and not a fine one, and that decides how many "
                        "levels the cone hierarchy should have. Pair with a matching --cut.")
    p.add_argument("--out", metavar="JSON",
                   help="write {class: family} for --hierarchy emergent, from the FIRST source")
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()

    if args.selfcheck:
        return selfcheck()
    if not (args.confmat or args.angles or args.centroids):
        sys.exit("need at least one of --confmat / --angles / --centroids")

    from comparison.dataset.ImageAttributionDataset.dataset import (
        _HIFI_HIERARCHY, model_class_to_label)
    idx_to_name = {v: k for k, v in model_class_to_label.items()}
    names = [idx_to_name[i] for i in range(len(idx_to_name))]
    hifi_labels = [_HIFI_HIERARCHY[n][args.against - 1] for n in names]
    print(f"{len(names)} classes, scoring against HiFi-Net level {args.against}:")
    for cid in sorted(set(hifi_labels)):
        members = [n for n, c in zip(names, hifi_labels) if c == cid]
        fam = L3_NAMES[cid] if args.against == 3 else f"group {cid}"
        print(f"  {fam:16s} <- " + ", ".join(sorted(members)))

    sources = []          # (short name, tree, labels)
    if args.confmat:
        for level in args.level:
            src = Path(args.confmat) / f"test_results_degraded_{level}.txt"
            conf = parse_conf(src)
            if conf is None:
                sys.exit(f"{src}: no conf_matrix in it")
            if len(conf) != len(names):
                sys.exit(f"conf_matrix is {len(conf)}x{len(conf)} but the active label map has "
                         f"{len(names)} classes — check IAB_EXCLUDE_GENERATORS")
            k = len(conf)
            cells = [conf[i][j] for i in range(k) for j in range(k) if i != j]
            filled = sum(1 for c in cells if c)
            print(f"\nlevel {level}: {sum(cells)} errors in {filled}/{len(cells)} off-diagonal "
                  f"cells ({filled / len(cells):.0%} filled)")
            if filled < len(cells) // 4:
                print("  \u26a0\ufe0f  too sparse to carry a taxonomy — nearly every distance is "
                      "exactly 1 and\n     the linkage will chain. Read the clusters below as "
                      "an artifact, not a tree.")
            sources.append((f"conf L{level}",
                            *report(f"confusion matrix (level {level})",
                                    confusion_distance(conf), names, hifi_labels, args.cut)))

    if args.angles:
        payload = json.loads(Path(args.angles).read_text())
        if payload["class_names"] != names:
            sys.exit(f"dump was made with a different label map: {payload['class_names']}")
        sources.append(("anchors", *report("anchor angles", payload["angles_deg"],
                                           names, hifi_labels, args.cut)))

    if args.centroids:
        sources.append(("centroids", *report("class centroids in feature space",
                                             centroid_distance(args.centroids, names),
                                             names, hifi_labels, args.cut)))

    # The decisive table: a tree only exists if the sources agree with EACH OTHER.
    # Disagreement among them means there is nothing stable to enforce, whatever any
    # single ARI against HiFi-Net happens to say.
    labelling = [("HiFi-Net", hifi_labels)] + [(n, l) for n, _, l in sources]
    if len(labelling) > 2:
        width = max(len(n) for n, _ in labelling)
        print("\n### ARI between every pair of sources\n")
        print(" " * (width + 2) + "".join(f"{n:>12s}" for n, _ in labelling))
        for a, la in labelling:
            row = "".join(f"{adjusted_rand(la, lb):12.3f}" for _, lb in labelling)
            print(f"  {a:{width}s}{row}")
        print("\n1.0 = identical partitions, 0 = chance. If the off-diagonal is ~0 the sources\n"
              "disagree with each other as much as with the taxonomy: there is no emergent tree\n"
              "to enforce, and Phase B should run with --hierarchy hifi only.")

    if args.out and sources:
        Path(args.out).write_text(json.dumps(sources[0][1], indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}  (source: {sources[0][0]})")


if __name__ == "__main__":
    main()
