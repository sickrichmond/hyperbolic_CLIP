"""Compare class-prompt directions in frozen CLIP text space.

Encode the default templates and supplied JSON prompt sets, then report
pairwise cosine statistics and the closest pairs. The structural prompt file
is included by default. IAB_EXCLUDE_GENERATORS selects the class map.
This measures text-space separation before LoRA and the projection head.

Usage: python -m tests.probe_anchor_prompts --help
"""
import argparse
import json
import os

os.environ.setdefault("IAB_EXCLUDE_GENERATORS", "dalle3")   # 22-class map

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from comparison.training.test_hypclip import harness_class_names
from training.anchors import build_anchors

DEFAULT_CANDIDATES = [None, "data/anchor_prompts_structural.json"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip_name", default="openai/clip-vit-large-patch14")
    p.add_argument("--prompts", nargs="*", default=None,
                   help="JSON prompt files to score. The default templates are always "
                        "included as the reference.")
    p.add_argument("--full_matrix", action="store_true",
                   help="Print the whole KxK cosine matrix, not just the summary.")
    p.add_argument("--top", type=int, default=8, help="How many closest pairs to list.")
    return p.parse_args()


@torch.no_grad()
def encode(texts, clip, tokenizer, device):
    tok = tokenizer(texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=77)
    out = clip.text_model(input_ids=tok["input_ids"].to(device),
                          attention_mask=tok["attention_mask"].to(device))
    return F.normalize(clip.text_projection(out.pooler_output), dim=-1)


def score(names, texts, emb, top, full_matrix):
    K = len(names)
    cos = (emb @ emb.T).clamp(-1, 1).float().cpu()
    off = cos.clone().fill_diagonal_(-1.0)

    if full_matrix:
        head = "".join(f"{n[:7]:>9}" for n in names)
        print(f"{'':16}{head}")
        for n, row in zip(names, cos):
            print(f"  {n:14s}" + "".join(f"{v:9.4f}" for v in row))

    iu = torch.triu_indices(K, K, offset=1)
    pairs = cos[iu[0], iu[1]]
    order = pairs.argsort(descending=True)[:top]
    print(f"  max off-diag = {pairs.max():.4f}   mean = {pairs.mean():.4f}   "
          f"min = {pairs.min():.4f}")
    print(f"  closest {top} pairs:")
    for k in order:
        i, j = int(iu[0][k]), int(iu[1][k])
        print(f"    {pairs[k]:.4f}  {names[i]:14s} ↔ {names[j]:14s}")
    return float(pairs.max()), float(pairs.mean())


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = harness_class_names()
    clip = CLIPModel.from_pretrained(args.clip_name, use_safetensors=True).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_name)

    candidates = DEFAULT_CANDIDATES if args.prompts is None else [None, *args.prompts]
    summary = []
    for path in candidates:
        label = "default templates" if path is None else path
        # build_anchors reorders 'real' first; realign to the harness order so the
        # printed matrix is indexed the same way everywhere else in the pipeline.
        by_name = dict(zip(*build_anchors(names, path)))
        texts = [by_name[n] for n in names]
        print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")
        for n, t in zip(names, texts):
            print(f"  {n:14s} : \"{t}\"")
        print()
        summary.append((label, *score(names, texts, encode(texts, clip, tokenizer, device),
                                      args.top, args.full_matrix)))

    print(f"\n{'=' * 70}\n  summary (lower is better)\n{'=' * 70}")
    print(f"  {'prompt set':45s} {'max':>8} {'mean':>8}")
    for label, mx, mean in summary:
        print(f"  {label:45s} {mx:8.4f} {mean:8.4f}")


if __name__ == "__main__":
    main()
