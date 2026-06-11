"""
Plot training curves from one or more train_attribution[_euclidean].py SLURM
.out logs — no extra logging needed, it just parses what the scripts already
print.

Per epoch it reads:
    Epoch N: train loss=...   L_img_cls=...  L_cap_cls=...  L_img_cap=...  L_norm=...
      val: overall=...%  balanced=...%

  - `train loss` is the total optimised loss.
  - L_img_cls / L_cap_cls / L_img_cap / L_norm are the RAW (pre-λ) hyperbolic
    cone-loss terms; the total combines them with the λ weights from training
    (e.g. L_img_cls + 1.0·L_cap_cls + 0.5·L_img_cap + 0.5·L_norm). Euclidean logs
    have no such terms, so only the total is plotted.

Produces a 2-panel figure: losses (symlog y — handles the L_norm spike that
decays to exactly 0) on top, val balanced accuracy on the bottom. With a SINGLE
log the per-term breakdown is shown; with several logs only the totals are drawn
(one line per log) to keep the comparison readable.

Usage:
    python -m scripts.plot_training_log \\
        --logs attr_all_45551934.out \\
        --labels "hyperbolic d=16" \\
        --output loss_curve.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"^Epoch\s+(\d+):\s*train loss=([0-9.]+)")
VAL_RE   = re.compile(r"^\s*val:\s*overall=([0-9.]+)%\s+balanced=([0-9.]+)%")
# Raw hyperbolic cone-loss terms (all on the same line as `train loss=`).
COMPONENT_RES = {
    name: re.compile(rf"{name}=([0-9.]+)")
    for name in ("L_img_cls", "L_cap_cls", "L_img_cap", "L_norm")
}


def parse_log(path: Path) -> dict:
    """Return {'epoch', 'loss', 'val_balanced', 'components': {name: [...]}}.
    Each val line is associated with the most recent epoch; component terms are
    read off that same epoch line when present."""
    epochs, loss, val_bal = [], [], []
    components: dict[str, list] = {}
    cur = None
    for line in path.read_text().splitlines():
        m = EPOCH_RE.match(line)
        if m:
            cur = {"epoch": int(m.group(1)), "loss": float(m.group(2)), "comp": {}}
            for name, rx in COMPONENT_RES.items():
                cm = rx.search(line)
                if cm:
                    cur["comp"][name] = float(cm.group(1))
            continue
        v = VAL_RE.match(line)
        if v and cur is not None:
            epochs.append(cur["epoch"])
            loss.append(cur["loss"])
            val_bal.append(float(v.group(2)))
            for name, val in cur["comp"].items():
                components.setdefault(name, []).append(val)
            cur = None
    return {"epoch": epochs, "loss": loss, "val_balanced": val_bal,
            "components": components}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="+", required=True, help="One or more .out files.")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Legend label per log (defaults to file stem).")
    p.add_argument("--output", default="loss_curve.png")
    args = p.parse_args()

    labels = args.labels or [Path(l).stem for l in args.logs]
    if len(labels) != len(args.logs):
        raise SystemExit("--labels must match the number of --logs")
    single = len(args.logs) == 1

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for log, label in zip(args.logs, labels):
        d = parse_log(Path(log))
        if not d["epoch"]:
            print(f"  WARNING: no epochs parsed from {log}")
            continue
        ax_loss.plot(d["epoch"], d["loss"], "o-", lw=2,
                     label=(f"{label}: total" if single else label))
        if single:
            for name, vals in d["components"].items():
                if len(vals) == len(d["epoch"]):
                    ax_loss.plot(d["epoch"], vals, ".--", lw=1, label=name)
        ax_acc.plot(d["epoch"], d["val_balanced"], "s-", label=label)
        comp_txt = "  ".join(f"{n}={v[-1]:.4f}" for n, v in d["components"].items())
        print(f"  {label}: {len(d['epoch'])} epochs, total loss {d['loss'][0]:.4f}→"
              f"{d['loss'][-1]:.4f}, best val balanced={max(d['val_balanced']):.1f}%"
              + (f"  | final terms: {comp_txt}" if comp_txt else ""))

    ax_loss.set_yscale("symlog", linthresh=1e-3)   # shows the 0-valued terms too
    ax_loss.set_ylabel("loss (symlog: terms can hit 0)")
    ax_loss.grid(True, which="both", alpha=0.3)
    ax_loss.legend(fontsize=8, ncol=2)
    ax_loss.set_title("Training curves")

    ax_acc.set_ylabel("val balanced accuracy (%)")
    ax_acc.set_xlabel("epoch")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"  saved → {args.output}")


if __name__ == "__main__":
    main()
