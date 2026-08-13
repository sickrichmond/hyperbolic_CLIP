"""Self-check for the Optuna pruning hook — stdlib only, runs on the login node.

The whole pruning mechanism rests on one regex matching a line the trainer prints
and NOT matching any of the lines around it. If that silently stops matching (a
tweak to the trainer's print), every trial would run to completion with `best=-1`
and the study would be garbage without failing.

    python -m tests.test_optuna_parse
"""
from scripts.optuna_search import VAL_RE

# Verbatim shape of one epoch of train_attribution.py stdout (:578-612).
TRAINER_STDOUT = """
Epoch 1: train loss=0.3121  L_img_cls=0.3105  L_norm=0.0032  lr=2.94e-04
           cone_acc=91.2%  inside_img=88.4%  ψ_anc=0.243  ξ_img→anc=0.180  ‖t̄_anc‖=4.11
  val: overall=93.7%  balanced=92.4%  (43733 samples)
    real      :  99.1%
    4o        :  95.0%
    mid-6.0   :   0.0%
  ↳ saved checkpoint (balanced val=92.4%) → /tmp/optuna_trial.pt

Epoch 2: train loss=0.1440  L_img_cls=0.1431  L_norm=0.0009  lr=1.51e-04
           cone_acc=97.8%  inside_img=96.9%  ψ_anc=0.241  ξ_img→anc=0.121  ‖t̄_anc‖=4.19
  val: overall=98.9%  balanced=98.86%  (43733 samples)
"""


def main():
    hits = [VAL_RE.match(l) for l in TRAINER_STDOUT.splitlines()]
    hits = [m for m in hits if m]
    assert len(hits) == 2, f"expected 2 val lines, matched {len(hits)}"

    balanced = [round(float(m.group(2)) / 100.0, 6) for m in hits]
    assert balanced == [0.924, 0.9886], balanced
    overall = [round(float(m.group(1)) / 100.0, 6) for m in hits]
    assert overall == [0.937, 0.989], overall

    # The per-class lines below each val line must NOT match: they also contain a
    # percentage, and a looser regex would report five "epochs" per epoch.
    for line in TRAINER_STDOUT.splitlines():
        if "val:" not in line:
            assert VAL_RE.match(line) is None, f"false positive on: {line!r}"

    print(f"OK — {len(hits)} val lines, balanced={balanced}")


if __name__ == "__main__":
    main()
