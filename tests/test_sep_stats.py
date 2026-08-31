"""The anchor-separation stats are now emitted unconditionally by the cone loss.

Two cases, because the whole point is that the Poincare snapshots cannot tell them
apart: random 128-d anchors (mean angle ~90 deg, healthy) and collapsed anchors both
draw as one overlapping blob in any 2-D projection.

    python -m tests.test_sep_stats
"""
import torch
import torch.nn.functional as F

from losses.attribution_loss import AttributionConeLoss

torch.manual_seed(0)
K, D, B = 22, 128, 64
loss = AttributionConeLoss(min_radius=0.5)
labels = torch.arange(B) % K
x_img = F.normalize(torch.randn(B, D), dim=-1) * 10.0

spread = F.normalize(torch.randn(K, D), dim=-1) * 3.63
_, st = loss(x_img, spread, labels)
print("spread :", {k: round(float(v), 2) for k, v in st.items() if k.startswith("sep")})
assert 85 < st["sep_mean_deg"] < 95, st["sep_mean_deg"]

collapsed = F.normalize(spread[:1].repeat(K, 1) + 1e-3 * torch.randn(K, D), dim=-1) * 3.63
_, st = loss(x_img, collapsed, labels)
print("collapsed:", {k: round(float(v), 2) for k, v in st.items() if k.startswith("sep")})
assert st["sep_mean_deg"] < 5, st["sep_mean_deg"]
assert st["sep_overlap"] == 1.0, st["sep_overlap"]
print("OK")
