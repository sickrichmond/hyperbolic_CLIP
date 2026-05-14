import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from models.clip_lora import HyperbolicCLIP
from losses.real_loss import Step1LossV2
from data.dataset import OpenFakePairedDataset, make_balanced_sampler


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data ---
    dataset = OpenFakePairedDataset(root="/mnt/data3/rtrebiani/openfake_simple")
    sampler = make_balanced_sampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Batch size: 64, dataset size: {len(dataset)}")

    # --- Model ---
    model = HyperbolicCLIP(hyperbolic_dim=128, curv=1.0).to(device)

    # --- Loss ---
    loss_fn = Step1LossV2(
        r_max=1.0,
        r_min=0.1,
        margin=0.5,
        lambda_push=0.1,
        spread_margin=0.2,
        lambda_spread=0.05,
        contrast_margin=1.0,
        lambda_contrast=0.5,
        curv=1.0,
    )

    # --- Optimizer ---
    optim = AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=0.01)

    # --- Training loop ---
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            pixel = batch["pixel_values"].to(device)
            is_real = batch["is_real"].to(device)

            x_hyp, tangent = model(pixel)
            loss, logs = loss_fn(x_hyp, tangent, is_real)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            optim.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "L_real": f"{logs['loss_real'].item():.3f}",
                "L_push": f"{logs['loss_push'].item():.3f}",
                "L_sprd": f"{logs['loss_spread'].item():.3f}",
                "L_ctr": f"{logs['loss_contrast'].item():.3f}",
                "d_real": f"{logs['mean_dist_real'].item():.2f}",
                "d_fake": f"{logs['mean_dist_fake'].item():.2f}",
            })

    torch.save({
        "lora_state": model.clip.state_dict(),
        "projection_state": model.projection.state_dict(),
        "curv": model.curv,
    }, "step1_checkpoint.pt")


if __name__ == "__main__":
    main()