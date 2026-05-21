"""
Verify that the model fits on GPU and runs a few training steps without
OOM or NaN. Run before the real training.
"""
import torch
from models.clip_lora import HyperbolicCLIP
from losses.real_loss import Step1Loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "No GPU detected"

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = HyperbolicCLIP(hyperbolic_dim=128, curv=1.0).to(device)
    loss_fn = Step1Loss(delta_pair=0.8, lambda_pair=0.05, curv=1.0)
    optim = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)

    for batch_size in [16, 32, 64, 128]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            pixel = torch.randn(batch_size, 3, 224, 224, device=device)
            is_real = torch.randint(0, 2, (batch_size,), dtype=torch.bool, device=device)
            # Fake paired prompt_ids: alternating pairs share a prompt_id, so
            # L_pair has at least some same-prompt (real, fake) pairs to evaluate.
            prompt_id = torch.arange(batch_size, device=device) // 2

            for step in range(3):
                x_hyp, tangent = model(pixel)
                loss, logs = loss_fn(x_hyp, tangent, is_real, prompt_id)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optim.step()
                assert torch.isfinite(loss), f"NaN/Inf at batch={batch_size}, step={step}"

            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"batch_size={batch_size}: OK, peak memory = {peak_gb:.2f} GB")

        except torch.cuda.OutOfMemoryError:
            print(f"batch_size={batch_size}: OUT OF MEMORY")
            break


if __name__ == "__main__":
    main()