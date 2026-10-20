import torch
import torch.nn as nn


class DeFakeImageEncoder(nn.Module):
    """Frozen OpenAI CLIP image encoder used by De-Fake's image-only branch."""

    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "De-Fake requires OpenAI CLIP. Install it with "
                "pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        clip_model, _ = clip.load(model_name, device="cpu", jit=False)
        self.visual = clip_model.visual
        self.output_dim = self.visual.proj.shape[-1]
        del clip_model

        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, images):
        dtype = self.visual.conv1.weight.dtype
        return self.visual(images.to(dtype=dtype))