import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from peft import LoraConfig, get_peft_model

from geometry.lorentz import exp_map0


class HyperbolicCLIP(nn.Module):
    """
    CLIP vision encoder with LoRA adapters and a projection head that maps
    image features to the Lorentz model of hyperbolic space.

    Following the HySAC/MERU convention, points on the hyperboloid are stored
    as their space components only (shape (B, D)); the time component is
    implicit and recovered when needed via the hyperboloid constraint
    x_time = sqrt(1/curv + ||x_space||^2).
    """

    def __init__(
        self,
        clip_name: str = "openai/clip-vit-base-patch32",
        hyperbolic_dim: int = 128,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        init_scale: float = 0.1,
        curv: float = 1.0,
    ):
        super().__init__()
        self.curv = curv

        # Frozen CLIP backbone
        self.clip = CLIPVisionModel.from_pretrained(clip_name, use_safetensors=True)
        for p in self.clip.parameters():
            p.requires_grad = False

        # LoRA on the ViT attention layers
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.clip = get_peft_model(self.clip, lora_config)

        # Projection head: CLIP features -> tangent space at origin
        clip_dim = self.clip.config.hidden_size  # 768 for ViT-B
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.GELU(),
            nn.Linear(clip_dim, hyperbolic_dim),
        )

        # Small init of the last layer so that initial tangent norms
        # are moderate; otherwise sinh(||v||) blows up immediately.
        with torch.no_grad():
            self.projection[-1].weight.mul_(init_scale)
            if self.projection[-1].bias is not None:
                self.projection[-1].bias.zero_()

    def forward(self, pixel_values: torch.Tensor):
        """
        Args:
            pixel_values: (B, 3, H, W) preprocessed image tensor.

        Returns:
            x_hyp:   (B, hyperbolic_dim) space components on the hyperboloid.
            tangent: (B, hyperbolic_dim) tangent vectors at origin.
                     ||tangent|| equals the hyperbolic distance from origin.
        """
        out = self.clip(pixel_values=pixel_values)
        feats = out.last_hidden_state[:, 0, :]   # CLS token
        tangent = self.projection(feats)
        x_hyp = exp_map0(tangent, curv=self.curv)
        return x_hyp, tangent

    def trainable_parameters(self):
        """Convenience accessor: LoRA + projection head parameters."""
        return [p for p in self.parameters() if p.requires_grad]