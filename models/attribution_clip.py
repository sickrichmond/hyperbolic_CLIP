import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model


class AttributionCLIP(nn.Module):
    """
    Full CLIP model (vision + text encoders) with LoRA adapters on both.

    Returns L2-normalised embeddings in the shared CLIP space.
    No hyperbolic projection — attribution is done via cosine similarity
    to static text anchors (Stage 1).  Hyperbolic / entailment-cone
    projection is added in Stage 2 on top of this model.
    """

    def __init__(
        self,
        clip_name: str = "openai/clip-vit-base-patch32",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name, use_safetensors=True)

        # Freeze everything; LoRA will re-enable gradients for its own params.
        for p in self.clip.parameters():
            p.requires_grad = False

        # Apply LoRA to q_proj and v_proj in both vision and text transformers.
        # get_peft_model finds all matching layer names across the full model.
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.clip = get_peft_model(self.clip, lora_cfg)

        # Unfreeze logit_scale so temperature adapts during training.
        self.clip.base_model.model.logit_scale.requires_grad_(True)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised image embeddings, shape (B, D)."""
        feats = self.clip.get_image_features(pixel_values=pixel_values)
        return F.normalize(feats, dim=-1)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Returns L2-normalised text embeddings, shape (B, D)."""
        feats = self.clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return F.normalize(feats, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (image_embeds, text_embeds), both L2-normalised."""
        return self.encode_image(pixel_values), self.encode_text(input_ids, attention_mask)

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.clip.base_model.model.logit_scale.exp().clamp(max=100)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        print(
            f"AttributionCLIP: {trainable:,} / {total:,} params trainable "
            f"({100 * trainable / total:.2f}%)"
        )
