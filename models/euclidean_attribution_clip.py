import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model


class EuclideanAttributionCLIP(nn.Module):
    """
    EUCLIDEAN ablation of AttributionCLIP — the baseline used to isolate the
    contribution of hyperbolic geometry.

    Everything is held identical to models.attribution_clip.AttributionCLIP:
      - same backbone: CLIP (vision + text) frozen, LoRA on q_proj/v_proj of
        both encoders (same r / alpha / dropout / target modules);
      - same projection-head capacity: a shared MLP clip_dim → clip_dim → embed_dim
        applied to both modalities, with the same small init on the last layer.

    The ONLY differences are the geometry of the embedding space and the matching
    rule:
      - AttributionCLIP lifts the head's output onto the Lorentz hyperboloid
        (exp_map0) and classifies by entailment-cone exterior angle (oxy_angle);
      - this model L2-normalises the head's output onto the unit sphere and
        classifies by COSINE SIMILARITY to the (also L2-normalised) text anchors,
        with a learnable temperature (logit_scale) — i.e. exactly zero-shot CLIP's
        classification rule, made trainable through LoRA + the projection head.

    Because the backbone, parameter count, anchors, data and training budget are
    matched, any accuracy gap against AttributionCLIP is attributable to the
    hyperbolic geometry rather than to extra capacity or data.
    """

    def __init__(
        self,
        clip_name: str = "openai/clip-vit-base-patch32",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        embed_dim: int = 128,
        init_scale: float = 0.1,
        logit_scale_init: float = 2.6593,   # log(1 / 0.07), the CLIP default
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.clip = CLIPModel.from_pretrained(clip_name, use_safetensors=True)
        for p in self.clip.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.clip = get_peft_model(self.clip, lora_cfg)

        clip_dim = self.clip.base_model.model.config.projection_dim
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.GELU(),
            nn.Linear(clip_dim, embed_dim),
        )
        # Same small init as the hyperbolic head, purely to start training from
        # the same place; on the sphere it only sets the initial pre-norm scale.
        with torch.no_grad():
            self.projection[-1].weight.mul_(init_scale)
            if self.projection[-1].bias is not None:
                self.projection[-1].bias.zero_()

        # Learnable temperature, exactly as in CLIP. Clamped at use time.
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))

    # ── CLIP-space encoding (L2-normalised, in shared CLIP space) ─────────────

    def _clip_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        feats = self.clip.visual_projection(vision_out.pooler_output)
        return F.normalize(feats, dim=-1)

    def _clip_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        feats = self.clip.text_projection(text_out.pooler_output)
        return F.normalize(feats, dim=-1)

    # ── Euclidean (spherical) projection ──────────────────────────────────────

    def _project(self, clip_emb: torch.Tensor) -> torch.Tensor:
        """clip_emb: (B, D_clip) → L2-normalised (B, embed_dim) on the unit sphere."""
        return F.normalize(self.projection(clip_emb), dim=-1)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._project(self._clip_image(pixel_values))

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self._project(self._clip_text(input_ids, attention_mask))

    def forward(
        self,
        pixel_values: torch.Tensor,
        caption_ids: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        DataParallel-friendly forward (mirrors AttributionCLIP).

        Image-only mode (caption_ids is None): returns x_img (B, embed_dim).
        Otherwise returns (x_img, x_cap). Anchors are NOT processed here — they
        have shape (K, *) not (B, *) and are encoded on the primary GPU via
        encode_text().
        """
        x_img = self.encode_image(pixel_values)
        if caption_ids is None:
            return x_img
        x_cap = self.encode_text(caption_ids, caption_mask)
        return x_img, x_cap

    # ── Convenience ───────────────────────────────────────────────────────────

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        print(
            f"EuclideanAttributionCLIP: {trainable:,} / {total:,} params trainable "
            f"({100 * trainable / total:.2f}%)  embed_dim={self.embed_dim}  "
            f"logit_scale_init={self.logit_scale.item():.4f}"
        )
