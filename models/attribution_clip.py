import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model

from geometry.lorentz import exp_map0


class AttributionCLIP(nn.Module):
    """
    CLIP (vision + text) with LoRA on both encoders, plus a shared projection
    head to the Lorentz model of hyperbolic space.

    Image and text are encoded through CLIP+LoRA into the shared CLIP space,
    then a small MLP head produces tangent vectors at the origin which are
    lifted onto the hyperboloid by exp_map0. The same head is used for both
    modalities so that image and text-anchor embeddings live in the same
    hyperbolic space and entailment cones can be computed between them.
    """

    def __init__(
        self,
        clip_name: str = "openai/clip-vit-base-patch32",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        hyperbolic_dim: int = 128,
        curv: float = 1.0,
        init_scale: float = 0.1,
        attn_implementation: str | None = None,
    ):
        super().__init__()
        self.curv = curv
        self.hyperbolic_dim = hyperbolic_dim

        # attn_implementation="eager" is required to read attention maps
        # (output_attentions=True), which the explainability pipeline needs.
        # Training leaves it None so transformers picks the fast default (sdpa).
        clip_kwargs = {"use_safetensors": True}
        if attn_implementation is not None:
            clip_kwargs["attn_implementation"] = attn_implementation
        self.clip = CLIPModel.from_pretrained(clip_name, **clip_kwargs)
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
            nn.Linear(clip_dim, hyperbolic_dim),
        )
        # Small init so initial tangent norms stay moderate; otherwise
        # sinh(||v||) inside exp_map0 blows up immediately.
        with torch.no_grad():
            self.projection[-1].weight.mul_(init_scale)
            if self.projection[-1].bias is not None:
                self.projection[-1].bias.zero_()

    # ── CLIP-space encoding (L2-normalised, in shared CLIP space) ─────────────

    def _clip_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        feats = self.clip.visual_projection(vision_out.pooler_output)
        return F.normalize(feats, dim=-1)

    def _clip_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        feats = self.clip.text_projection(text_out.pooler_output)
        return F.normalize(feats, dim=-1)

    # ── Hyperbolic projection ────────────────────────────────────────────────

    def to_hyperbolic(self, clip_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """clip_emb: (B, D_clip). Returns (x_hyp, tangent), both (B, D_hyp).

        Forces fp32 throughout: sinh/acosh/asin in the hyperbolic ops are unstable
        under fp16 autocast and easily produce NaN.
        """
        with torch.amp.autocast("cuda", enabled=False):
            clip_emb = clip_emb.float()
            tangent = self.projection(clip_emb)
            x_hyp = exp_map0(tangent, curv=self.curv)
        return x_hyp, tangent

    def encode_image(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.to_hyperbolic(self._clip_image(pixel_values))

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.to_hyperbolic(self._clip_text(input_ids, attention_mask))

    def forward(
        self,
        pixel_values: torch.Tensor,
        caption_ids: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        DataParallel-friendly forward.

        Image-only mode (caption_ids is None): returns x_img (B, D_hyp).
        Hierarchical mode: returns (x_img, x_cap), both (B, D_hyp). Both inputs
        are sliced along dim 0 by DataParallel — same B for both.

        Anchors are NOT processed here: they have shape (K, *) not (B, *) and
        must be encoded separately on the primary GPU via encode_text().
        """
        x_img, _ = self.encode_image(pixel_values)
        if caption_ids is None:
            return x_img
        x_cap, _ = self.encode_text(caption_ids, caption_mask)
        return x_img, x_cap

    # ── Convenience ───────────────────────────────────────────────────────────

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        print(
            f"AttributionCLIP: {trainable:,} / {total:,} params trainable "
            f"({100 * trainable / total:.2f}%)  "
            f"hyperbolic_dim={self.hyperbolic_dim}, curv={self.curv}"
        )
