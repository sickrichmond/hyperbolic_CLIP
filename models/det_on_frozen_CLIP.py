import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class DetectorDF(nn.Module):
    """
    A simple decoder to place on top of a frozen clip model to use as a baseline
    to compare performances against making classificaiton on features from
    fine-tuned CLIP
    """
    def __init__(
            self,
            clip_name: str="openai/clip-vit-large-patch14",
            num_classes: int =22,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name, use_safetensors=True)
        for p in self.clip.parameters():
            p.requires_grad=False
        clip_dim = self.clip.config.projection_dim
        self.classifier = nn.Linear(clip_dim, num_classes)

    def _clip_image(self, pixel_values:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vision_out = self.clip.vision_model(pixel_values=pixel_values)
            feats = self.clip.visual_projection(vision_out.pooler_output)
            return F.normalize(feats, dim=-1)

    def forward(
            self,
            pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        x_img = self._clip_image(pixel_values)
        return self.classifier(x_img) 


