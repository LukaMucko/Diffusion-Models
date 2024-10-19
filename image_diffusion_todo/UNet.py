import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, UNet2DConditionModel

class UNet(nn.Module):
    def __init__(self, num_classes=None, dropout=0.1, device="cpu", dtype=torch.float32):
        super().__init__()
        
        self.unet = UNet2DModel(
            sample_size=32,
            in_channels=3, 
            out_channels=3, 
            block_out_channels=(64, 128, 256, 512), 
            layers_per_block=3, 
            downsample_type="resnet", 
            upsample_type="resnet", 
            class_embed_type=None if num_classes is None else "identity",
            num_class_embeds=num_classes
            ).to(device).to(dtype)
        
        tdim = 64 * 4
        self.device = device
        self.dtype = dtype
        self.conditional = False
        if num_classes is not None:
            self.conditional=True
            self.class_embedding = nn.Embedding(num_classes, tdim)
            self.dropout = dropout
            self.class_embedding.to(device).to(dtype)
            
    def forward(self, x, timestep, class_label=None):
        
        if self.conditional:
            if class_label is None:
                raise ValueError("class_label must be provided if model is conditional")
            mask = torch.rand_like(class_label.float()) > self.dropout
            mask = mask.to(self.device)
            class_embed = self.class_embedding(class_label)
            class_embed = class_embed * mask[:, None]
            class_embed = class_embed.to(self.dtype)
            out = self.unet(x, timestep, class_embed)["sample"]
        else:
            out = self.unet(x, timestep)["sample"]
            
        return out
            