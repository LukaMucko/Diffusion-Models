from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample
from torch.nn import init
import random

class UNet(nn.Module):
    def __init__(
        self,
        T: int = 1000,
        image_resolution: int = 64,
        in_channels: int = 3,
        out_channels: int = 3,
        ch: int = 128,
        ch_mult: List[int] = [1, 2, 2, 2],
        attn: List[int] = [1],
        num_res_blocks: int = 4,
        dropout: float = 0.1,
        use_cfg: bool = False,
        cfg_dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.image_resolution = image_resolution
    
        # TODO: Implement an architecture according to the provided architecture diagram.
        # You can use the modules in `module.py`.

        self.image_resolution = image_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        self.num_classes = num_classes
        
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(tdim)
        
        if num_classes:
            self.class_embedding = nn.Embedding(num_classes, tdim)
            tdim = tdim * 2
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList()
        input_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(input_ch, out_ch, tdim, dropout, i in attn))
                input_ch = out_ch
            self.downs.append(DownSample(input_ch))
        
        #Upsampling
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_mult)):
            input_ch = out_ch + ch * mult
            out_ch = ch * mult
            self.ups.append(UpSample(input_ch))
            for _ in range(num_res_blocks):
                self.ups.append(ResBlock(input_ch, out_ch, tdim, dropout, i in attn))
                input_ch = out_ch

        self.final_conv = nn.Conv2d(out_ch, out_channels, kernel_size=3, padding=1)
               
    def forward(self, x, timestep, class_label=None):
        """
        Input:
            x (`torch.Tensor [B,C,H,W]`)
            timestep (`torch.Tensor [B]`)
            class_label (`torch.Tensor [B]`, optional)
        Output:
            out (`torch.Tensor [B,C,H,W]`): noise prediction.
        """
        assert (
            x.shape[-1] == x.shape[-2] == self.image_resolution
        ), f"The resolution of x ({x.shape[-2]}, {x.shape[-1]}) does not match with the image resolution ({self.image_resolution})."

        # TODO: Implement noise prediction network's forward function.

        out = self.init_conv(x)
        temb = self.time_embedding(timestep)
        
        if self.use_cfg and class_label is not None:
            drop = random.random() <= self.cfg_dropout
            if not drop:
                class_emb = self.class_embedding(class_label)
            else:
                class_emb = torch.zeros_like(temb)
            temb = torch.cat([temb, class_emb], dim=1)
        elif self.num_classes is not None:
            class_emb = torch.zeros_like(temb)
            temb = torch.cat([temb, class_emb], dim=1)
            
        intermediates = []
        for layer in self.downs:
            out = layer(out, temb)
            if isinstance(layer, DownSample):
                intermediates.append(out)
                        
        for layer in self.ups:
            if isinstance(layer, UpSample):
                out = torch.cat([intermediates.pop(), out], dim=1)
            out = layer(out, temb)
        
        out = self.final_conv(out)
        return out
