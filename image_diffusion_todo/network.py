from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from module import DownBlock, MiddleBlock, UpBlock, Upsample, Downsample, AttnBlock, ResBlock, TimeEmbedding, Swish
from torch.nn import init
import random


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_channels: int = 64,
        ch_mult: List[int] = [1, 2, 2, 4],
        attn: List[bool] = [False, False, True, True],
        n_blocks: int = 2,
        use_cfg: bool = False,
        cfg_dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        n_resolutions = len(ch_mult)
        self.out_channels = out_channels
        
        self.init_conv = nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1)
        
        # Time embedding
        self.tdim = n_channels * 4
        self.time_embedding = TimeEmbedding(self.tdim)
        
        # Class embedding
        self.use_cfg = use_cfg
        if use_cfg:
            self.cfg_dropout = cfg_dropout
            assert num_classes is not None, "num_classes must be provided when use_cfg is True."
            self.class_embedding = nn.Embedding(num_classes, self.tdim)
            
        
        # Downsampling
        self.downs = nn.ModuleList()
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):       
            out_channels = in_channels * ch_mult[i]
            for _ in range(n_blocks):
                self.downs.append(DownBlock(in_channels, out_channels, self.tdim, attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                self.downs.append(Downsample(out_channels))
             
        # Middle blocks
        self.middle = MiddleBlock(out_channels, self.tdim)
        
        # Upsampling
        self.ups = nn.ModuleList()
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):  
            in_channels = out_channels
            for _ in range(n_blocks):
                self.ups.append(UpBlock(in_channels, out_channels, self.tdim, attn[i]))
                in_channels = out_channels
            out_channels = in_channels // ch_mult[i]
            self.ups.append(UpBlock(in_channels, out_channels, self.tdim, attn[i]))
            if i > 0:
                self.ups.append(Upsample(out_channels))
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, n_channels),
            Swish(),
            nn.Conv2d(n_channels, self.out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, timestep, class_label=None):
        """
        Input:
            x (`torch.Tensor [B,C,H,W]`)
            timestep (`torch.Tensor [B]`)
            class_label (`torch.Tensor [B]`, optional)
        Output:
            out (`torch.Tensor [B,C,H,W]`): noise prediction.
        """

        # Time embedding
        t = self.time_embedding(timestep)
        
        # Class embedding with CFG
        if self.use_cfg and self.class_embedding is not None and class_label is not None:
            cemb = self.class_embedding(class_label)
            no_class_mask = (class_label==0)
            dropout_mask = torch.rand_like(class_label) < self.cfg_dropout
            mask = no_class_mask | dropout_mask
            cemb[mask, :] = 0
            temb = temb + cemb
        
        x = self.init_conv(x)
        h = [x]
        
        # Down path
        for m in self.downs:
            x = m(x, t)
            h.append(x)
        
        # Middle
        x = self.middle(x, t)
        
        # Up path
        for m in self.ups:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat([x, s], dim=1)
                x = m(x, t)

        # Final blocks
        out = self.final_conv(x)
        
        return out