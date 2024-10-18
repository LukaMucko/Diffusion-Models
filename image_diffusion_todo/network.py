from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample
from torch.nn import init

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    def update(self, new_model):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.decay * self.shadow[name].data + (1.0 - self.decay) * param.data
            
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].data

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].data        
            

class UNet(nn.Module):
    def __init__(
        self,
        T: int = 1000,
        image_resolution: int = 64,
        in_channels = 3,
        out_channels = 3,
        ch: int = 128,
        ch_mult: List[int] = [1, 2, 2, 2],
        attn: List[bool] = [False, False, True, True],
        num_res_blocks: int = 4,
        dropout: float = 0.1,
        use_cfg: bool = False,
        cfg_dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.image_resolution = image_resolution
        self.T = T
        self.num_classes = num_classes
        self.ch = ch
        self.ch_mult = ch_mult
        self.attn = attn
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        
        tdim = 4 * ch
        self.time_embedding = TimeEmbedding(tdim, tdim)
        
        self.input = nn.Conv2d(in_channels, ch, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.input.weight, gain=1e-5)
        
        if self.use_cfg:
            self.class_embedding = nn.Embedding(num_classes, tdim)
        
        self.down = nn.ModuleList()
        in_ch = ch
        out_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = in_ch * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(in_ch, out_ch, tdim=tdim, dropout=dropout, attn=attn[i]))
                in_ch = out_ch
            if i != len(ch_mult) - 1:
                self.down.append(DownSample(in_ch))
            
        self.middle = ResBlock(out_ch, out_ch, tdim, dropout, attn=True)          
        
        self.up = nn.ModuleList()

        for i in reversed(range(len(ch_mult))):
            out_ch = in_ch
            for _ in range(num_res_blocks):
                self.up.append(ResBlock(in_ch + out_ch, out_ch, tdim, dropout, attn=attn[i]))
            out_ch = out_ch // ch_mult[i]
            self.up.append(ResBlock(in_ch + out_ch, out_ch, tdim, dropout, attn=attn[i]))
            in_ch = out_ch
            if i > 0:
                self.up.append(UpSample(in_ch, in_ch))
                
        self.final = ResBlock(ch, ch, tdim, dropout, attn=False)
        self.output = nn.Conv2d(ch, out_channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.output.weight, gain=1e-5)

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
        
        x = self.input(x)
        t = self.time_embedding(timestep)
        if self.use_cfg:
            dropout = torch.bernoulli(torch.full_like(class_label.float(), self.cfg_dropout)).bool()
            mask = (class_label != 0) | dropout
            c_emb = torch.zeros_like(t)
            c_emb[mask] = self.class_embedding(class_label[mask] - 1)
            t += c_emb 
            
        h = [x]
        for layer in self.down:
            x = layer(x, t)
            h.append(x)
        
        x = self.middle(x, t)
        
        for layer in self.up:
            if isinstance(layer, UpSample):
                x = layer(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = layer(x, t)
        return self.output(self.final(x, t))