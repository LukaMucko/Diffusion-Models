import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tdim, attn=False):
        super().__init__()
        self.res = ResBlock(in_channels + out_channels, out_channels, tdim)
        if attn:
            self.attn = AttnBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tdim, attn=False):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, tdim)
        if attn:
            self.attn = AttnBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, n_channels, tdim):
        super().__init__()
        self.res1 = ResBlock(n_channels, n_channels, tdim)
        self.attn = AttnBlock(n_channels)
        self.res2 = ResBlock(n_channels, n_channels, tdim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x

        
class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        return self.conv(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels, n_heads = 1, d_k = None):
        super().__init__()
        if d_k is None:
            d_k = in_channels

        self.group_norm = nn.GroupNorm(32, in_channels)
        self.projection = nn.Linear(in_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, in_channels)
        self.scale = 1 / (d_k ** 0.5)
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x, t=None):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        
        qkv = self.projection(x).view(B, -1, self.n_heads, self.d_k * 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=2)

        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(B, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res += x
        res = res.permute(0, 2, 1).view(B, C, H, W)

        return res


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_ch, tdim, dropout=0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_ch, kernel_size=(3, 3), padding=(1, 1))
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), padding=(1, 1))
        )

        if in_channels != out_ch:
            self.shortcut = nn.Conv2d(in_channels, out_ch, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Sequential(
            Swish(), 
            nn.Linear(tdim, out_ch)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.block1(x)
        h += self.time_emb(t)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(n_channels // 4, n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb

