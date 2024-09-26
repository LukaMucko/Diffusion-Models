import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from jaxtyping import Array, Int, Float
import math

class PositionalEncoding(nn.Module):

    def __init__(self, t_channel: Int):
        """
        (Optional) Initialize positional encoding network

        Args:
            t_channel: number of modulation channel
        """
        super().__init__()
        self.t_channel = t_channel
        
        if t_channel % 2 !=0:
            raise ValueError("t_channel must be an even number")

    def forward(self, t: Float):
        """
        Return the positional encoding of

        Args:
            t: input time

        Returns:
            emb: time embedding
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        emb = torch.zeros(t.shape[0], self.t_channel, device=t.device)
        
        div_term = torch.exp(torch.arange(0, self.t_channel, 2, device=t.device) * (-math.log(10000.0) / self.t_channel))
        emb[:, 0::2] = torch.sin(t * div_term)
        emb[:, 1::2] = torch.cos(t * div_term)
        return emb

class MLP(nn.Module):

    def __init__(self,
                 in_dim: Int,
                 out_dim: Int,
                 hid_shapes: Int[Array, "h1 ... hn"]):
        '''
        (TODO) Build simple MLP

        Args:
            in_dim: input dimension
            out_dim: output dimension
            hid_shapes: array of hidden layers' dimension
        '''
        super().__init__()
        modules = [nn.Linear(in_dim, hid_shapes[0]), nn.ReLU()]
        
        for i in range(1, len(hid_shapes)):
            modules.append(nn.Linear(hid_shapes[i-1], hid_shapes[i]))
            if i != len(hid_shapes):
                modules.append(nn.ReLU())
        modules.append(nn.Linear(hid_shapes[-1], out_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, x: Array):
        return self.model(x)



class SimpleNet(nn.Module):

    def __init__(self,
                 in_dim: Int,
                 enc_shapes: Int[Array, "h1 ... hn"],
                 dec_shapes: Int[Array, "h1 ... hn"],
                 z_dim: Int,
                 t_channel: Int=64):
        super().__init__()
        '''
        (TODO) Build Score Estimation network.
        You are free to modify this function signature.
        You can design whatever architecture.

        hint: it's recommended to first encode the time and x to get
        time and x embeddings then concatenate them before feeding it
        to the decoder.

        Args:
            in_dim: dimension of input
            enc_shapes: array of dimensions of encoder
            dec_shapes: array of dimensions of decoder
            z_dim: output dimension of encoder
        '''
        self.embedder = PositionalEncoding(t_channel)
        self.encoder = MLP(in_dim + t_channel, z_dim, enc_shapes)
        self.decoder = MLP(z_dim, in_dim, enc_shapes)

    def forward(self, t: Array, x: Array):
        '''
        (TODO) Implement the forward pass. This should output
        the score s of the noisy input x.

        hint: you are free

        Args:
            t: the time that the forward diffusion has been running
            x: the noisy data after t period diffusion
        '''
        t_embedd = self.embedder(t)
        x_in = torch.cat([x, t_embedd], axis=-1)
        
        z = self.encoder(x_in)
        s = self.decoder(z)
        return s
