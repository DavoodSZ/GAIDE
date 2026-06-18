#!/usr/bin/env python3

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        position = x
        div_term = (
            torch.exp(
                torch.arange(0, self.embed_dim, 2, device=x.device)
                * (-math.log(10000.0) / self.embed_dim)
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(x.shape[0], x.shape[1], 1)
        )

        pe = torch.zeros((x.shape[0], x.shape[1], self.embed_dim), device=x.device)
        angles = position.unsqueeze(-1) * div_term
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles[:, :, : pe[:, :, 1::2].shape[-1]])
        return pe
