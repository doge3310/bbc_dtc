import torch
from torch import nn, rand, arange, zeros, exp, sin, cos
from math import log


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512):
        super().__init__()

        self.weights = nn.Parameter(
            rand(vocab_size, d_model)
        )

    def forward(self, index):
        return self.weights[index]


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 512, base: int = 10000):
        super().__init__()

        positions = zeros(seq_len, d_model)
        indexes = arange(0, seq_len).unsqueeze(1)

        div = exp(
            arange(0, d_model, 2, dtype=torch.float)
            * (-log(torch.tensor(base, dtype=torch.float)) / d_model)
        )

        indexes = indexes * div
        positions[:, 0::2] = sin(indexes)
        positions[:, 1::2] = cos(indexes)

        self.register_buffer("positions", positions)

    def forward(self, x):
        index = x.size(1)
        return self.positions[:index]
