import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, d_model: int, d_head):
        super().__init__()

        self.d_head = d_head
        self.Qw = nn.Linear(d_model, d_head, bias=False)
        self.Kw = nn.Linear(d_model, d_head, bias=False)
        self.Vw = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor, mask):
        Q = self.Qw(x)
        K = self.Kw(x)
        V = self.Vw(x)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores: torch.Tensor = scores.masked_fill(mask, float("-inf"))
        attention = torch.softmax(
            scores,
            dim=-1) @ V

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head

        self.Ow = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.attention = nn.ModuleList([
            Attention(
                d_model=d_model,
                d_head=d_model // n_head) for _ in range(n_head)
        ])

    def forward(self, x, mask):
        attention = torch.cat([
            i(x, mask) for i in self.attention
        ], dim=-1)

        return self.Ow(attention)


def pad_mask(x: torch.Tensor, pad_index: int = 0):
    return x.eq(pad_index)[:, None, :]
