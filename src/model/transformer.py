import torch
from torch import nn

from src.model.attention import MultiHeadAttention, pad_mask
from src.model.embedding import PositionalEmbedding, TokenEmbedding
from torch.nn.utils.rnn import pad_sequence


class FF(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(d_model * 4, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, dropout):
        super().__init__()

        self.mha = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            dropout=dropout
        )
        self.ff = FF(
            d_model=d_model,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attention = self.mha(x, mask)
        x = self.norm1(attention + x)

        ff = self.ff(x)
        x = self.norm2(ff + x)

        return x


class Transformer(nn.Module):
    def __init__(self,
            seq_len,
            d_model,
            vocab_size,
            n_head,
            num_layers,
            dropout,
            pad_index: int,
            n_features):
        super().__init__()

        self.pad_index = pad_index
        self.pos_embed = PositionalEmbedding(
            seq_len=seq_len,
            d_model=d_model
        )
        self.token_embed = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            pad_index=pad_index
        )
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                n_head=n_head,
                d_model=d_model,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, n_features)

    def forward(self, x):
        input_ids = x
        mask = pad_mask(x, pad_index=self.pad_index)
        tokens = self.token_embed(x)
        positions = self.pos_embed(tokens)
        x = tokens + positions

        for layer in self.encoder_layers:
            x = layer(x, mask)

        x = self.norm(x)
        not_pad = (input_ids != self.pad_index).unsqueeze(dim=-1)
        x = (x * not_pad).sum(dim=1) / not_pad.sum(dim=1).clamp(min=1)

        return self.linear(x)
