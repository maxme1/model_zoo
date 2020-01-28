import torch
from torch import nn
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.projectors = nn.ModuleList([
            nn.Linear(embedding_dim, n_heads * hidden_dim, bias=False) for _ in range(3)
        ])
        self.out = nn.Linear(n_heads * hidden_dim, embedding_dim)
        self.neg_inf = float('-inf')

    def forward(self, query, key, value, positional_mask: torch.BoolTensor = None, future_mask: bool = False):
        """
        input and output shape: (batch, seq, embedding_dim)
        positional_mask: (batch, seq)
        future_mask: whether to mask out future positions
        """
        batch, seq = query.shape[:2]
        # (batch, seq, embedding_dim) -> (batch, n_heads, seq, hidden_dim)
        query, key, value = [projector(x).reshape(*x.shape[:2], self.n_heads, -1).transpose(1, 2)
                             for x, projector in zip([query, key, value], self.projectors)]

        # (batch, n_heads, seq, seq)
        weights = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.hidden_dim)

        # masking
        if positional_mask is not None:
            assert positional_mask.dtype == torch.bool
            assert positional_mask.ndim == 2
            reweight = torch.zeros_like(positional_mask).to(weights)
            reweight[~positional_mask] = self.neg_inf

            weights = weights + reweight[:, None, None, :]

        if future_mask:
            weights = weights + torch.triu(torch.full((seq, seq), self.neg_inf).to(weights), diagonal=1)

        # attention
        weights = torch.softmax(weights, -1)
        out = torch.matmul(weights, value)
        # (batch, n_heads, seq, hidden_dim) -> (batch, seq, n_heads * hidden_dim) -> (batch, seq, embedding_dim)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.out(out)


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, attention_hidden_dim: int, linear_hidden_dim: int):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, n_heads, attention_hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(2)])
        self.fc = feedforward(embedding_dim, linear_hidden_dim)

    def forward(self, x, positional_mask=None):
        norm1, norm2 = self.norms
        x = norm1(x + self.attention(x, x, x, positional_mask))
        return norm2(x + self.fc(x))


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, attention_hidden_dim: int, linear_hidden_dim: int):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, n_heads, attention_hidden_dim)
        self.attention = MultiHeadAttention(embedding_dim, n_heads, attention_hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(3)])
        self.fc = feedforward(embedding_dim, linear_hidden_dim)

    def forward(self, x, encoded, positional_mask=None):
        norm1, norm2, norm3 = self.norms
        x = norm1(x + self.self_attention(x, x, x, positional_mask, future_mask=True))
        x = norm2(x + self.attention(x, encoded, encoded, positional_mask))
        return norm3(x + self.fc(x))


def feedforward(embedding_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(embedding_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, embedding_dim),
    )


class PositionalEncoding(nn.Module):
    const = 1e4

    def forward(self, x):
        batch, seq, embedding_dim = x.shape
        pos = torch.arange(seq)[:, None].to(x)
        feat = torch.arange(embedding_dim)[None].to(x)
        grid = pos * self.const ** (- feat / embedding_dim)
        grid[:, 1::2] += np.pi / 2  # alternating cos and sin
        return x + torch.sin(grid)


class Transformer(nn.Module):
    def __init__(self, input_vocab, output_vocab=None, n_blocks=6, embedding_dim=512, middle_features=64, h=8,
                 conv_features=2048):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab, embedding_dim)
        if output_vocab is None:
            self.out_embedding = self.embedding
            output_vocab = input_vocab
        else:
            self.out_embedding = nn.Embedding(output_vocab, embedding_dim)

        self.encoder = nn.Sequential(
            self.embedding, PositionalEncoding(),
            *(EncoderBlock(h, embedding_dim, middle_features, conv_features) for _ in range(n_blocks))
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(h, embedding_dim, middle_features, conv_features) for _ in range(n_blocks)]
        )
        self.decoder_in = nn.Sequential(self.out_embedding, PositionalEncoding())
        self.decoder_out = nn.Linear(embedding_dim, output_vocab, bias=False)

    def forward(self, x, y, positional_mask: torch.BoolTensor = None):
        x = self.encoder(x)
        y = self.decoder_in(y)
        for module in self.decoder:
            y = module(y, x, positional_mask)
        return self.decoder_out(y)
