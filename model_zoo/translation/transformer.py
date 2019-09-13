import torch
from torch import nn
import numpy as np

from dpipe.torch import to_cuda, is_on_cuda


class AttentionBlock(nn.Module):
    def __init__(self, h, embedding_dim, middle_features, mask):
        super().__init__()
        self.h = h
        self.middle_features = middle_features
        self.projectors = nn.ModuleList(
            [nn.Linear(embedding_dim, h * middle_features, bias=False) for _ in range(3)])
        self.out = nn.Linear(h * middle_features, embedding_dim)
        self.mask = mask

    def forward(self, query, key, value):
        """input shape: (batch, seq, in_features)"""
        batch, seq = query.shape[:2]
        # (batch, seq, in_features) -> (batch, h, seq, out_features)
        query, key, value = [projector(x).reshape(*x.shape[:2], self.h, -1).transpose(1, 2)
                             for x, projector in zip([query, key, value], self.projectors)]

        # (batch, h, seq, seq)
        weights = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.middle_features)
        if self.mask:
            i, j = torch.arange(query.shape[2])[:, None], torch.arange(key.shape[2])[None]
            weights[..., i < j] = float('-inf')

        weights = torch.softmax(weights, -1)
        # (batch, h, seq, out_features) -> (batch, seq, h * out_features)
        out = torch.matmul(weights, value).transpose(1, 2).reshape(batch, seq, -1)
        # (batch, seq, out_features)
        return self.out(out)


class PositionalEncoding(nn.Module):
    const = 1e4

    def forward(self, x):
        batch, seq, features = x.shape
        pos = to_cuda(torch.arange(seq, dtype=torch.float32)[:, None], is_on_cuda(x))
        feat = to_cuda(torch.arange(features, dtype=torch.float32)[None], is_on_cuda(x))
        grid = pos / self.const ** (2 / features * feat)
        grid[:, ::2] += np.pi / 2  # alternating cos and sin
        return x + torch.sin(grid[None])


class LayerNorm(nn.Module):
    def __init__(self, features, axis=-1, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
        self.axis = axis

    def forward(self, x):
        return self.gamma * (x - x.mean(self.axis, keepdim=True)) / (
                x.std(self.axis, keepdim=True) + self.eps) + self.beta


def conv(embedding_dim, conv_features):
    return nn.Sequential(
        nn.Conv1d(embedding_dim, conv_features, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(conv_features, embedding_dim, kernel_size=1),
    )


class EncoderBlock(nn.Module):
    def __init__(self, h, embedding_dim, middle_features, conv_features):
        super().__init__()
        self.attention = AttentionBlock(h, embedding_dim, middle_features, False)
        self.norms = nn.ModuleList([LayerNorm(embedding_dim), LayerNorm(embedding_dim)])
        self.conv = conv(embedding_dim, conv_features)

    def forward(self, x):
        x = self.norms[0](x + self.attention(x, x, x))
        return self.norms[1](x + self.conv(x.transpose(1, 2)).transpose(1, 2))


class DecoderBlock(nn.Module):
    def __init__(self, h, embedding_dim, middle_features, conv_features):
        super().__init__()
        self.masked_attention = AttentionBlock(h, embedding_dim, middle_features, True)
        self.attention = AttentionBlock(h, embedding_dim, middle_features, False)
        self.norms = nn.ModuleList([LayerNorm(embedding_dim), LayerNorm(embedding_dim), LayerNorm(embedding_dim)])
        self.conv = conv(embedding_dim, conv_features)

    def forward(self, x, embedding):
        x = self.norms[0](x + self.masked_attention(x, x, x))
        x = self.norms[1](x + self.attention(x, embedding, embedding))
        return self.norms[2](x + self.conv(x.transpose(1, 2)).transpose(1, 2))


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

    def forward(self, x, y):
        x = self.encoder(x)
        y = self.decoder_in(y)
        for module in self.decoder:
            y = module(y, x)
        return self.decoder_out(y)
