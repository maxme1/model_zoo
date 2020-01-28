import torch
from torch import nn
from transformers import OpenAIGPTModel

from dpipe.torch import get_device
from .transformer import MultiHeadAttention, feedforward


class TrainablePosEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, embedding_dim)

    def forward(self, x):
        positions = torch.arange(x.shape[1], dtype=torch.long, device=get_device(x))
        return self.embedding(positions[None])


OpenAIGPTModel


class Block(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, attention_hidden_dim: int, linear_hidden_dim: int):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, n_heads, attention_hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(2)])
        self.fc = feedforward(embedding_dim, linear_hidden_dim)

    #     TODO: dropout

    def forward(self, x, positional_mask=None):
        norm1, norm2 = self.norms
        x = norm1(x + self.attention(x, x, x, positional_mask, future_mask=True))
        return norm2(x + self.fc(x))


class GPT(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, attention_hidden_dim: int,
                 linear_hidden_dim: int, n_blocks: int, max_length: int, voc_size: int, padding_idx: int = None):
        super().__init__()
        self.pos_encoding = TrainablePosEncoding(embedding_dim, max_length)
        self.embedding = nn.Embedding(voc_size, embedding_dim, padding_idx)
        self.blocks = nn.ModuleList([
            Block(embedding_dim, n_heads, attention_hidden_dim, linear_hidden_dim) for _ in range(n_blocks)
        ])
        self.out = nn.Linear(embedding_dim, voc_size, bias=False)

    def forward(self, x, positional_mask=None):
        x = self.embedding(x) + self.pos_encoding(x)
        # TODO: dropout
        for block in self.blocks:
            x = block(x, positional_mask)

        return self.out(x)
