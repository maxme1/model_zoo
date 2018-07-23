import torch
from torch import nn


def initial_hidden(n_layers, batch_size, hidden_size, bidirectional, cuda=True):
    def params():
        p = torch.zeros(n_layers, batch_size, hidden_size)
        if cuda:
            p = p.cuda()
        return p

    if bidirectional:
        n_layers = 2 * n_layers

    return params(), params()


def is_on_cuda(module: torch.nn.Module):
    return next(module.parameters()).is_cuda


class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers):
        super().__init__()
        self.hidden_size, self.n_layers, self.vocab_size = hidden_size, n_layers, vocab_size

        self.lstm = nn.LSTM(vocab_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.last_lstm = nn.LSTM(hidden_size * 2, vocab_size, 1, batch_first=True)

    def forward(self, x):
        # input, output shape: (batch_size, seq_length, vocab_size)
        batch_size = x.shape[0]
        x, _ = self.lstm(x, initial_hidden(self.n_layers, batch_size, self.hidden_size, True, is_on_cuda(self)))
        x, _ = self.last_lstm(x, initial_hidden(1, batch_size, self.vocab_size, False, is_on_cuda(self)))
        return x
