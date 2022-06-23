import torch
from more_itertools import zip_equal
from torch import nn
from torch.nn import functional
import numpy as np
from dpipe.layers import ConsistentSequential


# def normal_init_layer(layer):
#     for p in layer.parameters():
#         torch.nn.init.normal_(p)
#     return layer


def init_linear(layer: nn.Linear):
    torch.nn.init.normal_(layer.weight)
    torch.nn.init.ones_(layer.bias)
    return layer


def init_conv(layer: nn.Conv2d):
    torch.nn.init.normal_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer


class Mapping(ConsistentSequential):
    def __init__(self, latent_dim, mapping_levels, style_dim: int = None):
        super().__init__(
            lambda x, y: nn.Sequential(init_linear(nn.Linear(x, y)), nn.LeakyReLU(0.2)),
            np.linspace(latent_dim, style_dim, mapping_levels + 1, dtype=int)
        )


class NoisyLayer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.factors = nn.Parameter(torch.zeros(n, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(len(x), 1, *x.shape[2:]).to(x)
        assert noise.shape[1] == 1 and noise.ndim == x.ndim, noise.shape
        return x + noise * self.factors


def norm(x):
    return (x - x.mean((2, 3), keepdim=True)) * (1 / (x.std((2, 3), keepdim=True) + 1e-9))


class Start(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.start = nn.Parameter(torch.ones(1, channels, 4, 4))
        self.add_noise = NoisyLayer(channels)

    def forward(self, batch_size, noise=None):
        return norm(self.add_noise(self.start.expand(batch_size, -1, -1, -1), noise))


class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.projection = init_linear(nn.Linear(style_dim, in_channels * 2))
        self.conv = init_conv(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.add_noise = NoisyLayer(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise=None):
        # 1. style projection
        style = self.projection(style).reshape(len(x), 2, x.shape[1], 1, 1)
        # 2. modulation
        slope, intercept = style[:, 0], style[:, 1]
        x = x * slope + intercept
        if self.scale_factor != 1:
            x = functional.interpolate(x, mode='bilinear', scale_factor=self.scale_factor)
        # 3. conv + activation
        x = self.activation(self.conv(x))
        # 4. noise
        x = self.add_noise(x, noise)
        # 5. normalize
        x = norm(x)
        return x


class StyleGenerator(nn.Module):
    def __init__(self, latent_dim: int, start_channels, stop_channels, levels: int, out_channels,
                 mapping_levels=8, style_dim: int = None):
        super().__init__()
        style_dim = style_dim or latent_dim
        synthesiser = [
            StyleBlock(start_channels, start_channels, style_dim, 1)
        ]
        for in_, out in consistent_channels(start_channels, stop_channels, levels):
            synthesiser.extend((
                StyleBlock(in_, in_, style_dim, 2),
                StyleBlock(in_, out, style_dim, 1),
            ))

        self.mapping = Mapping(latent_dim, mapping_levels, style_dim)
        self.start = Start(start_channels)
        self.synthesiser = nn.ModuleList(synthesiser)
        self.output = init_conv(nn.Conv2d(stop_channels, out_channels, 1))

    def forward(self, latent, noises=None):
        if noises is None:
            noises = [None] * (1 + len(self.synthesiser))
        assert len(noises) == 1 + len(self.synthesiser)

        style = self.mapping(latent)
        image = self.start(len(style), noise=noises[0])
        for level, noise in zip_equal(self.synthesiser, noises[1:]):
            image = level(image, style)
        return self.output(image)


class StyleDiscriminator(nn.Sequential):
    def __init__(self, in_channels, start_channels, stop_channels, levels: int):
        blocks = [nn.Conv2d(in_channels, start_channels, 1)]

        for in_, out in consistent_channels(start_channels, stop_channels, levels):
            blocks.extend((
                nn.Conv2d(in_, in_, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_, out, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ))

        blocks.extend((
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(stop_channels, 1),
        ))
        super().__init__(*blocks)


def sample_on_sphere(dim, count=None, state=None):
    if not isinstance(state, np.random.RandomState):
        state = np.random.RandomState(state)

    results = []
    for _ in range(count or 1):
        r = state.randn(dim)
        norm = np.linalg.norm(r)
        while norm < 1e-3:
            r = state.randn(dim)
            norm = np.linalg.norm(r)

        results.append(r / norm)

    if count is None:
        return results
    return np.stack(results)


def consistent_channels(start, stop, count):
    channels = np.linspace(start, stop, count + 1, dtype=int)
    return zip(channels, channels[1:])
