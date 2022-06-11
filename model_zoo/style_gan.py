from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional
import numpy as np

from dpipe.itertools import dmap
from dpipe.layers import ConsistentSequential
from dpipe.torch import to_var, to_np, optimizer_step


class Mapping(ConsistentSequential):
    def __init__(self, latent_dim, mapping_levels):
        super().__init__(
            lambda x, y: nn.Sequential(nn.Linear(x, y), nn.LeakyReLU(0.2)),
            [latent_dim] * (mapping_levels + 1)
        )


class NoisyLayer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.factors = nn.Parameter(torch.zeros(n, requires_grad=True), requires_grad=True)

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(len(x), 1, *x.shape[2:]).to(x)
        assert noise.shape[1] == 1 and noise.ndim == x.ndim, noise.shape
        return x + noise * self.factors[..., None, None]


class NoisyConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.add_noise = NoisyLayer(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, noise=None):
        return self.activation(self.add_noise(self.conv(x), noise))


class Mixer(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.projection = nn.Linear(latent_dim, channels * 2)
        self.eps = 1e-9

    def forward(self, x, latent):
        style = self.projection(latent).reshape(len(x), 2, x.shape[1], 1, 1)
        slope, intercept = style[:, 0], style[:, 1]
        x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + self.eps)
        return x * slope + intercept


class Start(nn.Module):
    def __init__(self, latent_dim, channels, shape):
        super().__init__()
        self.start = nn.Parameter(torch.ones(channels, *shape, requires_grad=True), requires_grad=True)
        self.add_noise = NoisyLayer(channels)
        self.conv = NoisyConv(channels, channels)
        self.mixer1, self.mixer2 = Mixer(channels, latent_dim), Mixer(channels, latent_dim)

    def forward(self, latent, noises=(None, None)):
        # TODO: torch function
        x = torch.stack([self.start] * len(latent))
        x = self.mixer1(self.add_noise(x, noises[0]), latent)
        x = self.mixer2(self.conv(x, noises[1]), latent)
        return x


class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=scale_factor),
            # TODO: blur
        )
        self.conv1, self.conv2 = NoisyConv(in_channels, out_channels), NoisyConv(out_channels, out_channels)
        self.mixer1, self.mixer2 = Mixer(out_channels, latent_dim), Mixer(out_channels, latent_dim)

    def forward(self, x, latent, noises=(None, None)):
        x = self.upsample(x)
        x = self.mixer1(self.conv1(x, noises[0]), latent)
        x = self.mixer2(self.conv2(x, noises[1]), latent)
        return x


class StyleGenerator(nn.Module):
    def __init__(self, latent_dim, levels, out_channels):
        super().__init__()
        start_channels, start_shape = 512, (4, 4)
        self.start = Start(latent_dim, start_channels, start_shape)

        synthesiser = []
        channels = start_channels
        for _ in range(levels):
            synthesiser.append(Upscale(channels, channels // 2, latent_dim, 2))
            channels //= 2

        self.mapping = Mapping(latent_dim, 8)
        self.synthesiser = nn.ModuleList(synthesiser)
        self.output = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x):
        latent = self.mapping(x)
        x = self.start(latent)
        for level in self.synthesiser:
            x = level(x, latent)
        return self.output(x)


class StyleDiscriminator(nn.Sequential):
    def __init__(self, channels, levels):
        blocks = [nn.Conv2d(channels, 16, 1)]
        channels = 16

        for _ in range(levels):
            blocks.extend((
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(channels, channels * 2, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2),
            ))
            channels *= 2

        blocks.extend((
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, 1),
        ))
        super().__init__(*blocks)


def sample_on_sphere(dim, count=None):
    r = np.random.randn(count or 1, dim)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    if count is None:
        r, = r
    return r


def train_step(image_groups, *, generator, discriminator, gen_optimizer, disc_optimizer, latent_dim, r1_weight):
    def latent(reference):
        return to_var(sample_on_sphere(latent_dim, len(reference))).to(reference)

    assert len(image_groups)
    discriminator.train()
    generator.eval()

    losses = defaultdict(list)
    for images in image_groups:
        real = to_var(images, device=discriminator)
        fake = generator(latent(real)).detach()

        real_logits = discriminator(real)
        fake_logits = discriminator(fake)

        r1 = r1_loss(images, discriminator)
        dis_fake = functional.softplus(fake_logits).mean()
        dis_real = functional.softplus(-real_logits).mean()

        loss = dis_fake + dis_real
        if r1_weight > 0:
            loss = loss + r1_weight * r1

        optimizer_step(disc_optimizer, loss)
        losses['dis_fake'].append(to_np(dis_fake))
        losses['dis_real'].append(to_np(dis_real))
        losses['dis_r1'].append(to_np(r1))

    # it's ok, `real` is already defined
    generator.train()
    discriminator.eval()
    fake = generator(latent(real))
    fake_logits = discriminator(fake)

    loss = functional.softplus(-fake_logits).mean()
    optimizer_step(gen_optimizer, loss)
    return {**dmap(np.mean, losses), 'gen': to_np(loss).item()}


def r1_loss(images, discriminator):
    images = to_var(images, device=discriminator, requires_grad=True)
    logits = discriminator(images)
    grads, = torch.autograd.grad(
        logits, images, torch.ones_like(logits),
        create_graph=True, retain_graph=True
    )
    return (grads ** 2).sum() / len(logits)
