from pathlib import Path
from collections import defaultdict

from dpipe.batch_iter import Infinite, sample
import torch
from dpipe.im import bytescale
from dpipe.train import Checkpoints, TBLogger, train as train_base, TimeProfiler, TQDM
from torch.nn import functional
import numpy as np
from dpipe.itertools import dmap
from dpipe.torch import to_var, to_np, optimizer_step, save_model_state

from .model import sample_on_sphere
from ..utils import stitch


def train(root, dataset, generator, discriminator, latent_dim, disc_iters, batch_size, batches_per_epoch, n_epochs,
          r1_weight, lr_discriminator, lr_generator, device='cuda'):
    root = Path(root)
    generator, discriminator = generator.to(device), discriminator.to(device)

    gen_opt = torch.optim.Adam(generator.parameters(), lr_generator)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr_discriminator)

    batch_iter = Infinite(
        sample(dataset.ids),
        dataset.image,
        combiner=lambda images: [
            tuple(np.array(images).reshape(disc_iters, len(images) // disc_iters, 1, *images[0].shape))
        ],

        buffer_size=10,
        batch_size=batch_size * disc_iters,
        batches_per_epoch=batches_per_epoch,
    )

    def val():
        generator.eval()
        t = sample_on_sphere(latent_dim, 144, state=0).astype('float32')
        with torch.no_grad():
            y = to_np(generator(to_var(t, device=generator)))[:, 0]

        return {'generated__image': bytescale(stitch(y, bytescale))}

    logger = TBLogger(root / 'logs')
    train_base(
        train_step, batch_iter, n_epochs, logger=logger, validate=val,
        checkpoints=Checkpoints(root / 'checkpoints', [generator, discriminator, gen_opt, disc_opt]),
        # lr=lr,

        generator=generator, discriminator=discriminator, gen_optimizer=gen_opt, disc_optimizer=disc_opt,
        latent_dim=latent_dim, r1_weight=r1_weight,

        time=TimeProfiler(logger.logger), tqdm=TQDM(False),
    )
    save_model_state(generator, root / 'generator.pth')
    save_model_state(discriminator, root / 'discriminator.pth')


def train_step(image_groups, *, generator, discriminator, gen_optimizer, disc_optimizer, latent_dim, r1_weight,
               **optimizer_kwargs):
    def latent(reference):
        return to_var(sample_on_sphere(latent_dim, len(reference))).to(reference)

    assert len(image_groups)
    discriminator.train()
    generator.eval()

    losses = defaultdict(list)
    for i, images in enumerate(image_groups):
        real = to_var(images, device=discriminator)
        fake = generator(latent(real))

        real_logits = discriminator(real)
        fake_logits = discriminator(fake)

        dis_fake = functional.softplus(fake_logits).mean()
        dis_real = functional.softplus(-real_logits).mean()
        loss = dis_fake + dis_real

        if i == 0 and r1_weight > 0:
            r1 = r1_loss(images, discriminator)
            loss = loss + r1_weight * r1
            losses['dis_r1'].append(to_np(r1))

        optimizer_step(disc_optimizer, loss, **optimizer_kwargs)
        losses['dis_fake'].append(to_np(dis_fake))
        losses['dis_real'].append(to_np(dis_real))

    # it's ok, `real` is already defined
    generator.train()
    discriminator.eval()
    fake = generator(latent(real))
    fake_logits = discriminator(fake)

    loss = functional.softplus(-fake_logits).mean()
    optimizer_step(gen_optimizer, loss, **optimizer_kwargs)
    return {**dmap(np.mean, losses), 'gen': to_np(loss).item()}


def r1_loss(images, discriminator):
    images = to_var(images, device=discriminator, requires_grad=True)
    logits = discriminator(images)
    grads, = torch.autograd.grad(
        logits, images, torch.ones_like(logits),
        create_graph=True, retain_graph=True
    )
    return (grads ** 2).sum() / len(logits)
