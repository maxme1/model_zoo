from functools import cache
from pathlib import Path

import numpy as np
from dpipe.im.utils import identity, build_slices

from torchvision.datasets.mnist import MNIST
from connectome import Source, meta


def stitch(images, process=identity):
    n = int(np.sqrt(len(images)))
    m = len(images) // n
    shape = np.array(images.shape[1:])
    result = np.empty_like(images, shape=shape * [n, m])
    images = images.reshape(n, m, *shape)
    for i in range(n):
        for j in range(m):
            start = [i, j] * shape
            result[build_slices(start, start + shape)] = process(images[i, j])

    return result


class MNISTSource(Source):
    _root: str

    @cache
    def _mnist(_root):
        return MNIST(Path(_root).expanduser(), download=True)

    def image(i, _mnist):
        return np.array(_mnist[i][0])

    @meta
    def ids(_mnist):
        return tuple(range(len(_mnist)))


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
