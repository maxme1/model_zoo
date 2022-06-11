import numpy as np
from dpipe.im.utils import identity, build_slices


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
