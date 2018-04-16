from itertools import product

import numpy as np


def get_grid(shape):
    ranges = [np.arange(s) for s in reversed(shape)]
    return np.array(np.meshgrid(*ranges))


def add_spatial(array, ndim):
    array = np.asarray(array)
    delta = ndim - array.ndim
    for _ in range(delta):
        array = np.expand_dims(array, axis=-1)
    return array


def get_anchor_shapes(scales, ratios):
    result = []
    for s, r in product(scales, ratios):
        x_dim = s / np.sqrt(r)
        result.append([x_dim, r * x_dim])
    return np.array(result)


def get_overlaps(first_begin, first_shape, second_begin, second_shape):
    delta = np.minimum(first_begin + first_shape, second_begin + second_shape) - np.maximum(first_begin, second_begin)
    delta[delta < 0] = 0
    inter = np.prod(delta, axis=0)

    return inter / np.prod(first_shape, axis=0)


def get_mask_and_params(coordinates, boxes, anchor_shape):
    mask = params = 0
    anchor_shape = add_spatial(anchor_shape, coordinates.ndim)
    for box in boxes:
        box = add_spatial(box, coordinates.ndim)
        overlaps = get_overlaps(coordinates, anchor_shape, box[:2], box[2:]) > .5
        mask = np.logical_or(mask, overlaps)
        params += (overlaps & np.logical_not(params)) * box
    return mask, params


def get_all_masks(coordinates, boxes, anchor_shapes):
    masks, params = [], []
    for anchor_shape in anchor_shapes:
        mask, pars = get_mask_and_params(coordinates, boxes, anchor_shape)

        anchor_shape = add_spatial(anchor_shape, coordinates.ndim)
        space = (pars[:2] - coordinates) / anchor_shape
        div = pars[2:] / anchor_shape
        # avoiding -inf in log
        # it's ok to write garbage here, because these values will be eventually masked out
        div[div == 0] = 1e-6
        shape = np.log(div)
        pars = np.concatenate([space, shape])

        masks.append(mask)
        params.append(pars)
    return np.stack(masks), np.stack(params)


def get_gt(img_spatial_shape, boxes, fmap_spatial_shape, anchors):
    grid = get_grid(fmap_spatial_shape)
    scale = add_spatial(np.array(img_spatial_shape) / fmap_spatial_shape, grid.ndim)
    masks, box_params = get_all_masks(grid * scale, boxes, anchors)
    return masks, box_params


def IoU(first_begin, first_shape, second_begin, second_shape):
    delta = np.minimum(first_begin + first_shape, second_begin + second_shape) - np.maximum(first_begin, second_begin)
    delta[delta < 0] = 0
    inter = np.prod(delta, axis=0)

    return inter / (np.prod(first_shape, axis=0) + np.prod(second_shape, axis=0) - inter)


def non_max_suppression(params, probs):
    result = []
    params = params.T
    while params.size:
        idx = probs.argmax()
        reference = params[idx]
        mask = IoU(params.T[:2], params.T[2:], reference[:2, None], reference[2:, None]) < .5
        params = params[mask]
        probs = probs[mask]

        result.append(reference)
    return np.array(result)
