from itertools import product

import numpy as np


def get_grid(shape):
    # TODO: generalize to ndim
    i = np.arange(shape[0])
    j = np.arange(shape[1])
    return np.array(np.meshgrid(j, i))


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


def get_overlaps(coordinates, anchor_shape, box_corner, box_shape):
    anchor_shape = anchor_shape.flatten()
    half_shape = anchor_shape / 2
    end = np.array(box_corner) + box_shape - half_shape
    begin = box_corner + half_shape

    result = []
    for shape, coord, e, b in zip(anchor_shape, coordinates, end, begin):
        temp = shape + np.minimum(coord, e) - np.maximum(coord, b)
        temp[temp < 0] = 0
        result.append(temp)

    return np.prod(result, axis=0) / np.prod(anchor_shape)


def get_mask_and_params(coordinates, boxes, anchor_shape):
    mask = params = 0
    for box in boxes:
        overlaps = get_overlaps(coordinates, anchor_shape, box[:2], box[2:]) > .5
        mask = np.logical_or(mask, overlaps)
        params += (overlaps & np.logical_not(params)) * add_spatial(box, coordinates.ndim)
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
    return np.stack(masks), np.concatenate(params)


def get_gt(img_spatial_shape, fmap_spatial_shape, boxes, anchors):
    grid = get_grid(fmap_spatial_shape)
    scale = add_spatial(np.array(img_spatial_shape) / fmap_spatial_shape, grid.ndim)
    masks, box_params = get_all_masks(grid * scale, boxes, anchors)
    return masks, box_params
