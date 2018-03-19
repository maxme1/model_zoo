from itertools import product

import numpy as np


def get_grid(shape):
    # TODO: generalize to ndim
    i = np.arange(shape[0])
    j = np.arange(shape[1])
    return np.array(np.meshgrid(j, i))


def get_overlaps(begin, end, anchor_shape, box_corner, box_shape):
    box_end = np.array(box_corner) + box_shape

    result = []
    for i in range(2):
        temp = np.minimum(end[i], box_end[i]) - np.maximum(begin[i], box_corner[i])
        temp[temp < 0] = 0
        result.append(temp)

    return np.prod(result, axis=0) / np.prod(anchor_shape)


def get_mask_and_params(coordinates, boxes, anchor_shape):
    mask = params = 0
    half_shape = (anchor_shape / 2)[:, None, None]
    begin = coordinates - half_shape
    end = coordinates + half_shape
    for box in boxes:
        overlaps = get_overlaps(begin, end, anchor_shape, box[:2], box[2:]) > .5
        mask = np.logical_or(mask, overlaps)
        # TODO: generalize to ndim
        params += (overlaps & np.logical_not(params)) * np.array(box)[:, None, None]
    return mask, params


def get_anchor_shapes(scales, ratios):
    result = []
    for s, r in product(scales, ratios):
        x_dim = s / np.sqrt(r)
        result.append([x_dim, r * x_dim])
    return np.array(result)


def get_all_masks(coordinates, boxes, anchor_shapes):
    masks, params = [], []
    for anchor_shape in anchor_shapes:
        mask, pars = get_mask_and_params(coordinates, boxes, anchor_shape)

        a_s = anchor_shape[:, None, None]
        space = (pars[:2] - coordinates) / a_s
        div = pars[2:] / a_s
        div[div == 0] = 1e-6
        shape = np.log(div)
        pars = np.concatenate([space, shape])

        masks.append(mask)
        params.append(pars)
    return np.stack(masks), np.concatenate(params)
