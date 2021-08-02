# -*- coding: utf-8 -*-

import numpy as np
import cv2
from skimage import img_as_float


def reshape_by_charblock(image, axis=1):
    height, width = image.shape[axis:axis+2]
    rows, cols = height // 8, width // 8
    reshaped = image.reshape(image.shape[:axis] + (rows, 8, cols, 8, -1))
    axis_numbers = list(range(len(reshaped.shape)))
    axis_numbers[axis + 1], axis_numbers[axis + 2] = axis_numbers[axis + 2], axis_numbers[axis + 1]  # swap to rows and cols together
    transposed = reshaped.transpose(axis_numbers)
    if transposed.shape[-1] == 1:
        transposed = transposed[..., 0]
    return transposed

def reshape_from_charblocks(image):
    rows, cols = image.shape[:2]
    height, width = rows * 8, cols * 8
    axis_i = list(range(len(image.shape)))
    axis_i[1:3] = 2, 1
    transposed = image.transpose(axis_i).reshape(height, width, -1)
    if transposed.shape[-1] == 1:
        transposed = transposed[..., 0]
    return transposed

def select_best_charblocks(data, best_attr_indexes):
    n_combs, height, width = data.shape[:3]
    rows, cols = height // 8, width // 8
    rows_i, cols_i = np.mgrid[:rows, :cols]
    return reshape_from_charblocks(reshape_by_charblock(data)[best_attr_indexes, rows_i, cols_i,...])

def attrs2rgb(attrs):
    # TODO refactor to AttrBlocksScreen
    return cv2.resize(img_as_float(attrs), (0, 0), fx=8, fy=8, interpolation=cv2.INTER_NEAREST)

def apply_attrs(bitmap, paper, ink):
    # TODO refactor to AttrBlocksScreen
    return np.where(bitmap[..., np.newaxis], ink, paper)
