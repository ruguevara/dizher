# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from skimage import img_as_ubyte

order4x4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
]) / 15 * 240 + 8


def ordered_dither(buffer, offset=0, order=order4x4):
    h, w = buffer.shape[-2:]
    oh, ow = order.shape
    offset %= oh
    tiled = np.tile(order, (-(-(h + offset) // oh), -(-w // ow)))   # ceil division: cover any screen size
    return (img_as_ubyte(buffer) > tiled[offset:h + offset, :w]) * 255
