# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from skimage import img_as_ubyte

order4x4 = np.tile((np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7,  13, 5],
]) ) / 15 * 240 + 8 , (64, 64))


def ordered_dither(buffer, offset=0, order=order4x4):
    h, w = buffer.shape[-2:]
    offset = offset % 4
    return (img_as_ubyte(buffer) > order[offset:h+offset, :w]) * 255
