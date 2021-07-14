# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from skimage import img_as_float


def pre_dither_gamma_corr(image, n_levels, gamma_correct=2.2):
    X = img_as_float(image)
    Y = np.zeros_like(X)
    for i in range(n_levels):
        dx = 1 / (n_levels - 1)
        x1 = i * dx
        x2 = x1 + dx
        y1, y2 = x1 ** gamma_correct, x2 ** gamma_correct
        dy = y2 - y1
        Xi = np.logical_and(x1 <= X, X < x2)
        Y[Xi] = (X[Xi] ** gamma_correct - y1) / dy * dx + x1
    return Y
