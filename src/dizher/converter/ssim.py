# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.metrics import  structural_similarity

from .colors import convert_color


class SSIMComparer:
    def __init__(self, ref_image, downscale=2):
        self.ref_image = ref_image
        self.downscale = downscale
        self.pp_ref = self.preprocess(ref_image)

    @staticmethod
    def rgb2luv_norm(image):
        image_luv = convert_color(image.astype(np.float32), 'RGB', 'LUV')
        image_luv[..., 0] /= 100
        image_luv[..., 1:3] /= 180
        return image_luv

    def preprocess(self, image):
        return self.rgb2luv_norm(cv2.resize(image, (0, 0), fx=1/self.downscale, fy=1/self.downscale, interpolation=cv2.INTER_AREA))

    def __call__(self, image):
        # TODO we can cache further ref image parts in skimage structural_similarity
        return structural_similarity(self.pp_ref, self.preprocess(image),
                                     channel_axis=-1,
                                     gaussian_weights=True,
                                     sigma=1.5,
                                     use_sample_covariance=False,
                                     data_range=1.0,
                                    )

def greedy_ssim_optimize(converter):
    from .zxconverter import select_best_charblocks  # TODO refactor to separate file

    attr_indexes = converter.best_attr_indexes.copy()
    comparer = SSIMComparer(converter.image_rgb, downscale=2)
    invbright = converter.palette.invert_bright_index
    current_convert = select_best_charblocks(converter.recolorized, attr_indexes)
    max_ssim = comparer(current_convert)

    for row in range(24):
        for col in range(32):
            cur_index = attr_indexes[row, col]
            new_index = invbright(cur_index)
            attr_indexes[row, col] = new_index
            current_convert = select_best_charblocks(converter.recolorized, attr_indexes)
            new_ssim = comparer(current_convert)
            if new_ssim > max_ssim:
                max_ssim = new_ssim
            else:
                attr_indexes[row, col] = cur_index  # put it back
    converter.best_attr_indexes = attr_indexes
