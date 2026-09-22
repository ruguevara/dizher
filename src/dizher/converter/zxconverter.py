# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import Generator, List, Any, Sequence, Tuple, Type
from abc import abstractmethod
from collections import namedtuple

import numpy as np
import cv2
from skimage import img_as_ubyte, img_as_float

from .palette import Palette, ZXPalette
from .colors import convert_color, lrgb2luminance, gray2rgb
from .dither import Ditherer
from .ssim import greedy_ssim_optimize
from .metrics import ConversionMetric, MetricTuner
from .utils import reshape_by_charblock, attrs2rgb, apply_attrs, select_best_charblocks


FitDuocolorResult = namedtuple('FitDuocolorResult', ['ammount', 'reconstructed'])


class Converter:
    def __init__(self,
            metric_classes: Sequence[Type[ConversionMetric]],
            default_weights: Sequence[float],
            size: Tuple[int, int] = (192, 256),
            palette: Palette = ZXPalette(),
            gamma: float = 2.2
    ):
        assert isinstance(palette, Palette)
        assert len(size) == 2
        self.size = size
        self.palette = palette
        self.gamma = gamma
        self.metric_tuner = MetricTuner(self, metric_classes, default_weights)
        self.color_pairs = self.palette.color_pairs()

    def invalidate(self):
        self.image_rgb = None
        self.image_lrgb = None
        self.image_luma = None
        self.levels = None
        self.recolorized = None
        self.best_attr_indexes = None
        self.metric_tuner.invalidate()
        self.invalidate_result()

    def invalidate_result(self):
        self.best_recolor = None
        self.best_levels = None
        self.best_paper = None
        self.best_ink = None
        self.dithered_bitmap = None
        self.dithered_result = None

    def load_image(self, filename: str):
        self.set_image(convert_color(cv2.imread(filename), 'BGR', 'RGB'))

    def preprocess_image(self, image_rgb: np.ndarray) -> np.ndarray:
        image_rgb = img_as_float(image_rgb).astype(np.float32)
        assert image_rgb.shape[:2] == self.size, "Wrong image size!"

        if image_rgb.shape[2] == 1:
            image_rgb = gray2rgb(image_rgb)
        assert image_rgb.shape[2] == 3
        return image_rgb

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.image_rgb = self.preprocess_image(image_rgb)
        self.image_lrgb = self.image_rgb ** self.gamma
        self.image_luma = lrgb2luminance(self.image_lrgb)
        self.levels, self.recolorized = self.fit_duocolors()
        self.metric_tuner.calc_metrics()

    def fit_duocolors(self) -> FitDuocolorResult:
        assert self.image_rgb is not None
        supershape = tuple([len(self.color_pairs)]) + self.image_rgb.shape[:-1]
        converts = np.empty(supershape + tuple([3]))
        levels = np.empty(supershape)
        for i, (c1, c2) in enumerate(self.palette.iter_color_pairs()):
            levels[i], converts[i] = self.fit_duocolor(c1, c2)
        return FitDuocolorResult(levels, converts)

    def fit_duocolor(self, c1, c2) -> FitDuocolorResult:
        c1, c2 = np.asarray(c1, dtype=np.float32), np.asarray(c2, dtype=np.float32)
        c1_lrgb, c2_lrgb = c1 ** self.gamma, c2 ** self.gamma
        color_vec = c2_lrgb - c1_lrgb
        c1_luminance, c2_luminance = lrgb2luminance(c1_lrgb), lrgb2luminance(c2_lrgb)
        c_lum_range = c2_luminance - c1_luminance

        sign = np.sign(c_lum_range)
        if sign < 0:
            c1_lrgb, c2_lrgb = c2_lrgb, c1_lrgb
            color_vec = -color_vec
            c1_luminance, c2_luminance = c2_luminance, c1_luminance
            c_lum_range = -c_lum_range

        if c_lum_range > 0:
            c1_c2_ammount = (self.image_luma - c1_luminance) / c_lum_range
            c1_c2_ammount = c1_c2_ammount.clip(0, 1)
        else:
            c1_c2_ammount = np.zeros_like(self.image_luma)

        recolored = c1_lrgb + c1_c2_ammount[..., np.newaxis] * color_vec
        if sign < 0:
            c1_c2_ammount = 1 - c1_c2_ammount

        return FitDuocolorResult(c1_c2_ammount, recolored ** (1/self.gamma))

    def best_paper_ink(self, attr_indexes):
        best_paper_i = self.color_pairs[attr_indexes][..., 0, :]
        best_ink_i = self.color_pairs[attr_indexes][..., 1, :]
        best_paper = attrs2rgb(best_paper_i)
        best_ink   = attrs2rgb(best_ink_i)
        return best_paper, best_ink

    def calc_best_on_metrics(self):
        self.metric_tuner.apply()

    def set_best_conversion(self, attr_indexes):
        self.best_attr_indexes = attr_indexes
        self.best_recolor = select_best_charblocks(self.recolorized, self.best_attr_indexes)
        self.best_levels = select_best_charblocks(self.levels, self.best_attr_indexes)
        self.best_paper, self.best_ink = self.best_paper_ink(self.best_attr_indexes)

    def dither(self, ditherer: Ditherer) -> np.ndarray:
        self.dithered_bitmap = (ditherer(img_as_ubyte(self.best_levels)) > 0).astype(np.float32)  # TODO add paper and ink
        self.dithered_result = apply_attrs(self.dithered_bitmap, self.best_paper, self.best_ink)
        return self.dithered_result

    def optimize_brights(self):
        # TODO somehow simulate dithering with blue noise to account dithering effect in SSIM
        greedy_ssim_optimize(self)
        self.set_best_conversion(self.best_attr_indexes)
