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

FitDuocolorResult = namedtuple('FitDuocolorResult', ['ammount', 'reconstructed'])


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


class ConversionMetric:
    label = 'You can not get label of an abstract base ConversionMetric class'

    def __init__(self, converter):
        self.converter = converter

    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError()


class LumaMetric(ConversionMetric):
    label = 'Luma'

    def __call__(self, **kwargs):
        # returns error in range 0..1
        gamma = self.converter.gamma
        image_luma = self.converter.image_luma ** (1/gamma)
        reconstruct_luma = lrgb2luminance(self.converter.recolorized ** gamma) ** (1/gamma)
        return np.abs(reconstruct_luma - image_luma)


class ChromaMetric(ConversionMetric):
    label = 'Chroma'

    def __call__(self, blur_size=3, **kwargs):
        # TODO make params adjustable
        # returns error in range 0..1
        image_rgb = cv2.GaussianBlur(self.converter.image_rgb, ksize=(blur_size, blur_size), sigmaX=0)
        image_luv = convert_color(image_rgb, 'RGB', 'LUV')

        reconstruct_luv = np.empty_like(self.converter.recolorized)
        for i, recolorized in enumerate(self.converter.recolorized):
            reconstruct_rgb = cv2.GaussianBlur(recolorized, ksize=(blur_size, blur_size), sigmaX=0)
            reconstruct_luv[i] = convert_color(reconstruct_rgb.astype(np.float32), 'RGB', 'LUV')

        diff_u = (image_luv[..., 1] - reconstruct_luv[..., 1]) / 180
        diff_v = (image_luv[..., 2] - reconstruct_luv[..., 2]) / 180
        # TODO можно перевести LUV в LHS и учитывать расстояния по H и S с разными весами
        result = np.sqrt(diff_u ** 2 + diff_v ** 2)
        return result


class DitherMetric(ConversionMetric):
    label = 'Ditherness'

    def __call__(self, **kwargs):
        # слабые уровни — мало точек для градиента: плохо. Точек 50/50 — хорошо, точек почти 0/100 — хорошо
        purity = (1 - np.abs((self.converter.levels - 0.5) * 2))
        purity = (1 - np.abs((purity - 0.5) * 2))
        purity[purity < 1/64] = 0
        return purity


class SmoothnessMetric(ConversionMetric):
    label = 'Smoothness'

    def __call__(self, **kwargs):
        color_pair_luma = lrgb2luminance(self.converter.color_pairs) ** (1/self.converter.gamma)
        luma_dist = np.abs(color_pair_luma[:, 1] - color_pair_luma[:, 0])
        height, width = self.converter.size
        luma_dist = luma_dist[:, np.newaxis].repeat(height * width, axis=1).reshape(-1, height, width)
        return luma_dist


class Converter:
    def __init__(self,
            metric_classes: List[Type[ConversionMetric]],
            size: Tuple[int, int] = (192, 256),
            palette: Palette = ZXPalette(),
            gamma: float = 2.2
    ):
        assert isinstance(palette, Palette)
        assert len(size) == 2
        self.size = size
        self.palette = palette
        self.gamma = gamma
        self.metrics = OrderedDict((metric_class.label, metric_class(self)) for metric_class in metric_classes)
        self.metric_arrays = OrderedDict()
        self.color_pairs = self.palette.color_pairs()

    def invalidate(self):
        self.image_rgb = None
        self.image_lrgb = None
        self.image_luma = None
        self.levels = None
        self.recolorized = None
        self.metric_arrays = OrderedDict()
        self.best_attr_indexes = None
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
        self.image_lrgb = image_rgb ** self.gamma
        self.image_luma = lrgb2luminance(self.image_lrgb)
        self.levels, self.recolorized = self.fit_duocolors()
        self.calc_metrics()

    def calc_metrics(self) -> None:
        for label, metric in self.metrics.items():
            self.metric_arrays[label] = metric()

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

    def calc_best_on_metrics(self, weights: Sequence):
        assert len(weights) == len(self.metric_arrays), "len(weights) == {:d} != len(self.metric_arrays) == {:d}".format(len(weights), len(self.metric_arrays))
        integral_errors = np.sum([
            metric * weight
            # TODO calc metrics on arrays already reshaped by charblocks
            for weight, metric in zip(weights, self.metric_arrays.values())
        ], axis=0)

        integral_errors = reshape_by_charblock(integral_errors)
        mse_by_combs_and_blocks = ((integral_errors * 255) ** 2).sum(axis=(3, 4)) / 64
        self.set_best_conversion(mse_by_combs_and_blocks.argmin(0))

    def set_best_conversion(self, attr_indexes):
        self.best_attr_indexes = attr_indexes
        self.best_recolor = select_best_charblocks(self.recolorized, self.best_attr_indexes)
        self.best_levels = select_best_charblocks(self.levels, self.best_attr_indexes)
        self.best_paper, self.best_ink = self.best_paper_ink(self.best_attr_indexes)

    def dither(self, ditherer: Ditherer) -> np.ndarray:
        self.dithered_bitmap = ditherer(img_as_ubyte(self.best_levels)).astype(np.float32) / 255  # TODO add paper and ink
        self.dithered_result = apply_attrs(self.dithered_bitmap, self.best_paper, self.best_ink)
        return self.dithered_result

    def optimize_brights(self):
        # TODO somehow simulate dithering with blue noise to account dithering effect in SSIM
        greedy_ssim_optimize(self)
        self.set_best_conversion(self.best_attr_indexes)
