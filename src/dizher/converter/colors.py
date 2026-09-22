# -*- coding: utf-8 -*-

import cv2
from skimage.color import lab2lch, lch2lab
import numpy as np


RGB_LUMINANCE_709YUV = np.array([0.2126, 0.7152, 0.0722])
RGB_LUMINANCE_240M   = np.array([0.212, 0.701, 0.087])
RGB_LUMINANCE_601YUV = np.array([0.299, 0.587, 0.114])


def lrgb2luminance(image_lrgb, luminances=RGB_LUMINANCE_709YUV):
    return np.sum(image_lrgb * luminances, axis=len(image_lrgb.shape)-1)





def convert_color_cv2(image, from_space, to_space):
    conv_code = from_space + '2' + to_space
    conv_code = conv_code.upper()
    return cv2.cvtColor(image, getattr(cv2, 'COLOR_' + conv_code))


def convert_color_through(image, from_space, inter_space, to_space):
    image_inter = convert_color_cv2(image, from_space, inter_space)
    return convert_color_cv2(image_inter, inter_space, to_space)


__convert_color_reentry_lock = False

def convert_color(image, from_space, to_space, fallback=True):
    global __convert_color_reentry_lock
    try:
        return convert_color_cv2(image, from_space, to_space)
    except AttributeError:
        if not fallback or __convert_color_reentry_lock:
            raise
        __convert_color_reentry_lock = True

        for try_intermediate in ('LAB', 'YUV'):
            try:
                return convert_color_through(image, from_space, try_intermediate, to_space)
            except AttributeError:
                pass

        conv_code = from_space + '2' + to_space
        converter = globals()[conv_code.lower()]
        __convert_color_reentry_lock = False
        return converter(image)
#         return convert_color_through(image, from_space, 'LAB', to_space)


def rgb2lab(image):
    return convert_color(image, 'RGB', 'LAB', fallback=False)

def rgb2luv(image):
    return convert_color(image, 'RGB', 'LUV', fallback=False)

def luv2rgb(image):
    return convert_color(image, 'LUV', 'RGB', fallback=False)

def lab2rgb(image):
    return convert_color(image, 'LAB', 'RGB', fallback=False)

def rgb2lch(image):
     return lab2lch(rgb2lab(image))

def lch2rgb(image):
    return lab2rgb(lch2lab(image))

def gray2rgb(image):
    return convert_color(image, 'GRAY', 'RGB')
