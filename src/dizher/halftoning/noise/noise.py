# !/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
import os

import imageio
import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage import img_as_ubyte, img_as_float


BLUE_NOISE_RESOLUTION = 512
BLUE_NOISE_FILE_GLOB = os.path.join(
    os.path.dirname(__file__), "data/{:d}_{:d}/LDR_LLL1_?.png".format(BLUE_NOISE_RESOLUTION, BLUE_NOISE_RESOLUTION)
)

def load_frames_as_animation(path):
    return np.asarray([imageio.imread(name) for name in sorted(glob(path))])


def load_anim_noise(path):
    frames = [rgb2gray(rgba2rgb(frame)) for frame in load_frames_as_animation(path)]
    frames.extend([frame.T for frame in frames])
    all_frames = frames[:]
    for r in range(3):
        for frame in frames:
            frame = np.rot90(frame)
            all_frames.append(frame)
    return np.asarray(all_frames)


def load_static_noise(path):
    for name in sorted(glob(path)):
        return rgb2gray(rgba2rgb(imageio.imread(name)))


blue_noise = load_static_noise(BLUE_NOISE_FILE_GLOB)


def apply_wrapped(ufunc, dest, src):
    noise_h, noise_w = src.shape
    img_h, img_w = dest.shape
    # TODO Maybe np.pad with wrap?
    for r in range(0, img_h, noise_h):
        for c in range(0, img_w, noise_w):
            win_h = min(img_h - r, noise_h)
            win_w = min(img_w - c, noise_w)
            frame_dst = dest[r:r + win_h, c:c + win_w]
            frame_src = src[:win_h, :win_w]
            ufunc(frame_dst, frame_src, out=frame_dst)


def add_noise(image, noise, noise_range=1):
    image = img_as_float(image, force_copy=True)
    # print("Scr values:", np.unique(image), " noise values:", np.unique(noise), "range:", noise_range)
    noise_offseted = (noise - noise_range / 2) * 0.99999
    # print("Offset values:", np.unique(noise_offseted))
    apply_wrapped(np.add, image, noise_offseted)
    # print("Result values:", np.unique(image))
    return np.clip(image, 0, 1)


def noise_dither(image, noise_range=1, noise=blue_noise):
    return add_noise(image, noise, noise_range=noise_range) >= 0.5
