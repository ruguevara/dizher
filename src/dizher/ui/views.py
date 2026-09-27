"""Debug views of a conversion: numpy images from a Converter, no imgui."""
import numpy as np

import cv2

from ..converter.energy import LIGHTNESS_REF, local_metric

ERROR_GAIN = 1.5   # the error view's contrast: local metric units -> linear RGB offset from mid grey
SEAM_DIM = 0.3     # brightness of the image under the seam view's lines


def error_view(c) -> np.ndarray:
    """Result minus target as the energy sees them (each channel of the weighted local metric eye-blurred), drawn
    around mid grey as mid grey's metric would show it: lighter or darker where the result's lightness is too high
    or low, tinted with the colour it adds (its complement where it misses one). Mid grey is no error."""
    w = c.energy.weights
    weight = np.sqrt(np.array([w['Luma'], w['Chroma'], w['Chroma']], dtype=np.float32))
    error = c.opponent(c.dithered_result ** c.gamma - c.image_lrgb) / weight   # unweighted, then blurred
    error = np.stack([cv2.filter2D(np.ascontiguousarray(error[..., k]), -1, h, borderType=cv2.BORDER_REFLECT_101)
                      for k, h in enumerate(c.eye_kernels())], axis=-1) * weight * ERROR_GAIN
    grey = np.full((1, 1, 3), LIGHTNESS_REF, dtype=np.float32)
    back = np.linalg.inv(local_metric(grey, c.flare)[0, 0])
    return (LIGHTNESS_REF + error @ back.T).clip(0, 1) ** (1 / c.gamma)


def energy_view(c) -> np.ndarray:
    """Each block's energy: red its own term, green the eye-model seams (their cancelling part dropped), blue the
    coherence cost. Each on its own scale, full at its 99th percentile: the own term is most of the total, so on
    one scale the other two would not show."""
    e = c.energy.cell_energies(c.best_attr_indexes).clip(0)
    return c.expand_cells(e / np.maximum(np.percentile(e, 99, axis=(0, 1)), 1e-12)).clip(0, 1)


def seam_view(c) -> np.ndarray:
    """Each block seam drawn over the dimmed target, brighter the more it counts as a real edge of the original,
    where a pair change costs no coherence."""
    Lh, Lv = c.energy.seam_smoothness()
    h, w = c.cell
    out = c.image_rgb * SEAM_DIM
    edge = c.expand_cells(1 - Lh)[..., None]                    # (R-1)h x W: row r's seam with row r+1
    for dy in (h - 1, h):                                       # the pixel rows either side of it
        out[dy::h][:len(Lh)] = np.maximum(out[dy::h][:len(Lh)], edge[::h])
    edge = c.expand_cells(1 - Lv)[..., None]                    # H x (C-1)w
    for dx in (w - 1, w):
        out[:, dx::w][:, :Lv.shape[1]] = np.maximum(out[:, dx::w][:, :Lv.shape[1]], edge[:, ::w])
    return out
