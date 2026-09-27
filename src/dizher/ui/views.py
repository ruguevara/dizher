"""Debug views of a conversion: numpy images from a Converter, no imgui."""
import numpy as np

from ..converter.energy import LRGB2OPP

ERROR_GAIN = 1.5   # the error view's contrast: weighted opponent units -> linear RGB offset from mid grey
SEAM_DIM = 0.3     # brightness of the image under the seam view's lines


def error_view(c) -> np.ndarray:
    """Result minus target as the energy sees them (eye-blurred, weighted opponent) drawn around mid grey: lighter
    or darker where the luma is too high or low, tinted with the colour the result adds (its complement where it
    misses one). Mid grey is no error."""
    w = c.energy.weights
    weight = np.sqrt(np.array([w['Luma'], w['Chroma'], w['Chroma']], dtype=np.float32))
    error = (c.eye_opponent(c.dithered_result) - c.eye_opponent(c.image_rgb)) * weight * ERROR_GAIN
    return (0.5 ** c.gamma + error @ np.linalg.inv(LRGB2OPP).T).clip(0, 1) ** (1 / c.gamma)


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
