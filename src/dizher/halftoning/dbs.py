"""Direct Binary Search halftoning under a per-pixel two-colour constraint.

Minimises E(b) = || h * (y - x) ||^2 where x is the target linear luminance, h the eye-model
kernel (converter/eye.py), and y = paper + b * (ink - paper) with paper/ink luminance given per pixel.
Toggling pixel n changes y by a_n and E by  a_n^2 * cpp[0] + 2 a_n * (cpp * e)[n]  with
cpp = h (*) h the filter autocorrelation and e = y - x. Toggles farther apart than the radius
of cpp do not interact, so one lattice phase of pixels is updated at once (parallel DBS).

Analoui & Allebach, "Model-based halftoning using direct binary search", 1992.
Lieberman & Allebach, "A dual interpretation for direct binary search", 2000.
"""
import numpy as np
import cv2

from ..converter.eye import eye_kernel

def dbs_duo(luma, paper, ink, init, scale=1.4, alpha=2.0, max_sweeps=10, stop_fraction=1e-3):
    h = eye_kernel(scale, alpha)
    radius = h.shape[0] // 2
    cpp = cv2.filter2D(np.pad(h, radius), -1, h, borderType=cv2.BORDER_CONSTANT)  # full autocorrelation
    c0 = cpp[2 * radius, 2 * radius]
    # drop the negligible tail of cpp: it sets the lattice spacing and hence the number of passes per sweep
    keep = np.argwhere(cpp > 1e-2 * c0)
    r = int(np.abs(keep - 2 * radius).max())
    cpp = cpp[2 * radius - r:2 * radius + r + 1, 2 * radius - r:2 * radius + r + 1]
    lattice = r + 1  # > radius of cpp, so same-phase toggles are independent

    # Luminance outside a pixel's own range is unrepresentable. Left in the target it becomes a permanent
    # error that the optimiser pays back with ink or paper lines along the neighbouring blocks' edges.
    luma = np.clip(luma, np.minimum(paper, ink), np.maximum(paper, ink))
    b = init.astype(bool)
    span = (ink - paper).astype(np.float32)
    e = (paper + b * span - luma).astype(np.float32)
    mask = np.zeros_like(b)
    for _ in range(max_sweeps):
        toggled = 0
        for r in range(lattice):
            for c in range(lattice):
                g = cv2.filter2D(e, -1, cpp, borderType=cv2.BORDER_CONSTANT)
                a = np.where(b, -span, span)
                delta = a * a * c0 + 2 * a * g
                mask[:] = False
                mask[r::lattice, c::lattice] = True
                toggle = mask & (delta < 0)
                b[toggle] ^= True
                e[toggle] += a[toggle]
                toggled += int(toggle.sum())
        if toggled < stop_fraction * b.size:  # converged in ~5 sweeps in practice; the tail buys nothing visible
            break
    return b
