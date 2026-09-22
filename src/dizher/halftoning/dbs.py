"""Direct Binary Search halftoning under a per-pixel two-colour constraint.

Minimises E(b) = || h * (y - x) ||^2 + structure * sum_p c_p (1 - SSIM_p(y, x))
where x is the target linear luminance, h the eye-model kernel (converter/eye.py), and
y = paper + b * (ink - paper) with paper/ink luminance given per pixel.

Tone term: toggling pixel n changes y by a_n and E by  a_n^2 * cpp[0] + 2 a_n * (cpp * e)[n]  with
cpp = h (*) h the filter autocorrelation and e = y - x.

Structure term (Pang et al. 2008, contrast weighting from Jiang et al. 2023): SSIM_p is the
structural similarity in a Gaussian window centred on p, weighted by the local contrast c_p of the
target so flat regions do not get holes. A toggle at n changes the window statistics of every
window containing n by a known amount, so its exact energy change is evaluated by re-computing
SSIM at the (2R+1)^2 windows around n; no linearisation.

Toggles farther apart than the interaction radius (cpp radius, or 2R for SSIM windows) do not
interact, so one lattice phase of pixels is updated at once (parallel DBS).

Analoui & Allebach, "Model-based halftoning using direct binary search", 1992.
Lieberman & Allebach, "A dual interpretation for direct binary search", 2000.
Pang, Qu, Wong, Cohen-Or, Heng, "Structure-aware halftoning", SIGGRAPH 2008.
Jiang et al., "Efficient halftoning via deep reinforcement learning", IEEE TIP 2023 (CSSIM).
"""
import numpy as np
import cv2

from ..converter.eye import eye_kernel

SSIM_RADIUS = 3       # 7x7 window: at 256x192 an 11x11 window spans more than an attribute block
SSIM_SIGMA = 1.0
C1, C2 = 0.01 ** 2, 0.03 ** 2  # SSIM constants for unit range
CONTRAST_GAIN = 2.0   # c_p = clip(gain * local std of target, 0, 1)

def _blur(image, kernel):
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)

class _Structure:
    """Running window statistics of y against a fixed target x; exact toggle deltas per lattice phase."""

    def __init__(self, x, contrast, radius=SSIM_RADIUS, sigma=SSIM_SIGMA):
        self.R = R = radius
        self.w = cv2.getGaussianKernel(2 * R + 1, sigma, cv2.CV_32F)
        self.w = (self.w @ self.w.T).astype(np.float32)
        self.x = x
        self.mx = _blur(x, self.w)
        self.vx = _blur(x * x, self.w) - self.mx ** 2
        pad = lambda a: np.pad(a, R)  # windows centred outside the image get zero contrast weight
        self.c = pad(np.clip(contrast * np.sqrt(np.maximum(self.vx, 0)), 0, 1))
        self.mx, self.vx = pad(self.mx), pad(self.vx)
        self.my = self.wyy = self.wxy = None

    def set(self, y):
        R = self.R
        self.my = np.pad(_blur(y, self.w), R)
        self.wyy = np.pad(_blur(y * y, self.w), R)
        self.wxy = np.pad(_blur(self.x * y, self.w), R)
        self.ssim = self._ssim(self.my, self.wyy, self.wxy)

    def _ssim(self, my, wyy, wxy):
        vy = wyy - my * my
        cov = wxy - self.mx * my
        return (2 * self.mx * my + C1) * (2 * cov + C2) / ((self.mx ** 2 + my ** 2 + C1) * (self.vx + vy + C2))

    def delta(self, r, c, lattice, a, y):
        """Energy change sum_p c_p (SSIM_old - SSIM_new) for toggling every pixel of phase (r, c) by a."""
        R = self.R
        an, yn, xn = a[r::lattice, c::lattice], y[r::lattice, c::lattice], self.x[r::lattice, c::lattice]
        nr, nc = an.shape
        d = np.zeros_like(an)
        for oy in range(-R, R + 1):
            for ox in range(-R, R + 1):
                w = self.w[R + oy, R + ox]
                sl = (slice(r + oy + R, r + oy + R + lattice * (nr - 1) + 1, lattice),
                      slice(c + ox + R, c + ox + R + lattice * (nc - 1) + 1, lattice))
                cp = self.c[sl]
                my = self.my[sl] + an * w
                wyy = self.wyy[sl] + an * (2 * yn + an) * w
                wxy = self.wxy[sl] + an * xn * w
                new = (2 * self.mx[sl] * my + C1) * (2 * (wxy - self.mx[sl] * my) + C2) / (
                    (self.mx[sl] ** 2 + my ** 2 + C1) * (self.vx[sl] + wyy - my * my + C2))
                d += cp * (self.ssim[sl] - new)
        return d

    def update(self, dy, y_old):
        R = self.R
        self.my[R:-R, R:-R] += _blur(dy, self.w)
        self.wyy[R:-R, R:-R] += _blur(dy * (2 * y_old + dy), self.w)
        self.wxy[R:-R, R:-R] += _blur(self.x * dy, self.w)
        self.ssim = self._ssim(self.my, self.wyy, self.wxy)

def dbs_duo(luma, paper, ink, init, scale=1.4, alpha=2.0, structure=0.06, max_sweeps=10, stop_fraction=1e-3):
    h = eye_kernel(scale, alpha)
    radius = h.shape[0] // 2
    cpp = cv2.filter2D(np.pad(h, radius), -1, h, borderType=cv2.BORDER_CONSTANT)  # full autocorrelation
    c0 = cpp[2 * radius, 2 * radius]
    keep = np.argwhere(cpp > 1e-2 * c0)
    r = int(np.abs(keep - 2 * radius).max())
    cpp = cpp[2 * radius - r:2 * radius + r + 1, 2 * radius - r:2 * radius + r + 1]
    lattice = r + 1  # > radius of cpp, so same-phase toggles are independent

    luma = np.clip(luma, np.minimum(paper, ink), np.maximum(paper, ink)).astype(np.float32)
    b = init.astype(bool)
    span = (ink - paper).astype(np.float32)
    y = (paper + b * span).astype(np.float32)
    e = y - luma
    struct = None
    if structure > 0:
        struct = _Structure(luma, CONTRAST_GAIN)
        struct.set(y)
        lattice = max(lattice, 2 * struct.R + 1)  # two toggles closer than 2R share a window
    mask = np.zeros_like(b)
    for _ in range(max_sweeps):
        toggled = 0
        for r in range(lattice):
            for c in range(lattice):
                g = _blur(e, cpp)
                a = np.where(b, -span, span)
                delta = a * a * c0 + 2 * a * g
                if struct is not None:
                    delta[r::lattice, c::lattice] += structure * struct.delta(r, c, lattice, a, y)
                mask[:] = False
                mask[r::lattice, c::lattice] = True
                toggle = mask & (delta < 0)
                n = int(toggle.sum())
                if n == 0:
                    continue
                dy = np.where(toggle, a, 0).astype(np.float32)
                b[toggle] ^= True
                if struct is not None:
                    struct.update(dy, y)
                y += dy
                e += dy
                toggled += n
        if toggled < stop_fraction * b.size:  # converged in ~5 sweeps in practice; the tail buys nothing visible
            break
    return b

if __name__ == '__main__':
    # Self-check: exact toggle deltas must match brute-force energy differences, and the structure
    # term must raise SSIM against the target compared to plain DBS.
    rng = np.random.default_rng(0)
    H, W = 24, 32
    x = (rng.random((H, W)).astype(np.float32) * 0.3 + 0.3)
    x[:, W // 2:] += 0.3
    paper, ink = np.zeros_like(x), np.ones_like(x)

    def structure_energy(y):
        s = _Structure(x, CONTRAST_GAIN)
        s.set(y)
        return float((s.c * (1 - s.ssim)).sum()), s

    y = rng.random((H, W)).astype(np.float32).round()
    E0, s = structure_energy(y)
    lattice = 2 * s.R + 1
    a = np.where(y > 0.5, -1.0, 1.0).astype(np.float32)
    d = s.delta(2, 3, lattice, a, y)
    for rr, cc in [(2, 3), (2, 3 + lattice), (2 + lattice, 3)]:
        y2 = y.copy()
        y2[rr, cc] += a[rr, cc]
        brute = structure_energy(y2)[0] - E0
        assert abs(brute - d[rr // lattice, cc // lattice]) < 1e-4, (brute, d[rr // lattice, cc // lattice])

    init = rng.random((H, W)) < x
    plain = dbs_duo(x, paper, ink, init, structure=0.0)
    saw = dbs_duo(x, paper, ink, init, structure=0.5)
    ssim = lambda b: structure_energy(b.astype(np.float32))[1].ssim.mean()
    assert ssim(saw) > ssim(plain), (ssim(saw), ssim(plain))
    print('ok', ssim(plain), ssim(saw))
