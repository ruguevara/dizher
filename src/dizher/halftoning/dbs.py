"""Direct Binary Search halftoning under a per-pixel two-colour constraint.

Minimises E(b) = sum_k (|| h_k * (y_k - x_k) ||^2 + noise_k ||y_k - x_k||^2)
                + structure * sum_p c_p (1 - SSIM_p(y_0, x_0))
where x is scalar luminance or weighted opponent colour, h the eye-model kernels
(converter/eye.py), and y = paper + b * (ink - paper) with colours given per pixel.

Tone term: toggling pixel n changes y by a_n and E by  a_n^2 * cpp[0] + 2 a_n * (cpp * e)[n]  with
cpp = h (*) h the filter autocorrelation and e = y - x.

Structure term (Pang et al. 2008, contrast weighting from Jiang et al. 2023): SSIM_p is the
structural similarity in a Gaussian window centred on p, weighted by the local contrast c_p of the
target so flat regions do not get holes. A toggle at n changes the window statistics of every
window containing n by a known amount, so its exact energy change is evaluated by re-computing
SSIM at the (2R+1)^2 windows around n; no linearisation.

Moves are a toggle, or a swap with one of the 8 neighbours (a toggle of both), which moves a dot
without changing the local tone; toggle-only search stalls at about twice the energy. The pixels
of a move interact only within the interaction radius (cpp radius, or 2R for SSIM windows), so
one lattice phase of pixels, spaced further than that plus the swap reach, is updated at once
(parallel DBS).

Analoui & Allebach, "Model-based halftoning using direct binary search", 1992.
Lieberman & Allebach, "A dual interpretation for direct binary search", 2000.
Pang, Qu, Wong, Cohen-Or, Heng, "Structure-aware halftoning", SIGGRAPH 2008.
Jiang et al., "Efficient halftoning via deep reinforcement learning", IEEE TIP 2023 (CSSIM).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2

from ..converter.eye import eye_kernel
from ..progress import report_stage

SSIM_RADIUS = 3       # 7x7 window: at 256x192 an 11x11 window spans more than an attribute block
SSIM_SIGMA = 1.0
C1, C2 = 0.01 ** 2, 0.03 ** 2  # SSIM constants for unit range
CONTRAST_GAIN = 2.0   # c_p = clip(gain * local std of target, 0, 1)
NEIGHBOURS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # swap partners

def _blur(image, kernel):
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)

class _Structure:
    """Running window statistics of y against a fixed target x; exact toggle deltas per lattice phase."""

    def __init__(self, x, contrast, radius=SSIM_RADIUS, sigma=SSIM_SIGMA):
        self.R = R = radius
        self.w = cv2.getGaussianKernel(2 * R + 1, sigma, cv2.CV_32F)
        self.w = (self.w @ self.w.T).astype(np.float32)
        self.x, self.xp = x, np.pad(x, 1)
        self.mx = _blur(x, self.w)
        self.vx = _blur(x * x, self.w) - self.mx ** 2
        pad = lambda a: np.pad(a, R + 1)  # windows centred outside the image get zero contrast weight; +1 for swap partners
        self.c = pad(np.clip(contrast * np.sqrt(np.maximum(self.vx, 0)), 0, 1))
        self.mx, self.vx = pad(self.mx), pad(self.vx)
        self.my = self.wyy = self.wxy = None

    def set(self, y):
        P = self.R + 1
        self.my = np.pad(_blur(y, self.w), P)
        self.wyy = np.pad(_blur(y * y, self.w), P)
        self.wxy = np.pad(_blur(self.x * y, self.w), P)
        self.ssim = self._ssim(self.my, self.wyy, self.wxy)

    def _ssim(self, my, wyy, wxy):
        vy = wyy - my * my
        cov = wxy - self.mx * my
        return (2 * self.mx * my + C1) * (2 * cov + C2) / ((self.mx ** 2 + my ** 2 + C1) * (self.vx + vy + C2))

    def delta(self, r, c, lattice, a, y, offsets=((0, 0),)):
        """Energy change sum_p c_p (SSIM_old - SSIM_new), one row per offset, for toggling every pixel n of
        phase (r, c) by a[n] together with its neighbour n + offset by a[n + offset]; (0, 0) is a plain toggle."""
        P = self.R + 1
        nr, nc = a[r::lattice, c::lattice].shape
        ap, yp = np.pad(a, 1), np.pad(y, 1)
        at = lambda arr, oy, ox: arr[r + oy + 1::lattice, c + ox + 1::lattice][:nr, :nc, None, None]
        # the (2P+1)^2 windows around each phase pixel: every one that may contain n or n + offset
        win = lambda arr: sliding_window_view(arr, (2 * P + 1, 2 * P + 1))[r::lattice, c::lattice]
        an, yn, xn = at(ap, 0, 0), at(yp, 0, 0), at(self.xp, 0, 0)
        wn = np.pad(self.w, 1)   # weight of n in the window centred at n + o
        mx, vx, cp, old = win(self.mx), win(self.vx), win(self.c), win(self.ssim)
        my0 = win(self.my) + an * wn
        wyy0 = win(self.wyy) + an * (2 * yn + an) * wn
        wxy0 = win(self.wxy) + an * xn * wn
        out = []
        for dr, dc in offsets:
            my, wyy, wxy = my0, wyy0, wxy0
            if (dr, dc) != (0, 0):
                am, ym, xm = at(ap, dr, dc), at(yp, dr, dc), at(self.xp, dr, dc)
                wm = np.roll(wn, (dr, dc), (0, 1))   # weight of n + offset in the window centred at n + o
                my, wyy, wxy = my + am * wm, wyy + am * (2 * ym + am) * wm, wxy + am * xm * wm
            new = (2 * mx * my + C1) * (2 * (wxy - mx * my) + C2) / ((mx ** 2 + my ** 2 + C1) * (vx + wyy - my * my + C2))
            out.append((cp * (old - new)).sum((-1, -2)))
        return np.stack(out)

    def update(self, dy, y_old):
        P = self.R + 1
        self.my[P:-P, P:-P] += _blur(dy, self.w)
        self.wyy[P:-P, P:-P] += _blur(dy * (2 * y_old + dy), self.w)
        self.wxy[P:-P, P:-P] += _blur(self.x * dy, self.w)
        self.ssim = self._ssim(self.my, self.wyy, self.wxy)

def dbs_duo(luma, paper, ink, init, scale=1.4, alpha=2.0, structure=0.06, max_sweeps=10,
            stop_fraction=1e-3, kernels=None, noise=0, on_step=None):
    luma, paper, ink = [np.asarray(a, dtype=np.float32) for a in (luma, paper, ink)]
    if luma.ndim == 2:
        luma, paper, ink = [a[..., None] for a in (luma, paper, ink)]
    channels = luma.shape[-1]
    if kernels is None:
        kernels = [eye_kernel(scale, alpha)] * channels
    noise = np.broadcast_to(noise, (channels,))
    if len(kernels) != channels or not np.isfinite(noise).all() or (noise < 0).any():
        raise ValueError('DBS needs one kernel and a finite nonnegative noise weight per channel')
    cpp, centres = [], []
    for h, n in zip(kernels, noise):
        radius = h.shape[0] // 2
        k = _blur(np.pad(h, radius), h)  # full autocorrelation: do not truncate its cross terms
        k[2 * radius, 2 * radius] += n
        cpp.append(k)
        centres.append(k[2 * radius, 2 * radius])
    c0 = np.asarray(centres, dtype=np.float32)
    groups = {}   # channels sharing a kernel object are filtered in one call
    for k, h in enumerate(kernels):
        groups.setdefault(id(h), []).append(k)
    cpp_near = np.stack([k[k.shape[0] // 2 - 1:k.shape[0] // 2 + 2, k.shape[1] // 2 - 1:k.shape[1] // 2 + 2] for k in cpp], -1)
    lattice = max(k.shape[0] // 2 for k in cpp) + 3  # swap partners reach one pixel further on each side

    b = init.astype(bool)
    span = (ink - paper).astype(np.float32)
    y = (paper + b[..., None] * span).astype(np.float32)
    e = y - luma
    struct = None
    if structure > 0:
        struct = _Structure(luma[..., 0], CONTRAST_GAIN)
        struct.set(y[..., 0])
        lattice = max(lattice, 2 * struct.R + 3)  # two moves closer than 2R (plus swap reach) share a window
    mask = np.zeros_like(b)
    toggled = None
    for sweep in range(max_sweeps):
        report_stage(f'DBS sweep {sweep + 1}' + (f' · {toggled} moves' if toggled is not None else ''))
        toggled = 0
        for r in range(min(lattice, b.shape[0])):
            for c in range(min(lattice, b.shape[1])):
                g = np.empty_like(e)
                for group in groups.values():
                    g[..., group] = _blur(np.ascontiguousarray(e[..., group]), cpp[group[0]]).reshape(g[..., group].shape)
                a = np.where(b[..., None], -span, span)
                delta = (a * a * c0 + 2 * a * g).sum(-1)
                sl = (slice(r, None, lattice), slice(c, None, lattice))
                best = delta[sl].copy()
                if struct is not None:
                    sdelta = structure * struct.delta(r, c, lattice, a[..., 0], y[..., 0], [(0, 0)] + NEIGHBOURS)
                    best += sdelta[0]
                rows, cols = np.arange(r, b.shape[0], lattice), np.arange(c, b.shape[1], lattice)
                partner = np.zeros(best.shape, dtype=np.int8)  # 0: toggle, i + 1: swap with NEIGHBOURS[i]
                for i, (dr, dc) in enumerate(NEIGHBOURS):
                    shift = (-dr, -dc), (0, 1)
                    inside = ((rows + dr >= 0) & (rows + dr < b.shape[0]))[:, None] & ((cols + dc >= 0) & (cols + dc < b.shape[1]))[None, :]
                    am = np.roll(a, *shift)[sl]
                    d = delta[sl] + np.roll(delta, *shift)[sl] + 2 * (a[sl] * am * cpp_near[1 + dr, 1 + dc]).sum(-1)
                    if struct is not None:
                        d += sdelta[i + 1]
                    better = inside & (np.roll(b, *shift)[sl] != b[sl]) & (d < best)
                    best[better] = d[better]
                    partner[better] = i + 1
                n = int((best < 0).sum())
                if n == 0:
                    continue
                toggle = np.zeros_like(b)
                toggle[sl] = best < 0
                for i, (dr, dc) in enumerate(NEIGHBOURS):  # the swap partners toggle too
                    mask[:] = False
                    mask[sl] = (best < 0) & (partner == i + 1)
                    toggle |= np.roll(mask, (dr, dc), (0, 1))
                dy = np.where(toggle[..., None], a, 0).astype(np.float32)
                b[toggle] ^= True
                if struct is not None:
                    struct.update(dy[..., 0], y[..., 0])
                y += dy
                e += dy
                toggled += n
                if on_step:
                    on_step(b)
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
    d = s.delta(2, 3, lattice, a, y)[0]
    for rr, cc in [(2, 3), (2, 3 + lattice), (2 + lattice, 3)]:
        y2 = y.copy()
        y2[rr, cc] += a[rr, cc]
        brute = structure_energy(y2)[0] - E0
        assert abs(brute - d[rr // lattice, cc // lattice]) < 1e-4, (brute, d[rr // lattice, cc // lattice])
    for (dr, dc), d in zip(NEIGHBOURS, s.delta(2, 3, lattice, a, y, NEIGHBOURS)):  # swap: both pixels toggle
        y2 = y.copy()
        y2[2, 3] += a[2, 3]
        y2[2 + dr, 3 + dc] += a[2 + dr, 3 + dc]
        brute = structure_energy(y2)[0] - E0
        assert abs(brute - d[0, 0]) < 1e-4, ((dr, dc), brute, d[0, 0])

    init = rng.random((H, W)) < x
    plain = dbs_duo(x, paper, ink, init, structure=0.0)
    saw = dbs_duo(x, paper, ink, init, structure=0.5)
    ssim = lambda b: structure_energy(b.astype(np.float32))[1].ssim.mean()
    assert ssim(saw) > ssim(plain), (ssim(saw), ssim(plain))
    print('ok', ssim(plain), ssim(saw))
