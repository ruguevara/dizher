"""Candidate full-reference metrics (PLAN.md, "The candidates") behind one interface:

    score(name, ref, test, ppd, factor=None) -> (value, local_map or None)

ref, test: float32 sRGB (H, W, 3) in 0..1 at Spectrum resolution (256x192, square pixels). Lower is more similar for
every metric here; a natively "higher is better" score is flipped (each function says how). local_map, where the
metric gives one, is (H, W) at the Spectrum grid, in the units of the score.

ppd is Spectrum pixels per degree: 19.4 / 12.9 / 9.7 at zoom x2 / x3 / x4 on the HP Z27k G3 (27", 3840x2160, pitch
0.1554 mm) at 65 cm, which is 73 device pixels per degree. Two ways to put the viewing geometry in:
- metrics that take it (ColorVideoVDP, S-CIELAB) get ppd directly, on the 256x192 image;
- metrics that do not (SSIMULACRA2, Butteraugli, MS-SSIM, DISTS, LPIPS) see the image as the monitor shows it: each
  Spectrum pixel upscaled by nearest neighbour to `factor` x `factor` device pixels. The default factor is the
  zoom's (DEVICE_FACTOR: x2 4, x3 6, x4 8; the exact ones are 73 / ppd = 3.77, 5.66, 7.53, rounded to the even
  factors the voting page uses), else round(73 / ppd). These metrics then assume their own fixed viewing of a device
  pixel; they do not know the distance.
- the converter's eye baseline works in Spectrum pixels and ignores ppd.

Installed (2026-10-20, Python 3.13, CPU only):
- torch 2.14.1+cpu, torchvision 0.29.1+cpu: pip install --index-url https://download.pytorch.org/whl/cpu torch
  torchvision
- cvvdp 0.5.7 (ColorVideoVDP, gfxdisp/ColorVideoVDP, module pycvvdp) and piq 0.8.0: pip install cvvdp piq. piq
  downloads the VGG16 (torchvision) and the LPIPS / DISTS weights to ~/.cache/torch on the first call.
- ssimulacra2 and butteraugli_main from libjxl v0.11.2, built from source (Ubuntu's libjxl-tools 0.7 has neither):
  git clone --depth 1 --branch v0.11.2 --recursive --shallow-submodules https://github.com/libjxl/libjxl;
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON -DJPEGXL_ENABLE_BENCHMARK=OFF
  -DJPEGXL_ENABLE_OPENEXR=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_SJPEG=OFF -DJPEGXL_ENABLE_MANPAGES=OFF ..;
  cmake --build . -j4 --target ssimulacra2 butteraugli_main; the two binaries copied to $JXL_TOOLS/bin and
  build/lib/libjxl*.so*, build/third_party/brotli/libbrotli*.so* to $JXL_TOOLS/lib (default /opt/libjxl-0.11.2).
"""
import functools
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from dizher.converter.colors import lrgb2luminance
from dizher.converter.energy import LRGB2OPP, SRGB2XYZ, XYZ2OPP, lightness_gain
from dizher.converter.eye import eye_kernel

DEVICE_PPD = 73.0                                 # 0.1554 mm pixels at 650 mm
ZOOM_PPD = {'x2': 19.4, 'x3': 12.9, 'x4': 9.7}
DEVICE_FACTOR = {19.4: 4, 12.9: 6, 9.7: 8}
PEAK = 200.0          # cd/m2, a standard SDR office monitor (ColorVideoVDP's standard_4k), also Butteraugli's target
JXL_TOOLS = os.environ.get('JXL_TOOLS', '/opt/libjxl-0.11.2')


def device_factor(ppd: float) -> int:
    return DEVICE_FACTOR.get(ppd, max(1, round(DEVICE_PPD / ppd)))


def upscale(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def block_max(m: np.ndarray, factor: int) -> np.ndarray:
    """A device-pixel map back on the Spectrum grid: the max of each factor x factor block."""
    h, w = m.shape[0] // factor, m.shape[1] // factor
    return m[:h * factor, :w * factor].reshape(h, factor, w, factor).max(axis=(1, 3))


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def lightness(srgb: np.ndarray) -> np.ndarray:
    """CIELAB L* / 100 of sRGB (D65)."""
    y = srgb_to_linear(srgb) @ SRGB2XYZ[1]
    return np.where(y > (6 / 29) ** 3, 1.16 * np.cbrt(y) - 0.16, y * 9.033)


def _torch(image: np.ndarray):
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    if image.ndim == 2:
        image = image[..., None]
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32))[None]


# ColorVideoVDP

@functools.lru_cache(maxsize=None)
def _cvvdp(ppd: float):
    import torch
    import pycvvdp
    from pycvvdp.display_model import vvdp_display_geometry, vvdp_display_photo_eotf
    torch.set_num_threads(os.cpu_count() or 4)
    photometry = vvdp_display_photo_eotf(PEAK, contrast=1000, source_colorspace='sRGB', E_ambient=250)
    geometry = vvdp_display_geometry((256, 192), ppd=ppd)    # resolution is unused once ppd is fixed
    return pycvvdp.cvvdp(display_photometry=photometry, display_geometry=geometry, heatmap='raw', quiet=True,
                         device=torch.device('cpu'))


def cvvdp(ref, test, ppd, factor):
    """ColorVideoVDP 0.5.7, the image's own pixels at ppd (vvdp_display_geometry(ppd=...)), no upscaling. Display:
    sRGB, 200 cd/m2 peak, 1000:1, 250 lux office light (the photometry of its standard_4k). Score 10 - JOD (JOD 10
    is identical; a sub-threshold difference can score exactly 0, as a 0.01 sRGB shift of a grey ramp does); map:
    the raw heatmap, which the package gives as 1 - JOD_local / 10, times 10 (local values can go above 10)."""
    q, stats = _cvvdp(float(ppd)).predict(test.astype(np.float32), ref.astype(np.float32), dim_order='HWC')
    return 10 - float(q), 10 * stats['heatmap'][0, 0, 0].float().numpy()


# libjxl tools

def _jxl(tool, ref, test, factor, *args):
    with tempfile.TemporaryDirectory() as d:
        paths = [os.path.join(d, n) for n in ('ref.png', 'test.png')]
        for p, im in zip(paths, (ref, test)):   # 16-bit PNG without a profile: read as sRGB by both tools
            cv2.imwrite(p, np.round(np.clip(upscale(im, factor)[..., ::-1], 0, 1) * 65535).astype(np.uint16))
        env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(JXL_TOOLS, 'lib'))
        out = subprocess.run([os.path.join(JXL_TOOLS, 'bin', tool), *paths, *[a.format(d=d) for a in args]],
                             env=env, check=True, capture_output=True, text=True).stdout
        raw = os.path.join(d, 'map.pfm')
        return out, cv2.imread(raw, cv2.IMREAD_UNCHANGED) if os.path.exists(raw) else None


def ssimulacra2(ref, test, ppd, factor):
    """SSIMULACRA2 (libjxl v0.11.2 tools/ssimulacra2) on the device-pixel upscale. Native: 100 identical, higher
    better, can go below 0; flipped to 100 - score. No map."""
    out, _ = _jxl('ssimulacra2', ref, test, factor)
    return 100 - float(out.split()[-1]), None


def _butteraugli(ref, test, factor):
    out, m = _jxl('butteraugli_main', ref, test, factor, '--rawdistmap', '{d}/map.pfm', '--intensity_target',
                  str(PEAK))
    lines = out.split('\n')
    return float(lines[0]), float(lines[1].split(':')[1]), block_max(m if m.ndim == 2 else m[..., 0], factor)


def butteraugli(ref, test, ppd, factor):
    """Butteraugli (libjxl v0.11.2 tools/butteraugli_main) on the device-pixel upscale, intensity target 200 nits
    (its default is 80). Score: its 3-norm over the image (what libjxl's benchmark reports with the max); lower
    better. Map: the raw distance map (PFM; OpenCV reads it top-down), block max to the Spectrum grid."""
    _, pnorm, m = _butteraugli(ref, test, factor)
    return pnorm, m


def butteraugli_max(ref, test, ppd, factor):
    """As butteraugli, the score the classic max-norm (Butteraugli's own distance; ~1 is the just-noticeable)."""
    mx, _, m = _butteraugli(ref, test, factor)
    return mx, m


# S-CIELAB

# Zhang & Wandell 1997 (and the scielab code of Wandell's lab): each opponent channel filtered by a sum of
# Gaussians w_i exp(-(x^2 + y^2) / s_i^2), s_i in degrees, each Gaussian of unit sum, the sum renormalised to 1.
# The values as commonly reproduced from the 1997 paper; not yet checked against the paper itself.
SCIELAB_FILTERS = (((0.921, 0.0283), (0.105, 0.133), (-0.108, 4.336)),   # O1, luminance
                   ((0.531, 0.0392), (0.330, 0.494)),                    # O2, red-green
                   ((0.488, 0.0536), (0.371, 0.386)))                    # O3, blue-yellow
D65 = SRGB2XYZ.sum(axis=1)


def _scielab_filter(channel, ppd, filters):
    # exp(-r^2 / s^2) is a Gaussian of sigma s / sqrt(2)
    out = sum(w * gaussian_filter(channel, s * ppd / np.sqrt(2), mode='reflect') for w, s in filters)
    return out / sum(w for w, _ in filters)


def _xyz2lab(xyz):
    f = xyz / D65
    f = np.where(f > (6 / 29) ** 3, np.cbrt(f), f * (29 / 6) ** 2 / 3 + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def scielab(ref, test, ppd, factor):
    """S-CIELAB at ppd on the 256x192 image itself (its narrowest Gaussians, 0.03-0.05 deg, are under a Spectrum
    pixel at these ppd, so the Spectrum grid samples the filters well enough). sRGB EOTF, XYZ D65, Poirson & Wandell
    opponent space, the filters above, back to XYZ, CIELAB against the white D65, CIEDE2000 per pixel. Score: the
    mean dE00; map: dE00."""
    from skimage.color import deltaE_ciede2000
    labs = []
    for im in (ref, test):
        opp = srgb_to_linear(im.astype(np.float64)) @ (XYZ2OPP @ SRGB2XYZ).T.astype(np.float64)
        opp = np.stack([_scielab_filter(opp[..., k], ppd, f) for k, f in enumerate(SCIELAB_FILTERS)], -1)
        labs.append(_xyz2lab(opp @ np.linalg.inv(XYZ2OPP.astype(np.float64)).T))
    m = deltaE_ciede2000(*labs).astype(np.float32)
    return float(m.mean()), m


# piq: MS-SSIM, DISTS, LPIPS

def msssim(ref, test, ppd, factor):
    """MS-SSIM (piq 0.8.0, 5 scales, Gaussian 11 / 1.5) on CIELAB L* / 100 of the device-pixel upscale. Flipped:
    1 - MS-SSIM. No map."""
    import piq
    import torch
    with torch.no_grad():
        v = piq.multi_scale_ssim(_torch(lightness(upscale(test, factor))), _torch(lightness(upscale(ref, factor))),
                                 data_range=1.0)
    return 1 - float(v), None


@functools.lru_cache(maxsize=None)
def _piq_model(name):
    import piq
    return {'dists': piq.DISTS, 'lpips': piq.LPIPS}[name](reduction='none')


def dists(ref, test, ppd, factor):
    """DISTS (piq 0.8.0, the authors' weights) on the sRGB device-pixel upscale; a distance, lower better. No map.
    piq's DISTS.forward, like the authors' code, first resizes an image whose short side is over 256 to 256 by
    bilinear interpolation without antialiasing, which aliases the dots of an upscaled dither; that step is skipped
    here (the parent class's forward, which DISTS.forward returns 1 minus), so DISTS sees the device pixels."""
    import piq
    import torch
    with torch.no_grad():
        model, x, y = _piq_model('dists'), _torch(upscale(test, factor)), _torch(upscale(ref, factor))
        return 1 - float(super(piq.DISTS, model).forward(x, y)), None


def lpips(ref, test, ppd, factor):
    """LPIPS-VGG (piq 0.8.0, LPIPS v0.1 linear weights) on the sRGB device-pixel upscale; a distance. No map (piq
    averages it)."""
    import torch
    with torch.no_grad():
        return float(_piq_model('lpips')(_torch(upscale(test, factor)), _torch(upscale(ref, factor)))), None


# The converter's own eye model

EYE_WEIGHTS = np.sqrt(np.array([1.0, 2.0, 2.0], dtype=np.float32))  # Luma 1, Chroma 2: the Exact mixture preset


def eye(ref, test, ppd, factor):
    """The converter's colour error, simplified (tests/test_converter_quality.py colour_error): linear light by the
    converter's gamma 2.2, the opponent error (energy.LRGB2OPP), times the lightness gain at the reference's
    luminance (flare 0.1), weighted Luma 1, Chroma 2, each channel blurred by the Gaussian eye kernel of sigma 1
    Spectrum pixel (eye.py, radius 4), squared, summed over channels. No unblurred dot term, no coherence. In
    Spectrum pixels: ppd and factor are ignored. Score: the sum over the image; map: per pixel."""
    lr, lt = ref.astype(np.float32) ** 2.2, test.astype(np.float32) ** 2.2
    gain = lightness_gain(lrgb2luminance(lr), 0.1)
    error = ((lt - lr) @ LRGB2OPP.T) * gain * EYE_WEIGHTS
    k = eye_kernel(1.4, 2.0, max_radius=4)
    m = sum(cv2.filter2D(np.ascontiguousarray(error[..., c]), -1, k, borderType=cv2.BORDER_REFLECT_101) ** 2
            for c in range(3))
    return float(m.sum()), m


METRICS = dict(cvvdp=cvvdp, ssimulacra2=ssimulacra2, butteraugli=butteraugli, butteraugli_max=butteraugli_max,
               scielab=scielab, msssim=msssim, dists=dists, lpips=lpips, eye=eye)


def score(name: str, ref: np.ndarray, test: np.ndarray, ppd: float = 12.9, factor: int = None):
    """(value, local map or None), lower = more similar. factor: device pixels per Spectrum pixel for the metrics
    that see device pixels, by default the zoom's (device_factor)."""
    assert ref.shape == test.shape and ref.shape[-1] == 3
    return METRICS[name](ref, test, ppd, factor or device_factor(ppd))


def runtimes(ref, test, ppd=12.9, names=None, repeat=2) -> dict:
    """Seconds per call of each metric on one pair: (the first call, which loads models, the best of the rest)."""
    result = {}
    for name in names or METRICS:
        times = []
        for _ in range(repeat + 1):
            t = time.perf_counter()
            score(name, ref, test, ppd)
            times.append(time.perf_counter() - t)
        result[name] = (times[0], min(times[1:]))
    return result
