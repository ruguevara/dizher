"""Eye model: isotropic alpha-stable blur  k(r) = exp(-(r / scale) ** alpha).

alpha = 2 is a Gaussian with sigma = scale / sqrt(2); alpha ~ 1 is an exponential kernel with a
sharper peak and heavier tails, closer to the spatial form of Näsänen's contrast sensitivity
function. Taken from the random-portrait experiments (alpha 0.95, size 7 on linear RGB).
"""
import numpy as np
import cv2

# Luma: Gaussian (alpha 2) near pixel resolution, where DBS dot placement is well understood and cheap.
# Chroma: Gaussian too, a little wider. A heavy-tailed chroma kernel (alpha 1) averages a 16 px alternation of
# blocks almost away and so rewards block-level colour dithering that is plainly visible; keep the tail light.
LUMA_ALPHA = 2.0
LUMA_SCALE = 1.4  # sigma 1.0 px for alpha 2
CHROMA_ALPHA = 2.0
CHROMA_SCALE = 1.4  # sigma 1.0 px

def eye_kernel(scale: float, alpha: float, cutoff: float = 1e-2) -> np.ndarray:
    # ponytail: 1% tail cutoff; 0.1% changes the DBS energy by <0.5% but doubles its run time
    radius = max(1, int(np.ceil(scale * (-np.log(cutoff)) ** (1 / alpha))))
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    k = np.exp(-(np.hypot(x, y) / scale) ** alpha).astype(np.float32)
    return k / k.sum()

def eye_blur(image: np.ndarray, scale: float, alpha: float) -> np.ndarray:
    return cv2.filter2D(image, -1, eye_kernel(scale, alpha), borderType=cv2.BORDER_REFLECT_101)
