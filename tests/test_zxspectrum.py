"""Run: python tests/test_zxspectrum.py (or pytest)."""
import numpy as np

from dizher.platforms.zxspectrum import ZXPalette, STANDARD


def test_palette():
    p = ZXPalette()
    assert p.as_ubyte().tolist()[:4] == [[0, 0, 0], [0, 0, 205], [205, 0, 0], [205, 0, 205]]  # black, blue, red, magenta
    assert (p[15] == 255).all() and (p[7] == 205).all()
    pairs = list(p.iter_idxs_pairs())
    assert len(pairs) == 72 and all((i1 >= 8) == (i2 >= 8) for i1, i2 in pairs)
    assert len(list(p.with_subset('Mono').iter_idxs_pairs())) == 3 and p.subset == 'All colours'


def test_mode():
    assert STANDARD.size == (192, 256) and STANDARD.cell == (8, 8)
    bitmap = np.zeros(STANDARD.size, dtype=bool)
    idx = np.zeros((24, 32, 2), dtype=int)
    assert len(STANDARD.encode(bitmap, idx)) == 6912


def test_c64_subsets():
    from dizher.platforms.c64 import HIRES
    p = HIRES.palette
    assert len(list(p.iter_idxs_pairs())) == 136
    assert len(list(p.with_subset('Grayscale').iter_idxs_pairs())) == 15   # 5 greys: 5 + C(5, 2)
    assert list(p.with_subset('Mono').iter_idxs_pairs()) == [(0, 0), (0, 1), (1, 1)]
    lum = p.as_float() ** 2.2 @ [0.2126, 0.7152, 0.0722]
    assert all(lum[i1] <= lum[i2] for i1, i2 in p.iter_idxs_pairs())   # paper darker than ink, e.g. (2, 1) not (1, 2)
    assert (2, 1) in list(p.iter_idxs_pairs())


if __name__ == '__main__':
    test_palette(); test_mode(); test_c64_subsets(); print('ok')
