"""Run: python tests/test_zxspectrum.py (or pytest)."""
import numpy as np

from dizher.platforms.zxspectrum import ZXPalette, STANDARD


def test_palette():
    p = ZXPalette()
    assert p.as_ubyte().tolist()[:4] == [[0, 0, 0], [0, 0, 205], [205, 0, 0], [205, 0, 205]]  # black, blue, red, magenta
    assert (p[15] == 255).all() and (p[7] == 205).all()
    pairs = list(p.iter_idxs_pairs())
    assert len(pairs) == 72 and all((i1 >= 8) == (i2 >= 8) for i1, i2 in pairs)
    assert len(list(p.with_subset('Mono').iter_idxs_pairs())) == 3 and p.enabled == set(range(16))


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


def test_c64_art():
    from dizher.platforms.c64 import HIRES
    bitmap = np.zeros(HIRES.size, dtype=bool)
    bitmap[8, 8] = True   # cell (1, 1), its first row, leftmost pixel
    idx = np.zeros((25, 40, 2), dtype=int)
    idx[1, 1] = 6, 14     # paper blue, ink light blue
    data = HIRES.encode(bitmap, idx)
    assert len(data) == 9009 and data[:2] == b'\x00\x20'
    assert data[2 + (40 + 1) * 8] == 0x80 and sum(data[2:8002]) == 0x80
    assert data[8002 + 41] == 0xE6


if __name__ == '__main__':
    test_palette(); test_mode(); test_c64_subsets(); test_c64_art(); print('ok')
