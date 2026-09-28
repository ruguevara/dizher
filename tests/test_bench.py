"""The pair-selection bench's helpers (tests/bench): screens, renders with rulers, variants, judgments, metrics."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench import render as R                                  # noqa: E402
from bench.scr import read_scr, shown, pairs_to_labels, label_pairs, parse_pair, pair_name, has_flash   # noqa: E402
from dizher.platforms.zxspectrum import ZXPalette              # noqa: E402
from dizher.platforms.zxspectrum.scr import to_scr             # noqa: E402
from dizher.platforms import Mode                              # noqa: E402
from dizher.converter.converter import Converter               # noqa: E402


def test_scr_round_trip_and_pairs(tmp_path):
    """to_scr -> read_scr gives the bitmap and (paper, ink) back; a labelling's pairs map back to the same labels,
    the two blacks one colour; a solid cell shows one colour twice."""
    rng = np.random.default_rng(0)
    bitmap = rng.random((192, 256)) < 0.5
    bitmap[:8, :8] = True                                        # a solid ink cell
    idx = rng.integers(0, 8, (24, 32, 2))
    idx[..., 1] += 8 * (idx[..., 0] % 2)                         # ink bright where paper's index is odd: no, keep legal
    idx[..., 1] = np.where(idx[..., 0] >= 8, idx[..., 1] % 8 + 8, idx[..., 1] % 8)
    path = tmp_path / 'x.scr'
    path.write_bytes(to_scr(bitmap, idx))
    back_bitmap, back_idx = read_scr(path)
    np.testing.assert_array_equal(back_bitmap, bitmap)
    np.testing.assert_array_equal(back_idx & 7, idx & 7)
    assert not has_flash(path)
    s = shown(back_bitmap, back_idx)
    assert s[0, 0, 0] == s[0, 0, 1]
    conv = Converter({'Luma': 1.0, 'Chroma': 1.0}, Mode('zx', (192, 256), (8, 8), ZXPalette()))
    labels = rng.integers(0, len(conv.color_pairs), (24, 32))
    back = pairs_to_labels(conv, np.array(list(conv.palette.iter_idxs_pairs()))[labels])
    np.testing.assert_array_equal(label_pairs(conv, back), label_pairs(conv, labels))
    assert pairs_to_labels(conv, np.array([[[8, 8]]]))[0, 0] == pairs_to_labels(conv, np.array([[[0, 0]]]))[0, 0]
    assert parse_pair('k/Y') == (0, 14) and pair_name(0, 14) == 'k/Y' and pair_name(9, 10) == 'B/R'


def test_ruled_keeps_the_pixels():
    """The picture's pixels sit at [MARGIN:, MARGIN:], enlarged without smoothing; only the grid lines on cell borders
    change, and only by blending; a crop with an origin gets the same numbers as the whole (same ruler pixels)."""
    rng = np.random.default_rng(1)
    img = rng.random((48, 64, 3)).astype(np.float32)
    zoom = 3
    plain = R.up(img, zoom)
    out = R.ruled(img, zoom)
    m = R.MARGIN
    assert out.shape == (48 * zoom + m, 64 * zoom + m, 3)
    pic = out[m:, m:]
    ys = np.arange(1, 6) * 8 * zoom
    xs = np.arange(1, 8) * 8 * zoom
    mask = np.ones(pic.shape[:2], bool)
    mask[ys] = False
    mask[:, xs] = False
    np.testing.assert_array_equal(pic[mask], plain[mask])
    assert (pic[ys].astype(int) >= plain[ys].astype(int)).all()      # blended towards white, never darker
    np.testing.assert_array_equal(R.ruled(img, zoom, grid=False)[m:, m:], plain)
    whole = R.ruled(img, zoom, origin=(0, 0))
    part = R.ruled(R.crop_cells(img, 2, 3, 4, 5), zoom, origin=(2, 3))
    # the column ruler over cells 3..5 of the whole equals the crop's, likewise the row ruler of cells 2..4
    np.testing.assert_array_equal(whole[:m, m + 3 * 8 * zoom:m + 6 * 8 * zoom], part[:m, m:])
    np.testing.assert_array_equal(whole[m + 2 * 8 * zoom:m + 5 * 8 * zoom, :m], part[m:, :m])


def test_sheet_layout():
    """Columns side by side, eye views under them, crops at twice the zoom; marked cells get a red outline."""
    img = np.full((16, 24, 3), 0.5, np.float32)
    m, z = R.MARGIN, 2
    tile_w = 24 * z + m
    assert R.sheet([('a', img), ('b', img)], z).shape[1] == 2 * tile_w + R.GAP
    sheet = R.sheet([('a', img), ('b', img)], z, eye=lambda p: p * 0.5, crops=[(0, 0, 1, 1)], marks={'b': [(1, 2)]})
    assert sheet.shape[1] == 2 * (16 * 2 * z + m) + R.GAP       # the crop row, 2 cells at twice the zoom, is the widest
    top = R.TITLE + m
    a = sheet[top:top + 16 * z, m:m + 24 * z]
    b = sheet[top:top + 16 * z, tile_w + R.GAP + m:tile_w + R.GAP + m + 24 * z]
    assert (a == 128).all(axis=-1).mean() > 0.9 and (b[8 * z, 16 * z] == (255, 0, 0)).all()
    eye_top = top + 16 * z + R.GAP + R.TITLE + m
    assert (sheet[eye_top + 4, m + 4] == 64).all()
