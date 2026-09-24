"""Run: .venv/bin/python tests/test_converter_quality.py (or pytest)."""
import cv2
import numpy as np

from dizher.converter.converter import Converter
from dizher.converter.energy import SEAM_COST
from dizher.converter.dither import ErrorDiffusion, Ordered, Stohastic, duo_levels
from dizher.halftoning.dbs import _Structure, CONTRAST_GAIN
from dizher.halftoning.error_distribution import ed_dither_duo, stucki_duo
from dizher.platforms import Mode
from dizher.platforms.zxspectrum import ZXPalette


def colour_error(converter, image):
    error = converter.opponent(image ** converter.gamma - converter.image_lrgb)
    noise = (converter.luma_noise, converter.chroma_noise, converter.chroma_noise)
    total = 0.0
    for channel, kernel in enumerate(converter.eye_kernels()):
        radius = kernel.shape[0] // 2
        e = error[..., channel]
        blurred = cv2.filter2D(np.pad(e, radius), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        total += float((blurred ** 2).sum() + noise[channel] * (e ** 2).sum())
    return total


def test_equal_luminance_colour_edge():
    converter = Converter({'Luma': 1.0, 'Chroma': 1.0},
                          Mode('one cell', (8, 8), (8, 8), ZXPalette()))
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[:, :4, 1] = 1
    image[:, 4:, 1:] = (0.7152 / (0.7152 + 0.0722)) ** (1 / converter.gamma)
    expected = np.zeros_like(image)
    expected[..., 1] = 1
    expected[:, 4:, 2] = 1
    for halftoner, optimise in ((Stohastic(), False), (Ordered(), False), (ErrorDiffusion(), False), (Stohastic(), True)):
        converter.set_image(image)
        np.testing.assert_allclose(converter.image_luma, 0.7152, atol=1e-7)
        np.testing.assert_array_equal(converter.dither(halftoner, optimise), expected)


def test_selection_matches_full_convolution():
    rng = np.random.default_rng(4)
    mode = Mode('small', (24, 32), (8, 8), ZXPalette(subset='Mono'))
    converter = Converter({'Luma': 0.7, 'Chroma': 0.9}, mode, coherence=0,
                          luma_noise=0.13, chroma_noise=0.07)
    for alpha, scale in ((2.0, 1.4), (0.5, 8.0)):
        converter.luma_alpha = converter.chroma_alpha = alpha
        converter.luma_scale = converter.chroma_scale = scale
        converter.set_image(rng.random((*mode.size, 3), dtype=np.float32))
        labels = rng.integers(len(converter.color_pairs), size=(3, 4))
        rows, cols = np.indices(mode.size)
        image = converter.realized[converter.expand_cells(labels), rows, cols]
        direct = colour_error(converter, image)
        np.testing.assert_allclose(converter.energy.energy(labels), direct, rtol=2e-6)
        assert direct >= 0

    converter.coherence = 1.5   # the coherence cost, seam by seam
    Lh, Lv = converter.energy.seam_smoothness()
    V = converter.pair_dissimilarity
    seams = sum(Lh[r, c] * V[labels[r, c], labels[r + 1, c]] for r in range(2) for c in range(4)) + \
        sum(Lv[r, c] * V[labels[r, c], labels[r, c + 1]] for r in range(3) for c in range(3))
    np.testing.assert_allclose(converter.energy.energy(labels), direct + 1.5 * SEAM_COST * seams, rtol=2e-6)

    for r, c in ((1, 1), (0, 3), (2, 0)):   # the inspector's per-cell scores, inside and on the corners
        costs = converter.energy.cell_candidates(labels, r, c).sum(1)
        totals = []
        for p in range(len(costs)):
            labels[r, c] = p
            totals.append(converter.energy.energy(labels))
        np.testing.assert_allclose(np.array(totals) - costs, totals[0] - costs[0], atol=1e-5)

    converter.luma_noise = converter.chroma_noise = converter.coherence = 0
    converter.set_image(np.full((*mode.size, 3), 0.5 ** (1 / converter.gamma), np.float32))
    converter.dither(Stohastic())
    assert (converter.best_attr_indexes == 1).all(), 'Uniform grey must not become solid block stripes'


def test_dbs_lowers_complete_colour_objective():
    rng = np.random.default_rng(42)
    mode = Mode('small', (16, 24), (8, 8), ZXPalette())
    converter = Converter({'Luma': 0.8, 'Chroma': 1.3}, mode,
                          luma_scale=0.7, chroma_scale=2.5, luma_noise=0.03, chroma_noise=0.2)
    image = rng.random((*mode.size, 3), dtype=np.float32)
    for structure in (0, 0.06):
        converter.structure = structure
        converter.set_image(image)
        seed = converter.dither(Stohastic()).copy()

        def objective(result):
            s = _Structure(converter.opponent(converter.image_lrgb)[..., 0], CONTRAST_GAIN)
            s.set(converter.opponent(result ** converter.gamma)[..., 0])
            return colour_error(converter, result) + structure * float((s.c * (1 - s.ssim)).sum())

        before = objective(seed)
        result = converter.dither(Stohastic(), optimise=True)
        assert objective(result) < before
        np.testing.assert_array_equal(np.where(converter.halftoned[..., None] > 0, converter.best_ink, converter.best_paper),
                                      seed, err_msg='the start must stay for the Unoptimised view')
        assert np.logical_or(np.all(result == converter.best_paper, axis=-1),
                             np.all(result == converter.best_ink, axis=-1)).all()


def test_halftone_target_is_reachable():
    mode = Mode('two cells', (8, 16), (8, 8), ZXPalette(subset='Mono'))
    converter = Converter({'Luma': 1.0, 'Chroma': 1.0}, mode)
    image = np.full((*mode.size, 3), (0.5, 0.2, 0.2), dtype=np.float32)   # red: off the black-white segment
    image[:, 8:] = (0.2, 0.2, 0.5)
    converter.set_image(image)
    converter.dither(Stohastic())
    paper, ink = converter.opponent(converter.best_paper ** converter.gamma), converter.opponent(converter.best_ink ** converter.gamma)
    target = converter.halftone_target(paper, ink)
    np.testing.assert_allclose(target[..., 1:], 0, atol=1e-6)               # chroma the pair cannot paint is dropped
    raw = converter.opponent(converter.image_lrgb)
    np.testing.assert_allclose(duo_levels(target, paper, ink), duo_levels(raw, paper, ink), atol=1e-6)   # same mixture


def test_colour_diffusion_preserves_scalar_projection():
    image = np.full((8, 8), 0.375, dtype=np.float32)
    paper, ink = np.zeros_like(image), np.ones_like(image)
    direction = np.array([2, -1, 0.5], dtype=np.float32)
    expected = stucki_duo(image, paper, ink)
    actual = stucki_duo(image[..., None] * direction, paper[..., None] * direction,
                       ink[..., None] * direction)
    np.testing.assert_array_equal(actual, expected)
    assert 0 < actual.sum() < actual.size


def test_ordered_matrices_cover_tone():
    """Every matrix: no ink at level 0, all ink at level 1, coverage never falls as the level rises, the origin rolls it."""
    from dizher.halftoning.ordered import ordered_dither
    from dizher.halftoning.ordered.matrices import MATRICES
    for name, (m, _) in MATRICES.items():
        coverage = np.array([ordered_dither(np.full(m.shape, level, np.float32), name).mean()
                             for level in np.linspace(0, 1, 65)])   # one whole tile per level
        assert coverage[0] == 0 and coverage[-1] == 1 and (np.diff(coverage) >= 0).all(), name
    flat = np.full((16, 16), 0.5, np.float32)
    assert (ordered_dither(flat, 'Bayer 4x4', (0, 0)) != ordered_dither(flat, 'Bayer 4x4', (1, 0))).any()

def test_error_diffusion_kernels_spread_error():
    """Every kernel: a flat mid grey between black and white comes out about half ink, the lossy ones a bit under."""
    from dizher.halftoning.error_distribution.kernels import KERNELS
    image = np.full((24, 32), 0.5, dtype=np.float32)
    paper, ink = np.zeros_like(image), np.ones_like(image)
    for kernel in KERNELS:
        coverage = ed_dither_duo(image, paper, ink, kernel).mean()
        assert 0.35 < coverage < 0.6, (kernel, coverage)

def test_candidates_follow_the_halftoner():
    """Pair candidates are painted by the converter's ditherer; the batched scalar diffusion matches the colour
    one pair by pair, so 72 candidates cost one raster pass."""
    from dizher.halftoning.error_distribution import ed_dither_duo, ed_dither_levels
    rng = np.random.default_rng(5)
    levels = rng.random((3, 16, 24), dtype=np.float32)
    batched = ed_dither_levels(levels, 'Shiau-Fan 3')
    for l, b in zip(levels, batched):
        np.testing.assert_array_equal(b, ed_dither_duo(l, np.zeros_like(l), np.ones_like(l), 'Shiau-Fan 3'))
    mode = Mode('small', (16, 24), (8, 8), ZXPalette())
    converter = Converter({'Luma': 1.0, 'Chroma': 1.0}, mode, ditherer=ErrorDiffusion('Shiau-Fan 3'))
    converter.set_image(rng.random((*mode.size, 3), dtype=np.float32))
    np.testing.assert_array_equal(converter.bitmaps, ed_dither_levels(converter.levels, 'Shiau-Fan 3'))
    assert (converter.bitmaps != Stohastic().threshold(converter.levels)).any()


def test_stages_preview_snapshots():
    """Live previews are Converter snapshots, so every view can draw the running stage: pair selection sends its
    labels with their blue-noise composite, the halftoner its bitmap; each a copy of what the stage mutates."""
    from dizher import progress as live
    rng = np.random.default_rng(7)
    mode = Mode('small', (16, 24), (8, 8), ZXPalette())
    converter = Converter({'Luma': 1.0, 'Chroma': 1.0}, mode)
    converter.set_image(rng.random((*mode.size, 3), dtype=np.float32))
    shots, report = [], lambda fraction, text: None
    report.preview = shots.append
    interval, live.PROGRESS_INTERVAL = live.PROGRESS_INTERVAL, 0
    try:
        with live.reporting(report):
            converter.dither(Stohastic(), optimise=True)
    finally:
        live.PROGRESS_INTERVAL = interval
    select = [s for s in shots if s.best_attr_indexes is not converter.best_attr_indexes]
    assert select and len(select) < len(shots), 'both stages must send snapshots'
    for s in select:
        np.testing.assert_array_equal(s.dithered_result, s.render_labels(s.best_attr_indexes))
    final = shots[-1]
    assert final.dithered_bitmap is not converter.dithered_bitmap and final.dithered_bitmap.shape == mode.size
    np.testing.assert_array_equal(final.dithered_result, np.where(final.dithered_bitmap[..., None] > 0,
                                                                  converter.best_ink, converter.best_paper))



def test_noise_origin_restarts_both_stages():
    """The noise origin rolls the blue-noise tile under the pair candidates and the DBS start alike."""
    rng = np.random.default_rng(3)
    mode = Mode('small', (16, 24), (8, 8), ZXPalette())
    image = rng.random((*mode.size, 3), dtype=np.float32)
    runs = []
    for origin in ((0, 0), (5, 7)):
        converter = Converter({'Luma': 1.0, 'Chroma': 1.0}, mode, ditherer=Stohastic(origin=origin))
        converter.set_image(image)
        runs.append((converter.bitmaps.copy(), converter.dither(Stohastic(), optimise=True).copy()))
    (b0, r0), (b1, r1) = runs
    assert (b0 != b1).any(), 'candidates must follow the origin'
    np.testing.assert_array_equal(b1, Stohastic(origin=(5, 7)).threshold(converter.levels))
    assert (r0 != r1).any(), 'the DBS start must follow the origin'

if __name__ == '__main__':
    test_equal_luminance_colour_edge()
    test_selection_matches_full_convolution()
    test_dbs_lowers_complete_colour_objective()
    test_halftone_target_is_reachable()
    test_colour_diffusion_preserves_scalar_projection()
    test_candidates_follow_the_halftoner()
    test_ordered_matrices_cover_tone()
    test_error_diffusion_kernels_spread_error()
    test_stages_preview_snapshots()
    test_noise_origin_restarts_both_stages()
    print('ok')
