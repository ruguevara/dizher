"""Run: .venv/bin/python tests/test_converter_cache.py (or pytest)."""
import numpy as np

from dizher.converter.converter import Converter
from dizher.converter.dither import OrderedBayer
from dizher.platforms import Mode
from dizher.platforms.zxspectrum import ZXPalette


TEST_MODE = Mode('test', size=(16, 16), cell=(8, 8), palette=ZXPalette())


def test_set_image_invalidates_cached_conversion():
    converter = Converter({'Luma': 1.0, 'Chroma': 1.0}, TEST_MODE)
    ditherer = OrderedBayer()
    black = np.zeros((*TEST_MODE.size, 3), dtype=np.uint8)
    white = np.full((*TEST_MODE.size, 3), 255, dtype=np.uint8)

    converter.set_image(black, ditherer)
    first = converter.dither(ditherer).copy()
    assert np.all(first == 0)
    converter.set_image(white, ditherer)
    second = converter.dither(ditherer)
    assert np.all(second == 1)
    assert not np.array_equal(first, second)


def test_invalid_image_keeps_last_valid_conversion():
    converter = Converter({'Luma': 1.0, 'Chroma': 1.0}, TEST_MODE)
    ditherer = OrderedBayer()
    image = np.zeros((*TEST_MODE.size, 3), dtype=np.uint8)
    converter.set_image(image, ditherer)
    expected = converter.dither(ditherer).copy()

    try:
        converter.set_image(np.zeros((15, 16, 3), dtype=np.uint8), ditherer)
    except AssertionError:
        pass
    else:
        assert False, 'wrong image size must be rejected'

    assert np.array_equal(expected, converter.dither(ditherer))


if __name__ == '__main__':
    test_set_image_invalidates_cached_conversion()
    test_invalid_image_keeps_last_valid_conversion()
    print('ok')
