"""The float sliders' snap: a round step, at most 1/500 of the range."""
from mokit.ui.params import slider_step


def test_slider_step():
    for (lo, hi), want in {(0, 1): (0.002, 3), (-100, 100): (0.2, 1), (0.02, 0.4): (0.0005, 4), (0, 300): (0.5, 1),
                           (0, 8): (0.01, 2)}.items():
        assert slider_step(lo, hi) == want, (lo, hi)
    step, _ = slider_step(0, 1)
    for v in (0.5, 0.25, 0.1, 0.05):
        assert abs(v / step - round(v / step)) < 1e-9, v
