"""Run: .venv/bin/python tests/test_ops.py (or pytest)."""
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from mokit.graph import Memo, evaluate

from dizher import ops
from dizher.converter.dither import OrderedBayer

IMAGE = Path(__file__).parent / 'images' / 'lena.png'


def pipeline(**halftone):
    graph = ops.make_graph()
    graph = graph.with_params('source', replace(graph['source'].params, path=IMAGE))
    return graph.with_params('halftone', replace(graph['halftone'].params, halftoner=OrderedBayer.label, **halftone))


def test_pipeline_converts_and_reuses_upstream():
    memo = Memo()
    graph = pipeline()
    result = evaluate(graph, 'halftone', memo)
    assert result.dithered_result.shape == (192, 256, 3)
    prepared = evaluate(graph, 'prepare', memo)
    changed = pipeline(structure=0.2)
    assert changed.key('prepare') == graph.key('prepare') and changed.key('halftone') != graph.key('halftone')
    again = evaluate(changed, 'halftone', memo)
    assert evaluate(changed, 'prepare', memo) is prepared                 # a halftone edit reuses the setup
    assert again.best_attr_indexes is result.best_attr_indexes            # ...and the pair selection
    assert prepared.best_attr_indexes is None and prepared.dithered_result is None   # stages never mutate upstream


def test_pipeline_matches_single_converter():
    """The staged ops give the one-shot Converter.dither."""
    memo = Memo()
    graph = pipeline()
    result = evaluate(graph, 'halftone', memo)
    direct = evaluate(graph, 'prepare', memo).copy()
    np.testing.assert_array_equal(direct.dither(OrderedBayer()), result.dithered_result)


def settle(host):
    """Drive the host like the UI's frame loop; the stages it started, in order."""
    started = []
    host._deadline = 0.0
    while True:
        host.update()
        if host.job is None:
            return started
        if not started or started[-1] != host.job.node_id:
            started.append(host.job.node_id)
        host.job.future.exception()


def test_host_reruns_only_downstream_of_an_edit():
    from dizher.ui.app import Pipeline
    host = Pipeline()
    host.set_params('halftone', replace(host.graph['halftone'].params, halftoner=OrderedBayer.label))
    host.open(IMAGE)
    assert settle(host)[-3:] == ['prepare', 'select', 'halftone'] and not host.errors
    host.set_params('halftone', replace(host.graph['halftone'].params, structure=0.2))
    assert settle(host) == ['halftone']
    host.set_params('select', replace(host.graph['select'].params, coherence=1.0))
    assert settle(host) == ['select', 'halftone']
    before = host.result('color')
    host.set_params('contrast', replace(host.graph['contrast'].params, contrast=30.0))
    assert host.result('color') is None and host.shown('color') is before   # the old picture stays up meanwhile
    assert settle(host) == ['contrast', 'color', 'prepare', 'select', 'halftone']
    assert host.shown('color') is host.result('color') is not before
    host.set_params('target', replace(host.graph['target'].params, mode='C64 hires', palette='Bright only'))
    settle(host)
    assert 'target' in host.errors                   # C64 has no bright subset: an error on its block
    host.close()


def test_tone_semantics():
    from dizher import tone
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.1, 0.9, (8, 8, 3)).astype(np.float32)
    grey = np.full((1, 1, 3), 0.5, np.float32)
    for f in (tone.light, tone.levels, tone.contrast, tone.color):
        assert f(rgb) is rgb                                          # neutral settings are the identity
    assert np.allclose(tone.light(grey, exposure=1.0), 2 ** (1 / 2.2) * 0.5, atol=1e-6)   # +1 stop in linear light
    warm = tone.light(grey, temperature=50.0)[0, 0]
    assert warm[0] > 0.5 > warm[2] and abs((warm ** 2.2) @ tone.LUMA - 0.5 ** 2.2) < 1e-6   # warmer, same luminance
    assert np.allclose(tone.levels(np.array([[[64, 128, 192]]], np.float32) / 255, 64, 192), [[[0, 0.5, 1]]], atol=1e-6)
    assert tone.levels(grey, gamma=2.0)[0, 0, 0] > 0.5                 # gamma > 1 brightens, as in Photoshop
    assert tone.auto_levels(rgb * 0.5 + 0.25, clip=0) == (int(np.floor(rgb.min() * 127.5 + 63.75)),
                                                          int(np.ceil(rgb.max() * 127.5 + 63.75)))
    ramp = np.repeat(np.linspace(0, 1, 11, dtype=np.float32)[:, None, None], 3, axis=2)   # (11, 1, 3) grey ramp
    more, less = tone.contrast(ramp, 50.0)[:, 0, 0], tone.contrast(ramp, -50.0)[:, 0, 0]
    assert np.all(np.diff(more) >= 0) and np.all(np.diff(less) >= 0)   # monotone both ways
    assert more[2] < ramp[2, 0, 0] and more[8] > ramp[8, 0, 0] and less[2] > ramp[2, 0, 0]
    assert np.allclose(tone.color(rgb, saturation=-100.0), tone.color(rgb, saturation=-100.0)[..., :1], atol=0.02)   # greyed out


if __name__ == '__main__':
    test_tone_semantics()
    test_pipeline_converts_and_reuses_upstream()
    test_pipeline_matches_single_converter()
    test_host_reruns_only_downstream_of_an_edit()
    print('ok')
