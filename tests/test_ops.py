"""Run: .venv/bin/python tests/test_ops.py (or pytest)."""
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from mokit.graph import Memo, evaluate

from dizher import ops
from dizher.converter.dither import Ordered

IMAGE = Path(__file__).parent / 'images' / 'lena.png'


def pipeline(**optimise):
    """Ordered halftone, the optimiser off: fast."""
    graph = ops.make_graph()
    graph = graph.with_params('source', replace(graph['source'].params, path=IMAGE))
    graph = graph.with_params('halftoner', replace(graph['halftoner'].params, halftoner=Ordered.label))
    return graph.with_params('optimise', replace(graph['optimise'].params, enabled=False, **optimise))


def test_pipeline_converts_and_reuses_upstream():
    memo = Memo()
    graph = pipeline()
    result = evaluate(graph, 'optimise', memo)
    assert result.dithered_result.shape == (192, 256, 3)
    prepared = evaluate(graph, 'prepare', memo)
    changed = pipeline(structure=0.2)
    assert changed.key('prepare') == graph.key('prepare') and changed.key('optimise') != graph.key('optimise')
    again = evaluate(changed, 'optimise', memo)
    assert evaluate(changed, 'prepare', memo) is prepared                 # an optimiser edit reuses the setup
    assert again.best_attr_indexes is result.best_attr_indexes            # ...and the pair selection
    assert prepared.best_attr_indexes is None and prepared.dithered_result is None   # stages never mutate upstream


def test_pipeline_matches_single_converter():
    """The staged ops give the one-shot Converter.dither."""
    memo = Memo()
    graph = pipeline()
    result = evaluate(graph, 'optimise', memo)
    direct = evaluate(graph, 'prepare', memo).copy()
    np.testing.assert_array_equal(direct.dither(Ordered('Void dispersed dots')), result.dithered_result)


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
    host.open(IMAGE)
    host.set_params('halftoner', replace(host.graph['halftoner'].params, halftoner=Ordered.label))
    host.set_params('optimise', replace(host.graph['optimise'].params, enabled=False))
    assert settle(host)[-4:] == ['prepare', 'select', 'halftone', 'optimise'] and not host.errors
    host.set_params('optimise', replace(host.graph['optimise'].params, structure=0.2))
    assert settle(host) == ['optimise']
    host.set_params('select', replace(host.graph['select'].params, coherence=1.0))
    assert settle(host) == ['select', 'halftone', 'optimise']
    before = host.result('detail')
    host.set_params('contrast', replace(host.graph['contrast'].params, contrast=30.0))
    assert host.result('detail') is None and host.shown('detail') is before   # the old picture stays up meanwhile
    assert settle(host) == ['contrast', 'color', 'snap', 'detail', 'prepare', 'select', 'halftone', 'optimise']
    assert host.shown('detail') is host.result('detail') is not before
    host.set_params('target', replace(host.graph['target'].params, mode='C64 hires', palette='Bright only'))
    settle(host)
    assert 'target' in host.errors                   # C64 has no bright subset: an error on its block
    host.open(IMAGE.with_name('david.png'))          # another image is a new document
    assert host.shown('detail') is None and host.shown('optimise') is None and not host.errors   # nothing left on screen
    host.close()


def test_host_discards_stale_completions():
    from concurrent.futures import Future
    from dizher.ui.app import Job, Pipeline

    for failed in (False, True):
        host = Pipeline()
        try:
            job = Job('target', host.keys['target'])
            job.future = Future()
            if failed:
                job.future.set_exception(ValueError('old settings'))
            else:
                job.future.set_result('old result')
            host.job = job
            host.set_params('target', replace(host.graph['target'].params, mode='C64 hires'))
            host.update()
            assert not host.errors
            assert host.shown('target') is None
            settle(host)
            assert host.result('target').name == 'C64 hires'
        finally:
            host.close()


def test_framing_geometry():
    from types import SimpleNamespace
    from dizher.platforms import zxspectrum
    mode = zxspectrum.STANDARD
    src = np.random.default_rng(0).uniform(0, 1, (192, 256, 3)).astype(np.float32)
    frame = lambda rgb, **p: ops.framing(SimpleNamespace(rgb=[rgb]), mode, **p)
    np.testing.assert_array_equal(frame(src), src)                                  # screen-sized: a plain copy
    shifted = frame(src, shift_x=3, shift_y=-2)
    np.testing.assert_array_equal(shifted[:-2, 3:], src[2:, :-3])
    assert not shifted[:, :3].any() and not shifted[-2:].any()                       # black where nothing lands
    np.testing.assert_array_equal(frame(src, left=8, right=-8), frame(src, shift_x=-8))   # edges move independently
    np.testing.assert_allclose(frame(src, rotation=180.0), src[::-1, ::-1], atol=1e-5)    # about the centre
    wide = frame(np.ones((64, 512, 3), np.float32), fit='Fit')                       # letterboxed: 256x32 in the middle
    assert wide[80:112].all() and not wide[:80].any() and not wide[112:].any()


def test_tone_semantics():
    from dizher import tone
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.1, 0.9, (8, 8, 3)).astype(np.float32)
    grey = np.full((1, 1, 3), 0.5, np.float32)
    for f in (tone.light, tone.levels, tone.contrast, tone.color, tone.local_tone, tone.detail):
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
    step = np.full((16, 32, 3), 0.3, np.float32)
    step[:, 16:] = 0.7
    sharp = tone.detail(step, sharpen=100.0)[8, :, 0]
    assert sharp[15] < 0.3 < 0.7 < sharp[16] and np.allclose(sharp[:8], 0.3, atol=1e-3)   # overshoot at the edge only
    texture = (0.5 + 0.03 * np.sign(np.sin(np.arange(64) / 2)))[None, :, None].repeat(64, 0).repeat(3, 2).astype(np.float32)
    assert tone.detail(texture, texture=100.0).std() > 1.5 * texture.std() > 1.5 * tone.detail(texture, texture=-100.0).std()
    assert np.allclose(tone.detail(grey, texture=100.0, sharpen=100.0), grey, atol=1e-4)   # flat stays flat
    # local tone: a broad ramp (the lighting) with a fine texture on it
    fine_texture = 0.03 * np.sign(np.sin(np.arange(256) / 2))
    lit = (np.linspace(0.1, 0.9, 256) + fine_texture)[None, :, None].repeat(64, 0).repeat(3, 2).astype(np.float32)
    L = lambda x: tone.rgb2lab(x)[32, :, 0]
    flat = L(tone.local_tone(lit, local_contrast=100.0))
    assert np.ptp(flat) < 0.3 * np.ptp(L(lit))                                     # the broad ramp is taken off...
    fine = lambda x: np.ptp(x[100:112] - np.convolve(x, np.ones(12) / 12, 'same')[100:112])
    assert abs(fine(flat) - fine(L(lit))) < 0.2 * fine(L(lit))                     # ...the texture on it kept
    assert fine(L(tone.local_tone(lit, clarity=100.0))) > 1.5 * fine(L(lit))
    patch = lambda v, **p: tone.local_tone(np.full((4, 4, 3), v, np.float32), **p)[0, 0, 0]   # flat: the base is the pixel
    greys = np.linspace(0, 1, 41)
    for sh in (-100.0, 100.0):
        for hi in (-100.0, 100.0):
            curved = np.array([patch(v, shadows=sh, highlights=hi) for v in greys])
            assert np.all(np.diff(curved) > 0) and np.allclose(curved[[0, -1]], [0, 1], atol=1e-3)   # monotone, ends fixed
    assert patch(0.3, shadows=100.0) > 0.3 and patch(0.7, highlights=-100.0) < 0.7


def test_palette_snap():
    from dizher import snap
    from dizher.platforms import zxspectrum
    palette = zxspectrum.STANDARD.palette
    found = snap.targets(palette)
    assert len(found[0]) == 53 and len(snap.targets(palette, mixes=False)[0]) == 15
    rgb = np.random.default_rng(0).uniform(0.1, 0.9, (8, 8, 3)).astype(np.float32)
    assert snap.snap(rgb, found) is rgb                                   # strength 0: the identity
    red, cyan = palette.as_float()[10], palette.as_float()[5]
    image = np.empty((32, 64, 3), np.float32)
    image[:, :32] = red * 0.92 + 0.03                                     # a surface a little off bright red...
    image[:, 32:] = cyan * 0.9 + 0.05                                     # ...and one off cyan, a step apart
    out, labels = snap.snap(image, found, 1.0, labels_out=(kept := [])), kept[0]
    assert (labels[:, :28] == labels[0, 0]).all() and (labels[:, 36:] == labels[0, -1]).all()   # whole surfaces
    assert np.abs(out[:, :24] - red).max() < 0.02 and np.abs(out[:, 40:] - cyan).max() < 0.02  # on their targets
    ramp = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[None, :, None], 32, 0).repeat(3, 2)
    assert np.abs(snap.snap(ramp, found, 1.0, radius=8.0) - ramp).max() < 0.05   # a steep ramp is no surface
    stripes = image.copy()
    stripes[:, :32:2] += 0.1                                              # texture on the red surface rides along
    moved = snap.snap(stripes, found, 1.0)
    assert (moved[:, 4:20:2, 1] - moved[:, 5:20:2, 1] > 0.05).all()   # (Lab detail: less in sRGB near black)


def test_project_round_trip_and_restore():
    import json
    from mokit import project
    from mokit.graph import Node
    from dizher.ui.app import Pipeline
    host = Pipeline()
    image = Path(__file__).parent / 'images' / 'lena.png'
    host.open(image)
    host.set_params('contrast', replace(host.graph['contrast'].params, contrast=25.0))
    with tempfile.TemporaryDirectory() as tmp:
        project.create_project(Path(tmp) / 'p', host.graph)
        stored = json.loads((Path(tmp) / 'p' / 'project.json').read_text())['nodes']['source']['params']['path']
        assert not Path(stored).is_absolute()   # relative to the folder: the project moves with its images
        loaded = Pipeline()
        loaded.restore(project.load_project(Path(tmp) / 'p').graph)
    assert loaded.graph == host.graph
    # an older project lacks a stage, a newer one has an unknown one: defaults for the first, the second dropped
    older = host.graph.without('metric').with_node('future', Node('dizher.ops:metric', None, ()))
    loaded.restore(older)
    assert loaded.graph['metric'] == ops.make_graph()['metric'] and 'future' not in loaded.graph
    assert loaded.graph['contrast'].params.contrast == 25.0
    host.close(), loaded.close()



def test_undo_redo():
    from dizher.ui.app import Pipeline
    host = Pipeline()
    start = host.graph
    contrast = lambda: host.graph['contrast'].params.contrast
    set_ = lambda v, held=False: host.set_params('contrast', replace(host.graph['contrast'].params, contrast=v), held)
    set_(10.0)
    for v in (20.0, 30.0, 40.0):   # a dragged slider: one step
        set_(v, held=True)
    host.release()
    set_(40.0, held=True)          # a press that changes nothing yet
    set_(50.0, held=True)          # a new drag after the release is a step of its own
    host.release()
    assert len(host.past) == 3
    host.undo()
    assert contrast() == 40.0
    host.undo()
    assert contrast() == 10.0
    host.redo()
    assert contrast() == 40.0
    set_(5.0)                      # an edit drops the redo branch
    assert not host.future and contrast() == 5.0
    for _ in range(5):
        host.undo()
    assert host.graph == start and not host.past
    host.redo()
    host.open(Path(__file__).parent / 'images' / 'lena.png')   # a new document has no history
    assert not host.past and not host.future
    host.close()



def test_project_folder():
    """An image's sidecar project: named as the image, the next free name when another image's project or
    anything else has it."""
    from mokit import project
    from dizher.ui.app import project_folder
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / 'pic.png', Path(tmp) / 'pic.jpg'
        a.write_bytes(IMAGE.read_bytes()), b.write_bytes(IMAGE.read_bytes())
        folder = project_folder(a)
        assert folder == Path(tmp).resolve() / 'pic' and not folder.exists()
        g = ops.make_graph()
        project.create_project(folder, g.with_params('source', replace(g['source'].params, path=a)))
        assert project_folder(a) == folder                          # found again
        assert project_folder(b) == folder.with_name('pic 2')       # a namesake image gets its own
        (Path(tmp) / 'other').mkdir()
        (Path(tmp) / 'other.png').write_bytes(b'')
        assert project_folder(Path(tmp) / 'other.png').name == 'other 2'   # a plain folder is not taken over

if __name__ == '__main__':
    test_framing_geometry()
    test_tone_semantics()
    test_pipeline_converts_and_reuses_upstream()
    test_pipeline_matches_single_converter()
    test_host_reruns_only_downstream_of_an_edit()
    test_host_discards_stale_completions()
    test_project_round_trip_and_restore()
    test_undo_redo()
    test_project_folder()
    print('ok')
