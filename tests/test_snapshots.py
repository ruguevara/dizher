"""Snapshot storage (src/dizher/ui/snapshots.py). Run: pytest tests/test_snapshots.py"""
import os
from dataclasses import replace

import numpy as np
import pytest

from dizher import ops
from dizher.ui import snapshots


def graph_with(folder):
    graph = ops.make_graph()
    graph = graph.with_params('source', replace(graph['source'].params, path=folder / 'img' / 'a.png'))
    graph = graph.with_params('optimise', replace(graph['optimise'].params, enabled=False))
    return graph.with_params('overpaint', replace(graph['overpaint'].params, overrides=((0, 0, 1, 2),)))


def test_roundtrip(tmp_path):
    graph = graph_with(tmp_path)
    snapshots.save(tmp_path, 'тест', graph)
    assert snapshots.load(tmp_path, 'тест') == graph
    assert '"img' in (tmp_path / 'snapshots' / 'тест.json').read_text()   # stored relative to the project
    assert snapshots.picture(tmp_path, 'тест') is None


def test_names_newest_first(tmp_path):
    for name, t in (('b', 10), ('a', 10), ('old', 1), ('new', 20)):
        snapshots.save(tmp_path, name, ops.make_graph())
        os.utime(tmp_path / 'snapshots' / f'{name}.json', (t, t))
    assert snapshots.names(tmp_path) == ['new', 'a', 'b', 'old']
    assert snapshots.names(tmp_path / 'none') == []


def test_picture(tmp_path):
    pic = np.zeros((4, 4, 3))
    pic[:2, :, 0] = 1   # red top, black bottom
    pic[2:, 1, 2] = 1   # one blue column below
    snapshots.save(tmp_path, 'тест', ops.make_graph(), pic)
    assert np.array_equal(snapshots.picture(tmp_path, 'тест'), pic.astype(np.float32))
    snapshots.save(tmp_path, 'тест', ops.make_graph())   # no picture given: the old one stays
    assert snapshots.picture(tmp_path, 'тест') is not None


def test_rename_delete(tmp_path):
    snapshots.save(tmp_path, 'a', ops.make_graph(), np.ones((2, 2, 3)))
    snapshots.save(tmp_path, 'b', ops.make_graph())
    with pytest.raises(FileExistsError):
        snapshots.rename(tmp_path, 'a', 'b')
    snapshots.rename(tmp_path, 'a', 'тест')
    assert sorted(p.name for p in (tmp_path / 'snapshots').iterdir()) == ['b.json', 'тест.json', 'тест.png']
    snapshots.delete(tmp_path, 'тест')
    snapshots.delete(tmp_path, 'b')
    assert not list((tmp_path / 'snapshots').iterdir())


def test_clean():
    assert snapshots.clean('a/b:c') == 'a-b-c'
    assert snapshots.clean('  .x. ') == 'x'
    assert snapshots.clean('...') == ''
    assert snapshots.clean('тест 1') == 'тест 1'


def test_free_name(tmp_path):
    assert snapshots.free_name(tmp_path, 'тест') == 'тест'
    snapshots.save(tmp_path, 'тест', ops.make_graph())
    snapshots.save(tmp_path, 'тест 2', ops.make_graph())
    assert snapshots.free_name(tmp_path, 'тест') == 'тест 3'
