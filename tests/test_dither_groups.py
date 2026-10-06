"""Run: .venv/bin/python tests/test_dither_groups.py (or pytest)."""
from dizher.halftoning.error_distribution.kernels import GROUPS as KERNEL_GROUPS, KERNELS
from dizher.halftoning.ordered.matrices import GROUPS as MATRIX_GROUPS, MATRICES


def grouped(groups):
    return [name for names in groups.values() for name in names]


def test_every_matrix_is_in_exactly_one_group():
    names = grouped(MATRIX_GROUPS)
    assert len(names) == len(set(names))
    assert set(names) == set(MATRICES)


def test_every_kernel_is_in_exactly_one_group():
    names = grouped(KERNEL_GROUPS)
    assert len(names) == len(set(names))
    assert set(names) == set(KERNELS)
