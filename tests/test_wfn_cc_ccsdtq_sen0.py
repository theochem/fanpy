"""Test fanpy.wavefunction.cc.ccsdqt_sen0."""
import pytest
from fanpy.wfn.cc.ccsdtq_sen0 import CCSDTQsen0


class TempCCSDTQsen0(CCSDTQsen0):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_ranks():
    """Test CCSDTQsen0.assign_ranks."""
    test = TempCCSDTQsen0()
    with pytest.raises(ValueError):
        test.assign_ranks([1, 2])
    test.nelec = 3
    with pytest.raises(ValueError):
        test.assign_ranks()
    test.nelec = 4
    test.assign_ranks()
    assert test.ranks == [1, 2, 3, 4]


def test_assign_exops():
    """Test CCSDTsen2Qsen0.assign_exops."""
    test = TempCCSDTQsen0()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_ranks()
    test.assign_exops()
    assert test.exops == [[0, 2], [0, 3], [0, 6], [0, 7],
                          [1, 2], [1, 3], [1, 6], [1, 7],
                          [4, 2], [4, 3], [4, 6], [4, 7],
                          [5, 2], [5, 3], [5, 6], [5, 7],
                          [0, 1, 2, 3], [0, 1, 2, 6], [0, 1, 2, 7],
                          [0, 1, 3, 6], [0, 1, 3, 7], [0, 1, 6, 7],
                          [0, 4, 2, 3], [0, 4, 2, 6], [0, 4, 2, 7],
                          [0, 4, 3, 6], [0, 4, 3, 7], [0, 4, 6, 7],
                          [0, 5, 2, 3], [0, 5, 2, 6], [0, 5, 2, 7],
                          [0, 5, 3, 6], [0, 5, 3, 7], [0, 5, 6, 7],
                          [1, 4, 2, 3], [1, 4, 2, 6], [1, 4, 2, 7],
                          [1, 4, 3, 6], [1, 4, 3, 7], [1, 4, 6, 7],
                          [1, 5, 2, 3], [1, 5, 2, 6], [1, 5, 2, 7],
                          [1, 5, 3, 6], [1, 5, 3, 7], [1, 5, 6, 7],
                          [4, 5, 2, 3], [4, 5, 2, 6], [4, 5, 2, 7],
                          [4, 5, 3, 6], [4, 5, 3, 7], [4, 5, 6, 7],
                          [0, 1, 4, 2, 3, 6], [0, 1, 4, 2, 3, 7],
                          [0, 1, 4, 2, 6, 7], [0, 1, 4, 3, 6, 7],
                          [0, 1, 5, 2, 3, 6], [0, 1, 5, 2, 3, 7],
                          [0, 1, 5, 2, 6, 7], [0, 1, 5, 3, 6, 7],
                          [0, 4, 5, 2, 3, 6], [0, 4, 5, 2, 3, 7],
                          [0, 4, 5, 2, 6, 7], [0, 4, 5, 3, 6, 7],
                          [1, 4, 5, 2, 3, 6], [1, 4, 5, 2, 3, 7],
                          [1, 4, 5, 2, 6, 7], [1, 4, 5, 3, 6, 7],
                          [0, 4, 1, 5, 2, 6, 3, 7]]
