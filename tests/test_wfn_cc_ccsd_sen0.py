"""Test fanpy.wavefunction.cc.ccsd_sen0."""
import pytest
from fanpy.wfn.cc.ccsd_sen0 import CCSDsen0


class TempCCSDsen0(CCSDsen0):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_ranks():
    """Test CCSDsen0.assign_ranks."""
    test = TempCCSDsen0()
    with pytest.raises(ValueError):
        test.assign_ranks([1, 2])
    test.nelec = 1
    with pytest.raises(ValueError):
        test.assign_ranks()
    test.nelec = 2
    test.assign_ranks()
    assert test.ranks == [1, 2]


def test_assign_exops():
    """Test CCSDsen0.assign_exops."""
    test = TempCCSDsen0()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_exops()
    assert test.exops == [[0, 2], [0, 3], [0, 6], [0, 7],
                          [1, 2], [1, 3], [1, 6], [1, 7],
                          [4, 2], [4, 3], [4, 6], [4, 7],
                          [5, 2], [5, 3], [5, 6], [5, 7],
                          [0, 4, 2, 6], [0, 4, 3, 7],
                          [1, 5, 2, 6], [1, 5, 3, 7]]
