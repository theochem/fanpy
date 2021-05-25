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
    assert test.exops == {(0, 2): 0, (0, 3): 1, (0, 6): 2, (0, 7): 3,
                          (1, 2): 4, (1, 3): 5, (1, 6): 6, (1, 7): 7,
                          (4, 2): 8, (4, 3): 9, (4, 6): 10, (4, 7): 11,
                          (5, 2): 12, (5, 3): 13, (5, 6): 14, (5, 7): 15,
                          (0, 4, 2, 6): 16, (0, 4, 3, 7): 17,
                          (1, 5, 2, 6): 18, (1, 5, 3, 7): 19}
