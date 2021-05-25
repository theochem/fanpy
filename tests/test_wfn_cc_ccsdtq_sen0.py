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
    assert test.exops == {(0, 2): 0, (0, 3): 1, (0, 6): 2, (0, 7): 3,
                          (1, 2): 4, (1, 3): 5, (1, 6): 6, (1, 7): 7,
                          (4, 2): 8, (4, 3): 9, (4, 6): 10, (4, 7): 11,
                          (5, 2): 12, (5, 3): 13, (5, 6): 14, (5, 7): 15,
                          (0, 1, 2, 3): 16, (0, 1, 2, 6): 17, (0, 1, 2, 7): 18, (0, 1, 3, 6): 19,
                          (0, 1, 3, 7): 20, (0, 1, 6, 7): 21, (0, 4, 2, 3): 22, (0, 4, 2, 6): 23,
                          (0, 4, 2, 7): 24, (0, 4, 3, 6): 25, (0, 4, 3, 7): 26, (0, 4, 6, 7): 27,
                          (0, 5, 2, 3): 28, (0, 5, 2, 6): 29, (0, 5, 2, 7): 30, (0, 5, 3, 6): 31,
                          (0, 5, 3, 7): 32, (0, 5, 6, 7): 33, (1, 4, 2, 3): 34, (1, 4, 2, 6): 35,
                          (1, 4, 2, 7): 36, (1, 4, 3, 6): 37, (1, 4, 3, 7): 38, (1, 4, 6, 7): 39,
                          (1, 5, 2, 3): 40, (1, 5, 2, 6): 41, (1, 5, 2, 7): 42, (1, 5, 3, 6): 43,
                          (1, 5, 3, 7): 44, (1, 5, 6, 7): 45, (4, 5, 2, 3): 46, (4, 5, 2, 6): 47,
                          (4, 5, 2, 7): 48, (4, 5, 3, 6): 49, (4, 5, 3, 7): 50, (4, 5, 6, 7): 51,
                          (0, 1, 4, 2, 3, 6): 52, (0, 1, 4, 2, 3, 7): 53,
                          (0, 1, 4, 2, 6, 7): 54, (0, 1, 4, 3, 6, 7): 55,
                          (0, 1, 5, 2, 3, 6): 56, (0, 1, 5, 2, 3, 7): 57,
                          (0, 1, 5, 2, 6, 7): 58, (0, 1, 5, 3, 6, 7): 59,
                          (0, 4, 5, 2, 3, 6): 60, (0, 4, 5, 2, 3, 7): 61,
                          (0, 4, 5, 2, 6, 7): 62, (0, 4, 5, 3, 6, 7): 63,
                          (1, 4, 5, 2, 3, 6): 64, (1, 4, 5, 2, 3, 7): 65,
                          (1, 4, 5, 2, 6, 7): 66, (1, 4, 5, 3, 6, 7): 67,
                          (0, 4, 1, 5, 2, 6, 3, 7): 68}
