"""Test fanpy.wavefunction.cc.apset1rog_sd."""
import pytest
from fanpy.wfn.cc.apset1rog_sd import APset1roGSD


class TempAPset1GroSD(APset1roGSD):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_ranks():
    """Test APset1roGSD.assign_ranks."""
    test = TempAPset1GroSD()
    with pytest.raises(TypeError):
        test.assign_ranks([1, 2])
    test.assign_ranks()
    assert test.ranks == [1, 2]


def test_assign_exops():
    """Test APset1roGSD.assign_exops."""
    test = TempAPset1GroSD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(ValueError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_exops()
    assert test.exops == {(0, 4, 2, 4): 0, (0, 4, 3, 4): 1,
                          (1, 5, 2, 5): 2, (1, 5, 3, 5): 3,
                          (0, 4, 0, 6): 4, (0, 4, 0, 7): 5,
                          (1, 5, 1, 6): 6, (1, 5, 1, 7): 7,
                          (0, 4, 2, 6): 8, (0, 4, 2, 7): 9, (0, 4, 3, 6): 10, (0, 4, 3, 7): 11,
                          (1, 5, 2, 6): 12, (1, 5, 2, 7): 13, (1, 5, 3, 6): 14, (1, 5, 3, 7): 15}


def test_generate_possible_exops():
    """Test APset1GroSD.generate_possible_exops."""
    test = TempAPset1GroSD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_ranks()
    test.assign_refwfn()
    test.assign_exops()
    test.generate_possible_exops([0, 1, 4], [2, 3, 6])
    assert test.exop_combinations[(0, 1, 4, 2, 3, 6)] == [[[0, 4, 2, 4], [1, 5, 3, 5], [0, 4, 0,
                                                                                        6]],
                                                          [[0, 4, 3, 4], [1, 5, 2, 5], [0, 4, 0,
                                                                                        6]],
                                                          [[1, 5, 2, 5], [0, 4, 3, 6]],
                                                          [[1, 5, 3, 5], [0, 4, 2, 6]]]
