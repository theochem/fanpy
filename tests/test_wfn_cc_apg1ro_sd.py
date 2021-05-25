"""Test fanpy.wavefunction.cc.apg1ro_sd."""
import pytest
from fanpy.wfn.cc.apg1ro_sd import APG1roSD


class TempAPG1roSD(APG1roSD):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_ranks():
    """Test APG1roGSD.assign_ranks."""
    test = TempAPG1roSD()
    with pytest.raises(TypeError):
        test.assign_ranks([1, 2])
    test.assign_ranks()
    assert test.ranks == [1, 2]


def test_assign_exops():
    """Test APG1roGSD.assign_exops."""
    test = TempAPG1roSD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    test.assign_ranks()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_exops()
    assert test.exops == {(0, 4, 2, 4): 0, (0, 4, 3, 4): 1, (0, 4, 6, 4): 2, (0, 4, 7, 4): 3,
                          (1, 5, 2, 5): 4, (1, 5, 3, 5): 5, (1, 5, 6, 5): 6, (1, 5, 7, 5): 7,
                          (0, 4, 0, 2): 8, (0, 4, 0, 3): 9, (0, 4, 0, 6): 10, (0, 4, 0, 7): 11,
                          (1, 5, 1, 2): 12, (1, 5, 1, 3): 13, (1, 5, 1, 6): 14, (1, 5, 1, 7): 15,
                          (0, 4, 2, 3): 16, (0, 4, 2, 6): 17, (0, 4, 2, 7): 18, (0, 4, 3, 6): 19,
                          (0, 4, 3, 7): 20, (0, 4, 6, 7): 21, (1, 5, 2, 3): 22, (1, 5, 2, 6): 23,
                          (1, 5, 2, 7): 24, (1, 5, 3, 6): 25, (1, 5, 3, 7): 26, (1, 5, 6, 7): 27}


def test_generate_possible_exops():
    """Test APG1roSD.generate_possible_exops."""
    test = TempAPG1roSD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    test.assign_ranks()
    test.assign_exops()
    test.generate_possible_exops([0, 1, 4], [2, 3, 6])
    assert test.exop_combinations[(0, 1, 4, 2, 3, 6)] == [[[0, 4, 2, 4], [1, 5, 3, 5], [0, 4, 0, 6]],
                                                          [[0, 4, 2, 4], [1, 5, 6, 5], [0, 4, 0, 3]],
                                                          [[0, 4, 3, 4], [1, 5, 2, 5], [0, 4, 0, 6]],
                                                          [[0, 4, 3, 4], [1, 5, 6, 5], [0, 4, 0, 2]],
                                                          [[0, 4, 6, 4], [1, 5, 2, 5], [0, 4, 0, 3]],
                                                          [[0, 4, 6, 4], [1, 5, 3, 5], [0, 4, 0, 2]],
                                                          [[1, 5, 2, 5], [0, 4, 3, 6]],
                                                          [[1, 5, 3, 5], [0, 4, 2, 6]],
                                                          [[1, 5, 6, 5], [0, 4, 2, 3]]]
