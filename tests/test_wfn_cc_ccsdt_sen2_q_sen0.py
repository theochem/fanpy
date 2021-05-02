"""Test fanpy.wavefunction.cc.ccsdt_sen2_q_sen0."""
import pytest
from fanpy.wfn.cc.ccsdt_sen2_q_sen0 import CCSDTsen2Qsen0


class TempCCSDTsen2Qsen0(CCSDTsen2Qsen0):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_exops():
    """Test CCSDTsen2Qsen0.assign_exops."""
    test = TempCCSDTsen2Qsen0()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
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
                          [0, 4, 1, 2, 6, 3], [0, 4, 1, 2, 6, 7], [0, 4, 1, 3, 7, 2], [0, 4, 1,
                                                                                       3, 7, 6],
                          [0, 4, 5, 2, 6, 3], [0, 4, 5, 2, 6, 7], [0, 4, 5, 3, 7, 2], [0, 4, 5,
                                                                                       3, 7, 6],
                          [1, 5, 0, 2, 6, 3], [1, 5, 0, 2, 6, 7], [1, 5, 0, 3, 7, 2], [1, 5, 0,
                                                                                       3, 7, 6],
                          [1, 5, 4, 2, 6, 3], [1, 5, 4, 2, 6, 7], [1, 5, 4, 3, 7, 2], [1, 5, 4,
                                                                                       3, 7, 6],

                          [0, 4, 1, 5, 2, 6, 3, 7]]
