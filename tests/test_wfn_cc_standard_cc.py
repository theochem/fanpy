"""Test fanpy.wavefunction.cc.standard_cc."""
import pytest
from fanpy.tools import slater
from fanpy.wfn.cc.standard_cc import StandardCC


class TempStandardCC(StandardCC):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_exops():
    """Test StandardCC.assign_exops."""
    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_ranks([2])
    test.assign_exops()
    assert test.exops == [[0, 1, 2, 3], [0, 1, 2, 6], [0, 1, 2, 7],
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
                          [4, 5, 3, 6], [4, 5, 3, 7], [4, 5, 6, 7]]


def test_assign_refwfn():
    """Test StandardCC.assign_refwfn."""
    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    with pytest.raises(TypeError):
        test.assign_refwfn("This is not a gmpy2 instance")
    with pytest.raises(ValueError):
        test.assign_refwfn(0b00010001)
    test.assign_refwfn()
    assert test.refwfn == 0b00110011
