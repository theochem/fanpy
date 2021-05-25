"""Test fanpy.wavefunction.cc.pccd_ap1rog."""
import pytest
from fanpy.tools import slater
from fanpy.wfn.cc.pccd_ap1rog import PCCD


class TempPCCD(PCCD):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_nelec():
    """Test PCCD.assign_nelec."""
    test = TempPCCD()
    test.assign_nelec(4)
    assert test.nelec == 4
    with pytest.raises(TypeError):
        test.assign_nelec(4.0)
    with pytest.raises(ValueError):
        test.assign_nelec(-4)
    with pytest.raises(ValueError):
        test.assign_nelec(5)


def test_assign_ranks():
    """Test PCCD.assign_ranks."""
    test = TempPCCD()
    with pytest.raises(ValueError):
        test.assign_ranks([1, 2])
    test.assign_nelec(2)
    with pytest.raises(ValueError):
        test.assign_ranks([3])
    test.assign_nelec(4)
    test.assign_ranks()
    assert test.ranks == [2]


def test_assign_exops():
    """Test PCCD.assign_exops."""
    test = TempPCCD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_exops()
    assert test.exops == {(0, 4, 2, 6): 0, (0, 4, 3, 7): 1, (1, 5, 2, 6): 2, (1, 5, 3, 7): 3}


def test_assign_refwfn():
    """Test PCCD.assign_refwfn."""
    test = TempPCCD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    with pytest.raises(TypeError):
        test.assign_refwfn("This is not a gmpy2 instance")
    with pytest.raises(ValueError):
        test.assign_refwfn(0b00010001)
    with pytest.raises(ValueError):
        test.assign_refwfn(0b0001100011)
    with pytest.raises(ValueError):
        test.assign_refwfn(0b11000011)
    test.assign_refwfn()
    assert test.refwfn == (0b00110011)
