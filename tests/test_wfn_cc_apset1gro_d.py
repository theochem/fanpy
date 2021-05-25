"""Test fanpy.wavefunction.cc.apset1rog_d."""
import pytest
from fanpy.wfn.cc.apset1rog_d import APset1roGD


class TempAPset1roGD(APset1roGD):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_exops():
    """Test APset1roGD.assign_exops."""
    test = TempAPset1roGD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(ValueError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_exops()
    assert test.exops == {(0, 4, 2, 6): 0, (0, 4, 2, 7): 1, (0, 4, 3, 6): 2, (0, 4, 3, 7): 3,
                          (1, 5, 2, 6): 4, (1, 5, 2, 7): 5, (1, 5, 3, 6): 6, (1, 5, 3, 7): 7}
