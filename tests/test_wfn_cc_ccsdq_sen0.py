"""Test fanpy.wavefunction.cc.ccsdq_sen0."""
import pytest
from fanpy.wfn.cc.ccsdq_sen0 import CCSDQsen0


class TempCCSDsen0(CCSDQsen0):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_ranks():
    """Test CCSDQsen0.assign_ranks."""
    test = TempCCSDsen0()
    with pytest.raises(ValueError):
        test.assign_ranks([1, 2])
    test.nelec = 2
    with pytest.raises(ValueError):
        test.assign_ranks()
    test.nelec = 5
    test.assign_ranks()
    assert test.ranks == [1, 2, 4]
