"""Test fanpy.wavefunction.cc.generalized_cc."""
import pytest
from fanpy.wfn.cc.generalized_cc import GeneralizedCC


class TempGeneralizedCC(GeneralizedCC):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_exops():
    """Test GeneralizedCC.assign_exops."""
    test = TempGeneralizedCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_refwfn()
    test.assign_ranks()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 2], [1, 3]])
    test.assign_exops()
    # FIXME: default exops changed
    assert test.exops == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
                          [0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2],
                          [1, 2, 0, 3], [1, 3, 0, 2], [2, 3, 0, 1]]
