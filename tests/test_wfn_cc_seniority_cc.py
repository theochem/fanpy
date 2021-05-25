"""Test fanpy.wavefunction.cc.seniority_cc."""
import numpy as np
import pytest
from fanpy.tools import slater
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.cc.seniority_cc import SeniorityCC


class TempSeniorityCC(SeniorityCC):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_nelec():
    """Test SeniorityCC.assign_nelec."""
    test = TempSeniorityCC()
    test.assign_nelec(4)
    assert test.nelec == 4
    with pytest.raises(TypeError):
        test.assign_nelec(4.0)
    with pytest.raises(ValueError):
        test.assign_nelec(-4)
    with pytest.raises(ValueError):
        test.assign_nelec(5)


def test_assign_refwfn():
    """Test SeniorityCC.assign_refwfn."""
    # check method
    test = TempSeniorityCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_refwfn()
    assert test.refwfn == slater.ground(nocc=test.nelec, norbs=test.nspin)
    ci_test = CIWavefunction(nelec=2, nspin=4, spin=0, seniority=0)
    test.assign_refwfn(ci_test)
    # assert test.refwfn == CIWavefunction(nelec=2, nspin=4, spin=0,
    #                                                      seniority=0)
    assert test.refwfn.nelec == 2
    assert test.refwfn.nspin == 4
    assert test.refwfn.spin == 0
    assert test.refwfn.seniority == 0
    # FIXME: check if sds is correct
    slater_test = 0b0101
    test.assign_refwfn(slater_test)
    assert test.refwfn == 0b0101
    # check errors
    # FIXME: bad tests
    with pytest.raises(TypeError):
        test.assign_refwfn("This is not a gmpy2 or CIwfn object")
    # with pytest.raises(AttributeError):
    #     test.assign_refwfn("This doesn't have a sd_vec attribute")
    with pytest.raises(ValueError):
        test.assign_refwfn(0b1111)
    # with pytest.raises(ValueError):
    #     test.assign_refwfn(0b001001)


def test_get_overlap():
    """Test SeniorityCC.get_overlap."""
    test = TempSeniorityCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_ranks([1])
    test.assign_refwfn()
    test.refresh_exops = None
    test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_params(np.array(range(17)[1:]))
    test.assign_memory("1gb")
    test.load_cache()
    sd = 0b10100011
    np.allclose([test.get_overlap(sd)], [16])
