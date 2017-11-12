"""Test wfn.wavefunction.composite.base_composite_one."""
from nose.tools import assert_raises
import numpy as np
from wfns.wfn.base_wavefunction import BaseWavefunction
from wfns.wfn.composite.base_composite_one import BaseCompositeOneWavefunction


class Container:
    pass


class TestWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abstract class structure."""
    _spin = None
    _seniority = None

    def get_overlap(self):
        pass

    @property
    def spin(self):
        return self._spin

    @property
    def seniority(self):
        return self._seniority

    @property
    def template_params(self):
        return np.identity(10)


def test_assign_wfn():
    """Test BaseCompositeOneWavefunction.assign_wfn."""
    test = Container()
    assert_raises(TypeError, BaseCompositeOneWavefunction.assign_wfn, test, 1)
    assert_raises(TypeError, BaseCompositeOneWavefunction.assign_wfn, test,
                  (TestWavefunction(4, 10), ))
    test.nelec = 4
    assert_raises(ValueError, BaseCompositeOneWavefunction.assign_wfn, test,
                  TestWavefunction(5, 10))
    test.dtype = np.float64
    assert_raises(ValueError, BaseCompositeOneWavefunction.assign_wfn, test,
                  TestWavefunction(4, 10, dtype=complex))
    test.memory = np.inf
    assert_raises(ValueError, BaseCompositeOneWavefunction.assign_wfn, test,
                  TestWavefunction(4, 10, memory='2gb'))
    BaseCompositeOneWavefunction.assign_wfn(test, TestWavefunction(4, 10))
    assert test.wfn.nelec == 4
    assert test.wfn.nspin == 10
