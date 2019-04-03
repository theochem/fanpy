"""Test wfn.wavefunction.composite.base_one."""
import pytest
import numpy as np
from wfns.wfn.base import BaseWavefunction
from wfns.wfn.composite.base_one import BaseCompositeOneWavefunction
from utils import skip_init, disable_abstract


class TempWavefunction(BaseWavefunction):
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
    def params_shape(self):
        return (10, 10)

    @property
    def template_params(self):
        return np.identity(10)


def test_assign_wfn():
    """Test BaseCompositeOneWavefunction.assign_wfn."""
    test = skip_init(disable_abstract(BaseCompositeOneWavefunction))
    with pytest.raises(TypeError):
        BaseCompositeOneWavefunction.assign_wfn(test, 1)
    with pytest.raises(TypeError):
        BaseCompositeOneWavefunction.assign_wfn(test, (TempWavefunction(4, 10),))
    test.nelec = 4
    with pytest.raises(ValueError):
        BaseCompositeOneWavefunction.assign_wfn(test, TempWavefunction(5, 10))
    test.dtype = np.float64
    with pytest.raises(ValueError):
        BaseCompositeOneWavefunction.assign_wfn(test, TempWavefunction(4, 10, dtype=complex))
    test.memory = np.inf
    with pytest.raises(ValueError):
        BaseCompositeOneWavefunction.assign_wfn(test, TempWavefunction(4, 10, memory="2gb"))
    BaseCompositeOneWavefunction.assign_wfn(test, TempWavefunction(4, 10))
    assert test.wfn.nelec == 4
    assert test.wfn.nspin == 10
