"""Test wfn.wavefunction.composite.base_one."""
import numpy as np
import pytest
from utils import disable_abstract, skip_init
from wfns.wfn.base import BaseWavefunction
from wfns.wfn.composite.base_one import BaseCompositeOneWavefunction


class TempWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abstract class structure."""

    _spin = None
    _seniority = None

    def get_overlap(self):
        """Do nothing."""
        pass

    @property
    def spin(self):
        """Return the spin of the wavefunction."""
        return self._spin

    @property
    def seniority(self):
        """Return the seniority of the wavefunction."""
        return self._seniority

    @property
    def params_shape(self):
        """Return the shape of the parameters."""
        return (10, 10)

    @property
    def template_params(self):
        """Return the default parameters."""
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
