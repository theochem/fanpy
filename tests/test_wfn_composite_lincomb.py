"""Test wfns.wavefunction.composite.lincomb."""
import types

import numpy as np
import pytest
from utils import skip_init
from wfns.wfn.base import BaseWavefunction
from wfns.wfn.composite.lincomb import LinearCombinationWavefunction


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


def test_assign_wfns():
    """Test LinearCombinationWavefunction.assign_wfns."""
    test_wfn = TempWavefunction(4, 10)
    test = skip_init(LinearCombinationWavefunction)
    with pytest.raises(TypeError):
        LinearCombinationWavefunction.assign_wfns(test, (1, test_wfn))
    with pytest.raises(TypeError):
        LinearCombinationWavefunction.assign_wfns(test, (test_wfn, 2))
    test.nelec = 4
    with pytest.raises(ValueError):
        LinearCombinationWavefunction.assign_wfns(test, (test_wfn, TempWavefunction(5, 10)))
    test.dtype = np.float64
    with pytest.raises(ValueError):
        LinearCombinationWavefunction.assign_wfns(
            test, (test_wfn, TempWavefunction(4, 10, dtype=complex))
        )
    test.memory = np.inf
    with pytest.raises(ValueError):
        LinearCombinationWavefunction.assign_wfns(
            test, (test_wfn, TempWavefunction(4, 10, memory="2gb"))
        )
    with pytest.raises(ValueError):
        LinearCombinationWavefunction.assign_wfns(test, (test_wfn,))
    # NOTE: wavefunctions with different numbers of spin orbitals are allowed
    LinearCombinationWavefunction.assign_wfns(test, (test_wfn, TempWavefunction(4, 12)))
    assert test.wfns[0].nelec == 4
    assert test.wfns[0].nspin == 10
    assert test.wfns[1].nelec == 4
    assert test.wfns[1].nspin == 12


def test_spin():
    """Test LinearCombinationWavefunction.spin."""
    test_wfn = TempWavefunction(4, 10)
    test_wfn._spin = 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn,) * 3)
    assert test.spin == 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn, TempWavefunction(4, 10)))
    assert test.spin is None


def test_seniority():
    """Test LinearCombinationWavefunction.seniority."""
    test_wfn = TempWavefunction(4, 10)
    test_wfn._seniority = 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn,) * 3)
    assert test.seniority == 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn, TempWavefunction(4, 10)))
    assert test.seniority is None


def test_template_params():
    """Test LinearCombinationWavefunction.template_params."""
    test_wfn = TempWavefunction(4, 10)
    test = LinearCombinationWavefunction(4, 10, (test_wfn,) * 3)
    assert np.allclose(test.template_params, np.array([1, 0, 0]))
    assert np.allclose(test.template_params, np.array([1, 0, 0]))
    test = LinearCombinationWavefunction(4, 10, (test_wfn, TempWavefunction(4, 10)))
    assert np.allclose(test.template_params, np.array([1, 0]))


# TODO: test deriv functionality
def test_get_overlap():
    """Test LinearCombinationWavefunction.get_overlap.

    Make simple CI wavefunction as a demonstration.

    """
    test_wfn_1 = TempWavefunction(4, 10)
    test_wfn_1.params = np.array([0.9])
    test_wfn_2 = TempWavefunction(4, 10)
    test_wfn_2.params = np.array([0.1])

    def olp_one(self, sd, deriv=None):
        """Overlap of wavefunction 1."""
        if sd == 0b0101:
            return self.params[0]
        else:
            return 0.0

    def olp_two(self, sd, deriv=None):
        """Overlap of wavefunction 2."""
        if sd == 0b1010:
            return self.params[0]
        else:
            return 0.0

    test_wfn_1.get_overlap = types.MethodType(olp_one, test_wfn_1)
    test_wfn_2.get_overlap = types.MethodType(olp_two, test_wfn_2)

    test = LinearCombinationWavefunction(
        4, 10, (test_wfn_1, test_wfn_2), params=np.array([0.7, 0.3])
    )
    assert test.get_overlap(0b0101) == 0.7 * test_wfn_1.params[0] + 0.3 * 0
    assert test.get_overlap(0b1010) == 0.7 * 0 + 0.3 * test_wfn_2.params[0]
