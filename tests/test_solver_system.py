"""Test wfns.solver.system."""
import pytest
import numpy as np
from wfns.wfn.base import BaseWavefunction
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.objective.schrodinger.system_nonlinear import SystemEquations
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
import wfns.solver.system as system


class TempBaseWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abc structure and overwrite properties and attributes."""

    def __init__(self):
        """Do nothing."""
        pass

    def get_overlap(self, sd, deriv=None):
        """Get overlap between wavefunction and Slater determinant."""
        if sd == 0b0011:
            if deriv is None:
                return (self.params[0] - 3) * (self.params[1] - 2)
            elif deriv == 0:
                return self.params[1] - 2
            elif deriv == 1:
                return self.params[0] - 3
            else:
                return 0
        elif sd == 0b1100:
            if deriv is None:
                return self.params[0] ** 3 + self.params[1] ** 2
            elif deriv == 0:
                return 3 * self.params[0] ** 2
            elif deriv == 1:
                return 2 * self.params[1]
            else:
                return 0
        else:
            return 0

    @property
    def params_shape(self):
        """Return shape of the parameters."""
        return (2,)

    @property
    def template_params(self):
        """Return default parameters."""
        return np.array([0.0, 0.0])


def test_least_squares():
    """Test wfns.solver.least_squares."""
    wfn = TempBaseWavefunction()
    wfn._cache_fns = {}
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_dtype(float)
    wfn.assign_params(np.array([1.0, -1.0]))
    ham = RestrictedChemicalHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    objective = SystemEquations(wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100])

    results = system.least_squares(objective)
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(objective.objective(wfn.params), 0)
    assert np.allclose((wfn.params[0] - 3) ** 2 * (wfn.params[1] - 2) ** 2, 1)

    with pytest.raises(TypeError):
        system.least_squares(OneSidedEnergy(wfn, ham))


def test_root():
    """Test wfns.solver.root."""
    wfn = TempBaseWavefunction()
    wfn._cache_fns = {}
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_dtype(float)
    wfn.assign_params(np.array([1.0, -1.0]))
    ham = RestrictedChemicalHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    objective = SystemEquations(wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100], constraints=[])

    results = system.root(objective)
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(objective.objective(wfn.params), 0)

    with pytest.raises(TypeError):
        system.root(OneSidedEnergy(wfn, ham))
    with pytest.raises(ValueError):
        system.root(SystemEquations(wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100]))
