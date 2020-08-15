"""Test fanpy.solver.system."""
import fanpy.solver.system as system
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.wfn.base import BaseWavefunction

import numpy as np

import pytest


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
            return np.array([self.params[1] - 2, self.params[0] - 3])
        elif sd == 0b1100:
            if deriv is None:
                return self.params[0] ** 3 + self.params[1] ** 2
            return np.array([3 * self.params[0] ** 2, 2 * self.params[1]])
        else:
            if deriv is None:
                return 0
            return np.array([0, 0])

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction."""
        if params is None:
            params = np.array([0.0, 0.0])

        super().assign_params(params=params, add_noise=add_noise)


def test_least_squares():
    """Test fanpy.solver.least_squares."""
    wfn = TempBaseWavefunction()
    wfn._cache_fns = {}
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_params(np.array([1.0, -1.0]))
    ham = RestrictedMolecularHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    objective = ProjectedSchrodinger(wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100])

    results = system.least_squares(objective)
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(objective.objective(wfn.params), 0)
    assert np.allclose((wfn.params[0] - 3) ** 2 * (wfn.params[1] - 2) ** 2, 1)

    objective.indices_component_params[ham] = np.arange(ham.nparams)
    results = system.least_squares(objective)
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(objective.objective(objective.active_params), 0)
    assert np.allclose((wfn.params[0] - 3) ** 2 * (wfn.params[1] - 2) ** 2, 1)

    with pytest.raises(TypeError):
        system.least_squares(EnergyOneSideProjection(wfn, ham))


def test_root():
    """Test fanpy.solver.root."""
    wfn = TempBaseWavefunction()
    wfn._cache_fns = {}
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_params(np.array([1.0, -1.0]))
    ham = RestrictedMolecularHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    objective = ProjectedSchrodinger(
        wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100], constraints=[]
    )

    results = system.root(objective)
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(objective.objective(wfn.params), 0)

    objective = ProjectedSchrodinger(
        wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100, 0b0110], constraints=[]
    )
    objective.indices_component_params[ham] = np.arange(ham.nparams)
    results = system.root(objective)
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(objective.objective(objective.active_params), 0)

    with pytest.raises(TypeError):
        system.root(EnergyOneSideProjection(wfn, ham))
    with pytest.raises(ValueError):
        system.root(ProjectedSchrodinger(wfn, ham, refwfn=0b0011, pspace=[0b0011], constraints=[]))
