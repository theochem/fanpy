"""Test fanpy.solver.equation."""
import os

import numpy as np
import pytest
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.eqn.least_squares import LeastSquaresEquations
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
import fanpy.solver.equation as equation
from fanpy.wfn.base import BaseWavefunction


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
            return 0

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction."""
        if params is None:
            params = 10 * (np.random.rand(2) - 0.5)

        super().assign_params(params=params, add_noise=add_noise)


def test_cma(tmp_path):
    """Test fanpy.solver.equation.cma."""
    pytest.importorskip("cma")
    wfn = TempBaseWavefunction()
    wfn._cache_fns = {}
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_params()
    ham = RestrictedMolecularHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))

    results = equation.cma(EnergyOneSideProjection(wfn, ham, refwfn=[0b0011, 0b1100]))
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 2)
    assert results["message"] == "Following termination conditions are satisfied: tolfun: 1e-11."

    results = equation.cma(LeastSquaresEquations(wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100]))
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 0, atol=1e-7)
    assert results["message"] in [
        "Following termination conditions are satisfied: tolfun: 1e-11.",
        "Following termination conditions are satisfied: tolfun: 1e-11, tolfunhist: 1e-12.",
    ]

    with pytest.raises(TypeError):
        equation.cma(lambda x, y: (x - 3) * (y - 2) + x ** 3 + y ** 2)
    with pytest.raises(ValueError):
        equation.cma(ProjectedSchrodinger(wfn, ham, refwfn=0b0011))
    with pytest.raises(ValueError):
        equation.cma(EnergyOneSideProjection(wfn, ham, param_selection=[[wfn, np.array([0])]]))

    path = str(tmp_path / "temp.npy")
    results = equation.cma(
        EnergyOneSideProjection(wfn, ham, refwfn=[0b0011, 0b1100], tmpfile=path)
    )
    path = str(tmp_path / "temp_TempBaseWavefunction.npy")
    test = np.load(path)
    assert np.allclose(results["params"], test)
    os.remove(path)


def test_minimize():
    """Test wnfs.solver.equation.minimize."""
    wfn = TempBaseWavefunction()
    wfn._cache_fns = {}
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_params()
    ham = RestrictedMolecularHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))

    results = equation.minimize(EnergyOneSideProjection(wfn, ham, refwfn=[0b0011, 0b1100]))
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 2)

    results = equation.minimize(
        LeastSquaresEquations(wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100])
    )
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 0, atol=1e-7)

    with pytest.raises(TypeError):
        equation.minimize(lambda x, y: (x - 3) * (y - 2) + x ** 3 + y ** 2)
    with pytest.raises(ValueError):
        equation.minimize(ProjectedSchrodinger(wfn, ham, refwfn=0b0011))
