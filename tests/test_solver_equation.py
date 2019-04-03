"""Test wfns.solver.equation."""
import os
import pytest
import numpy as np
from wfns.wfn.base import BaseWavefunction
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
from wfns.objective.schrodinger.least_squares import LeastSquaresEquations
from wfns.objective.schrodinger.system_nonlinear import SystemEquations
import wfns.solver.equation as equation


class TempBaseWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abc structure and overwrite properties and attributes."""

    def __init__(self):
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
        return (2,)

    @property
    def template_params(self):
        return 10 * (np.random.rand(2) - 0.5)


def check_cma():
    """Check if cma module is available."""
    try:
        import cma
    except ModuleNotFoundError:
        return False
    else:
        return True


@pytest.mark.skipif(not check_cma(), "The module `cma` is unavailable.")
def test_cma():
    """Test wnfs.solver.equation.cma."""
    wfn = TempBaseWavefunction()
    wfn._cache_fns = {}
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_dtype(float)
    wfn.assign_params()
    ham = RestrictedChemicalHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))

    results = equation.cma(OneSidedEnergy(wfn, ham, refwfn=[0b0011, 0b1100]))
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
        equation.cma(SystemEquations(wfn, ham, refwfn=0b0011))
    with pytest.raises(ValueError):
        equation.cma(OneSidedEnergy(wfn, ham, param_selection=[[wfn, np.array([0])]]))

    results = equation.cma(OneSidedEnergy(wfn, ham, refwfn=[0b0011, 0b1100]), save_file="temp.npy")
    test = np.load("temp.npy")
    assert np.allclose(results["params"], test)
    os.remove("temp.npy")


def test_minimize():
    """Test wnfs.solver.equation.minimize."""
    wfn = TempBaseWavefunction()
    wfn._cache_fns = {}
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_dtype(float)
    wfn.assign_params()
    ham = RestrictedChemicalHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))

    results = equation.minimize(OneSidedEnergy(wfn, ham, refwfn=[0b0011, 0b1100]))
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
        equation.minimize(SystemEquations(wfn, ham, refwfn=0b0011))
