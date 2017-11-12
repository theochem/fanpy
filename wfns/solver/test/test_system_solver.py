"""Test wfns.solver.system_solver."""
from nose.tools import assert_raises
import numpy as np
from wfns.wavefunction.base_wavefunction import BaseWavefunction
from wfns.ham.chemical import ChemicalHamiltonian
from wfns.objective.schrodinger.system_nonlinear import SystemEquations
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
import wfns.solver.system_solver as system_solver


class TestBaseWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abc structure and overwrite properties and attributes."""
    def __init__(self):
        pass

    def get_overlap(self, sd, deriv=None):
        if sd == 0b0011:
            if deriv is None:
                return (self.params[0]-3)*(self.params[1]-2)
            elif deriv == 0:
                return (self.params[1]-2)
            elif deriv == 1:
                return (self.params[0]-3)
            else:
                return 0
        elif sd == 0b1100:
            if deriv is None:
                return self.params[0]**3 + self.params[1]**2
            elif deriv == 0:
                return 3*self.params[0]**2
            elif deriv == 1:
                return 2*self.params[1]
            else:
                return 0
        else:
            return 0

    @property
    def template_params(self):
        return 10*(np.random.rand(2) - 0.5)


def test_least_squares():
    """Test wfns.solver.least_squares."""
    wfn = TestBaseWavefunction()
    wfn._cache_fns = {}
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_dtype(float)
    wfn.assign_params()
    ham = ChemicalHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    objective = SystemEquations(wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100])

    results = system_solver.least_squares(objective)
    assert results['success']
    assert np.allclose(results['energy'], 2)
    assert np.allclose(objective.objective(wfn.params), 0)
    assert np.allclose((wfn.params[0] - 3)**2 * (wfn.params[1] - 2)**2, 1)

    assert_raises(TypeError, system_solver.least_squares, OneSidedEnergy(wfn, ham))


def test_root():
    """Test wfns.solver.root."""
    wfn = TestBaseWavefunction()
    wfn._cache_fns = {}
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_dtype(float)
    wfn.assign_params()
    ham = ChemicalHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    objective = SystemEquations(wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100], constraints=[])

    results = system_solver.root(objective)
    assert results['success']
    assert np.allclose(results['energy'], 2)
    assert np.allclose(objective.objective(wfn.params), 0)

    assert_raises(TypeError, system_solver.root, OneSidedEnergy(wfn, ham))
    assert_raises(ValueError, system_solver.root, SystemEquations(wfn, ham, refwfn=0b0011,
                                                                  pspace=[0b0011, 0b1100]))
