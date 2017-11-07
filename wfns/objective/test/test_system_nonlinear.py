"""Test wfns.objective.system_nonlinear."""
from nose.tools import assert_raises
import numpy as np
from wfns.param import ParamContainer
from wfns.objective.system_nonlinear import SystemEquations
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


class TestSystemEquations(SystemEquations):
    def __init__(self):
        pass


def test_system_init_energy():
    """Test energy initialization in SystemEquations.__init__."""
    wfn = CIWavefunction(2, 4)
    ham = ChemicalHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))

    test = SystemEquations(wfn, ham, energy=None, energy_type='compute')
    assert isinstance(test.energy, ParamContainer)
    assert test.energy.params == test.get_energy_one_proj(0b0101)

    test = SystemEquations(wfn, ham, energy=2.0, energy_type='compute')
    assert test.energy.params == 2.0

    test = SystemEquations(wfn, ham, energy=np.complex128(2.0), energy_type='compute')
    assert test.energy.params == 2.0

    assert_raises(TypeError, SystemEquations, wfn, ham, energy=0, energy_type='compute')
    assert_raises(TypeError, SystemEquations, wfn, ham, energy='1', energy_type='compute')

    assert_raises(ValueError, SystemEquations, wfn, ham, energy=None, energy_type='something else')
    assert_raises(ValueError, SystemEquations, wfn, ham, energy=None, energy_type=0)

    test = SystemEquations(wfn, ham, energy=0.0, energy_type='variable')
    assert np.allclose(test.param_selection._masks_container_params[test.energy],
                       np.array([0]))
    assert np.allclose(test.param_selection._masks_objective_params[test.energy],
                       np.array([False, False, False, False, False, False, True]))

    test = SystemEquations(wfn, ham, energy=0.0, energy_type='fixed')
    assert np.allclose(test.param_selection._masks_container_params[test.energy],
                       np.array([]))
    assert np.allclose(test.param_selection._masks_objective_params[test.energy],
                       np.array([False, False, False, False, False, False]))


def test_system_nproj():
    """Test SystemEquation.nproj"""
    test = TestSystemEquations()
    test.pspace = [0b0101, 0b1010]
    assert test.nproj == 2
    test.pspace = [0b0101, 0b1010, 0b0110]
    assert test.nproj == 3


def test_system_assign_pspace():
    """Test SystemEquations.assign_pspace."""
    test = TestSystemEquations()
    test.wfn = CIWavefunction(2, 4)

    test.assign_pspace()
    for sd, sol_sd in zip(test.pspace, [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]):
        assert sd == sol_sd

    test.assign_pspace([0b0101, 0b1010])
    for sd, sol_sd in zip(test.pspace, [0b0101, 0b1010]):
        assert sd == sol_sd

    assert_raises(TypeError, test.assign_pspace, 0b0101)
    assert_raises(TypeError, test.assign_pspace, '0101')


def test_system_assign_ref_state():
    """Test SystemEquations.assign_refstate."""
    test = TestSystemEquations()
    test.wfn = CIWavefunction(2, 4)

    test.assign_refstate()
    assert test.refstate == (0b0101, )

    test.assign_refstate(0b0110)
    assert test.refstate == (0b0110, )

    test.assign_refstate([0b0101, 0b0110])
    assert test.refstate == (0b0101, 0b0110)

    ciwfn = CIWavefunction(2, 4)
    test.assign_refstate(ciwfn)
    assert test.refstate == ciwfn

    assert_raises(TypeError, test.assign_refstate, [ciwfn, ciwfn])
    assert_raises(TypeError, test.assign_refstate, '0101')


def test_system_assign_eqn_weights():
    """Test SystemEquations.assign_eqn_weights."""
    test = TestSystemEquations()
    test.wfn = CIWavefunction(2, 4)
    test.assign_pspace()

    test.assign_eqn_weights()
    assert np.allclose(test.eqn_weights, np.array([1, 1, 1, 1, 1, 1, 7]))

    test.assign_eqn_weights(np.array([0, 0, 0, 0, 0, 0, 0], dtype=float))
    assert np.allclose(test.eqn_weights, np.array([0, 0, 0, 0, 0, 0, 0]))

    assert_raises(TypeError, test.assign_eqn_weights, [1, 1, 1, 1, 1, 1, 1])

    assert_raises(TypeError, test.assign_eqn_weights, np.array([0, 0, 0, 0, 0, 0, 0]))

    assert_raises(ValueError, test.assign_eqn_weights, np.array([0, 0, 0, 0, 0, 0], dtype=float))


def test_system_objective():
    """Test SystemEquation.objective."""
    wfn = CIWavefunction(2, 4)
    ham = ChemicalHamiltonian(np.arange(1, 5, dtype=float).reshape(2, 2),
                              np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2))
    weights = np.random.rand(7)
    test = SystemEquations(wfn, ham, eqn_weights=weights)
    # check assignment
    test.objective(np.arange(1, 7, dtype=float))
    np.allclose(wfn.params, np.arange(1, 7))

    # <SD | H | Psi> - E <SD | Psi>
    # computed energy
    objective = test.objective(np.random.rand(6))
    for eqn, sd, weight in zip(objective[:-1],
                               [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]):
        assert np.allclose(eqn,
                           weight * (sum(ham.integrate_wfn_sd(wfn, sd)) -
                                     test.get_energy_one_proj(0b0101) * wfn.get_overlap(sd)))
    assert np.allclose(objective[-1], weights[-1] * (wfn.get_overlap(0b0101)**2 - 1))

    # variable energy
    test = SystemEquations(wfn, ham, energy=1.0, energy_type='variable', eqn_weights=weights)
    guess = np.random.rand(7)
    objective = test.objective(guess)
    for eqn, sd, weight in zip(objective[:-1],
                               [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]):
        assert np.allclose(eqn,
                           weight * (sum(ham.integrate_wfn_sd(wfn, sd)) -
                                     guess[-1] * wfn.get_overlap(sd)))
    assert np.allclose(objective[-1], weights[-1] * (wfn.get_overlap(0b0101)**2 - 1))

    # fixed energy
    test = SystemEquations(wfn, ham, energy=1.0, energy_type='fixed', eqn_weights=weights)
    objective = test.objective(np.random.rand(6))
    for eqn, sd, weight in zip(objective[:-1],
                               [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]):
        assert np.allclose(eqn,
                           weight * (sum(ham.integrate_wfn_sd(wfn, sd)) -
                                     1.0 * wfn.get_overlap(sd)))
    assert np.allclose(objective[-1], weights[-1] * (wfn.get_overlap(0b0101)**2 - 1))
