"""Test wfns.objective.system_nonlinear."""
from nose.tools import assert_raises
import numpy as np
from wfns.param import ParamContainer, ParamMask
from wfns.objective.schrodinger.system_nonlinear import SystemEquations
from wfns.objective.constraints.norm import NormConstraint
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


def test_system_assign_refstate():
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
    assert_raises(TypeError, test.assign_refstate, np.array([0b0101, 0b0110]))


def test_system_assign_eqn_weights():
    """Test SystemEquations.assign_eqn_weights."""
    test = TestSystemEquations()
    test.wfn = CIWavefunction(2, 4)
    test.assign_pspace()
    test.assign_refstate()
    test.assign_param_selection()
    test.assign_constraints()

    test.assign_eqn_weights()
    assert np.allclose(test.eqn_weights, np.array([1, 1, 1, 1, 1, 1, 6]))

    test.assign_eqn_weights(np.array([0, 0, 0, 0, 0, 0, 0], dtype=float))
    assert np.allclose(test.eqn_weights, np.array([0, 0, 0, 0, 0, 0, 0]))

    test.param_selection = ParamMask((ParamContainer(test.wfn.params), np.ones(6, dtype=bool)))
    norm_constraint = NormConstraint(test.wfn, param_selection=test.param_selection)
    test.assign_constraints([norm_constraint, norm_constraint])
    test.assign_eqn_weights(np.zeros(8))

    assert_raises(TypeError, test.assign_eqn_weights, [1, 1, 1, 1, 1, 1, 1])

    assert_raises(TypeError, test.assign_eqn_weights, np.array([0, 0, 0, 0, 0, 0, 0]))

    assert_raises(ValueError, test.assign_eqn_weights, np.array([0, 0, 0, 0, 0, 0], dtype=float))


def test_system_assign_constraints():
    """Test SystemEquations.assign_constraints."""
    test = TestSystemEquations()
    test.wfn = CIWavefunction(2, 4)
    test.refstate = (0b0101, )
    test.param_selection = ParamMask((ParamContainer(test.wfn.params), np.ones(6, dtype=bool)))

    test.assign_constraints()
    assert isinstance(test.constraints, list)
    assert len(test.constraints) == 1
    assert isinstance(test.constraints[0], NormConstraint)
    assert test.constraints[0].wfn == test.wfn
    assert test.constraints[0].refwfn == (0b0101, )

    norm_constraint = NormConstraint(test.wfn, param_selection=test.param_selection)
    test.assign_constraints(norm_constraint)
    assert isinstance(test.constraints, list)
    assert len(test.constraints) == 1
    assert isinstance(test.constraints[0], NormConstraint)
    assert test.constraints[0].wfn == test.wfn
    assert test.constraints[0].refwfn == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)

    assert_raises(TypeError, test.assign_constraints, lambda x: None)
    assert_raises(TypeError, test.assign_constraints, np.array(norm_constraint))
    assert_raises(TypeError, test.assign_constraints, [norm_constraint, lambda x: None])
    norm_constraint.assign_param_selection(ParamMask((ParamContainer(test.wfn.params),
                                                      np.ones(6, dtype=bool))))
    assert_raises(ValueError, test.assign_constraints, norm_constraint)
    assert_raises(ValueError, test.assign_constraints, [norm_constraint])


def test_num_eqns():
    """Test SystemEquation.num_eqns."""
    test = TestSystemEquations()
    test.pspace = (0b0101, 0b1010)
    assert test.num_eqns == 3


def test_system_objective():
    """Test SystemEquation.objective."""
    wfn = CIWavefunction(2, 4)
    ham = ChemicalHamiltonian(np.arange(1, 5, dtype=float).reshape(2, 2),
                              np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2))
    weights = np.random.rand(7)
    # check assignment
    test = SystemEquations(wfn, ham, eqn_weights=weights)
    test.objective(np.arange(1, 7, dtype=float))
    np.allclose(wfn.params, np.arange(1, 7))

    # <SD1 | H | Psi> - E <SD | Psi>
    ciref = CIWavefunction(2, 4)
    ciref.assign_params(np.random.rand(6))
    for refwfn in [0b0101, [0b0101, 0b1010], ciref]:
        guess = np.random.rand(7)
        # computed energy
        test = SystemEquations(wfn, ham, eqn_weights=weights, refstate=refwfn)
        wfn.assign_params(guess[:6])
        if refwfn == 0b0101:
            norm_answer = weights[-1] * (wfn.get_overlap(0b0101)**2 - 1)
        elif refwfn == [0b0101, 0b1010]:
            norm_answer = weights[-1] * (wfn.get_overlap(0b0101)**2 +
                                         wfn.get_overlap(0b1010)**2 - 1)
        elif refwfn == ciref:
            norm_answer = weights[-1] * (sum(ciref.get_overlap(sd) * wfn.get_overlap(sd)
                                             for sd in ciref.sd_vec) - 1)

        objective = test.objective(guess[:6])
        for eqn, sd, weight in zip(objective[:-1],
                                   [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]):
            assert np.allclose(eqn,
                               weight * (sum(ham.integrate_wfn_sd(wfn, sd)) -
                                         test.get_energy_one_proj(refwfn)
                                         * wfn.get_overlap(sd)))
        assert np.allclose(objective[-1], norm_answer)

        # variable energy
        test = SystemEquations(wfn, ham, energy=1.0, energy_type='variable', eqn_weights=weights,
                               refstate=refwfn)
        objective = test.objective(guess)
        for eqn, sd, weight in zip(objective[:-1],
                                   [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]):
            assert np.allclose(eqn,
                               weight * (sum(ham.integrate_wfn_sd(wfn, sd)) -
                                         guess[-1] * wfn.get_overlap(sd)))
        assert np.allclose(objective[-1], norm_answer)

        # fixed energy
        test = SystemEquations(wfn, ham, energy=1.0, energy_type='fixed', eqn_weights=weights,
                               refstate=refwfn)
        objective = test.objective(guess[:6])
        for eqn, sd, weight in zip(objective[:-1],
                                   [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]):
            assert np.allclose(eqn,
                               weight * (sum(ham.integrate_wfn_sd(wfn, sd)) -
                                         1.0 * wfn.get_overlap(sd)))
        assert np.allclose(objective[-1], norm_answer)


def test_system_jacobian():
    """Test SystemEquation.jacobian with only wavefunction parameters active."""
    wfn = CIWavefunction(2, 4)
    ham = ChemicalHamiltonian(np.arange(1, 5, dtype=float).reshape(2, 2),
                              np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2))
    weights = np.random.rand(7)

    # check assignment
    test = SystemEquations(wfn, ham, eqn_weights=weights)
    test.jacobian(np.arange(1, 7, dtype=float))
    np.allclose(wfn.params, np.arange(1, 7))

    # df_1/dx_1 = d/dx_1 <SD_1 | H | Psi> - dE/dx_1 <SD_1 | Psi> - E d/dx_1 <SD_1 | Psi>
    ciref = CIWavefunction(2, 4)
    ciref.assign_params(np.random.rand(6))
    for refwfn in [0b0101, [0b0101, 0b1010], ciref]:
        guess = np.random.rand(7)
        # computed energy
        test = SystemEquations(wfn, ham, eqn_weights=weights, refstate=refwfn)
        wfn.assign_params(guess[:6])
        if refwfn == 0b0101:
            norm_answer = [weights[-1] * (2 * wfn.get_overlap(0b0101) *
                                          wfn.get_overlap(0b0101, deriv=i)) for i in range(6)]
        elif refwfn == [0b0101, 0b1010]:
            norm_answer = [weights[-1] * (2 * wfn.get_overlap(0b0101) *
                                          wfn.get_overlap(0b0101, deriv=i) +
                                          2 * wfn.get_overlap(0b1010) *
                                          wfn.get_overlap(0b1010, deriv=i)) for i in range(6)]
        elif refwfn == ciref:
            norm_answer = [weights[-1] * (sum(ciref.get_overlap(sd) * wfn.get_overlap(sd, deriv=i)
                                              for sd in ciref.sd_vec)) for i in range(6)]

        jacobian = test.jacobian(guess[:6])
        for eqn, sd, weight in zip(jacobian[:-1],
                                   [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]):
            for i in range(6):
                assert np.allclose(eqn[i],
                                   weight * (sum(ham.integrate_wfn_sd(wfn, sd, wfn_deriv=i)) -
                                             test.get_energy_one_proj(refwfn, deriv=i)
                                             * wfn.get_overlap(sd) -
                                             test.get_energy_one_proj(refwfn)
                                             * wfn.get_overlap(sd, deriv=i)))
        assert np.allclose(jacobian[-1], norm_answer)

        # variable energy
        test = SystemEquations(wfn, ham, energy=3.0, energy_type='variable', eqn_weights=weights,
                               refstate=refwfn)
        jacobian = test.jacobian(guess)
        for eqn, sd, weight in zip(jacobian[:-1],
                                   [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]):
            for i in range(7):
                assert np.allclose(eqn[i],
                                   weight * (sum(ham.integrate_wfn_sd(wfn, sd, wfn_deriv=i)) -
                                             int(i == 6) * wfn.get_overlap(sd) -
                                             guess[-1] * wfn.get_overlap(sd, deriv=i)))
        assert np.allclose(jacobian[-1], norm_answer + [0.0])

        # fixed energy
        test = SystemEquations(wfn, ham, energy=1.0, energy_type='fixed', eqn_weights=weights,
                               refstate=refwfn)
        jacobian = test.jacobian(guess[:6])
        for eqn, sd, weight in zip(jacobian[:-1],
                                   [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]):
            for i in range(6):
                assert np.allclose(eqn[i],
                                   weight * (sum(ham.integrate_wfn_sd(wfn, sd, wfn_deriv=i)) -
                                             0.0 * wfn.get_overlap(sd) -
                                             1 * wfn.get_overlap(sd, deriv=i)))
        assert np.allclose(jacobian[-1], norm_answer)


def test_system_jacobian_active_ciref():
    """Test SystemEquation.jacobian with CIWavefunction reference with active parameters."""
    wfn = CIWavefunction(2, 4)
    ham = ChemicalHamiltonian(np.arange(1, 5, dtype=float).reshape(2, 2),
                              np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2))
    weights = np.random.rand(7)

    ciref = CIWavefunction(2, 4)
    ciref.assign_params(np.random.rand(6))

    # computed energy
    test = SystemEquations(wfn, ham, eqn_weights=weights, refstate=ciref,
                           param_selection=((wfn, np.ones(6, dtype=bool)),
                                            (ciref, np.ones(6, dtype=bool))))

    jacobian = test.jacobian(np.random.rand(12))
    for eqn, sd, weight in zip(jacobian[:-1],
                               [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]):
        for i in range(12):
            assert np.allclose(eqn[i],
                               weight * (sum(ham.integrate_wfn_sd(wfn, sd, wfn_deriv=i)) -
                                         test.get_energy_one_proj(ciref, deriv=i)
                                         * wfn.get_overlap(sd) -
                                         test.get_energy_one_proj(ciref)
                                         * wfn.get_overlap(sd, deriv=i)))
    assert np.allclose(jacobian[-1],
                       [weights[-1] * (sum(ciref.get_overlap(sd) * wfn.get_overlap(sd, deriv=i)
                                           for sd in ciref.sd_vec)) for i in range(6)] +
                       [weights[-1] * (sum(ciref.get_overlap(sd, deriv=i) * wfn.get_overlap(sd)
                                           for sd in ciref.sd_vec)) for i in range(6)])
