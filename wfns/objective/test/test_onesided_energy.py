"""Test wfns.objective.onesided_energy."""
from nose.tools import assert_raises
import numpy as np
from wfns.objective.onesided_energy import OneSidedEnergy
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


class TestOneSidedEnergy(OneSidedEnergy):
    def __init__(self):
        pass


def test_onesided_energy_assign_refwfn():
    """Test OneSidedEnergy.assign_refwfn."""
    test = TestOneSidedEnergy()
    test.wfn = CIWavefunction(2, 4)

    test.assign_refwfn(refwfn=None)
    assert list(test.refwfn) == [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]

    test.assign_refwfn(refwfn=[0b0101])
    assert test.refwfn == (0b0101, )
    test.assign_refwfn(refwfn=(0b0101, ))
    assert test.refwfn == (0b0101, )
    ciwfn = CIWavefunction(2, 4)
    test.assign_refwfn(refwfn=ciwfn)
    assert test.refwfn == ciwfn

    assert_raises(TypeError, test.assign_refwfn, refwfn=set([0b0101, 0b1010]))
    assert_raises(ValueError, test.assign_refwfn, refwfn=[0b1101])
    assert_raises(ValueError, test.assign_refwfn, refwfn=[0b10001])
    assert_raises(ValueError, test.assign_refwfn, refwfn=CIWavefunction(3, 4))
    assert_raises(ValueError, test.assign_refwfn, refwfn=CIWavefunction(2, 6))
    assert_raises(TypeError, test.assign_refwfn, refwfn=['0101'])


def test_onesided_energy_objective():
    """Test OneSidedEnergy.objective.

    The actualy values of the objective is not checked because it is the same as
    BaseObjective.get_energy_one_proj.

    """
    wfn = CIWavefunction(2, 4)
    ham = ChemicalHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = OneSidedEnergy(wfn, ham)
    # check assignment
    guess = np.random.rand(6)
    test.objective(guess)
    assert np.allclose(wfn.params, guess)


def test_onesided_energy_gradient():
    """Test OneSidedEnergy.gradient.

    The actualy values of the gradient is not checked because it is the same as
    BaseObjective.get_energy_one_proj with derivatization.

    """
    wfn = CIWavefunction(2, 4)
    ham = ChemicalHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = OneSidedEnergy(wfn, ham)
    # check assignment
    guess = np.random.rand(6)
    test.gradient(guess)
    assert np.allclose(wfn.params, guess)
