"""Test wfns.objective.twosided_energy."""
from nose.tools import assert_raises
import numpy as np
from wfns.objective.schrodinger.twosided_energy import TwoSidedEnergy
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.ham.chemical import ChemicalHamiltonian


class TestTwoSidedEnergy(TwoSidedEnergy):
    def __init__(self):
        pass


def test_twosided_energy_assign_pspacess():
    """Test TwoSidedEnergy.assign_pspacess."""
    test = TestTwoSidedEnergy()
    test.wfn = CIWavefunction(2, 4)
    # default pspace_l
    test.assign_pspaces(pspace_l=None, pspace_r=(0b0101, ), pspace_n=[0b0101])
    assert test.pspace_l == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)
    assert test.pspace_r == (0b0101, )
    assert test.pspace_n == (0b0101, )
    # default pspace_r
    test.assign_pspaces(pspace_l=[0b0101], pspace_r=None, pspace_n=(0b0101, ))
    assert test.pspace_l == (0b0101, )
    assert test.pspace_r is None
    assert test.pspace_n == (0b0101, )
    # default pspace_n
    test.assign_pspaces(pspace_l=[0b0101], pspace_r=(0b0101, ), pspace_n=None)
    assert test.pspace_l == (0b0101, )
    assert test.pspace_r == (0b0101, )
    assert test.pspace_n is None
    # error checking
    assert_raises(TypeError, test.assign_pspaces, pspace_l=set([0b0101, 0b1010]))
    assert_raises(TypeError, test.assign_pspaces, pspace_n=CIWavefunction(2, 4))
    assert_raises(ValueError, test.assign_pspaces, pspace_r=[0b1101])
    assert_raises(ValueError, test.assign_pspaces, pspace_r=[0b10001])


def test_num_eqns():
    """Test TwoSidedEnergy.num_eqns."""
    wfn = CIWavefunction(2, 4)
    ham = ChemicalHamiltonian(np.arange(1, 5, dtype=float).reshape(2, 2),
                              np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2))
    test = TwoSidedEnergy(wfn, ham)
    assert test.num_eqns == 1
    assert_raises(TypeError, test.assign_pspaces, pspace_n=['0101'])


def test_twosided_energy_objective():
    """Test TwoSidedEnergy.objective.

    The actualy values of the objective is not checked because it is the same as
    BaseObjective.get_energy_two_proj.

    """
    wfn = CIWavefunction(2, 4)
    ham = ChemicalHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = TwoSidedEnergy(wfn, ham)
    # check assignment
    guess = np.random.rand(6)
    test.objective(guess)
    assert np.allclose(wfn.params, guess)


def test_twosided_energy_gradient():
    """Test TwoSidedEnergy.gradient.

    The actualy values of the gradient is not checked because it is the same as
    BaseObjective.get_energy_two_proj with derivatization.

    """
    wfn = CIWavefunction(2, 4)
    ham = ChemicalHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = TwoSidedEnergy(wfn, ham)
    # check assignment
    guess = np.random.rand(6)
    test.gradient(guess)
    assert np.allclose(wfn.params, guess)
