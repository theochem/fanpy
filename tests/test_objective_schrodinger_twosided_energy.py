"""Test wfns.eqn.twosided_energy."""
import numpy as np
import pytest
from utils import skip_init
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.eqn.twosided_energy import TwoSidedEnergy
from wfns.wfn.ci.base import CIWavefunction


def test_twosided_energy_assign_pspaces():
    """Test TwoSidedEnergy.assign_pspaces."""
    test = skip_init(TwoSidedEnergy)
    test.wfn = CIWavefunction(2, 4)
    # default pspace_l
    test.assign_pspaces(pspace_l=None, pspace_r=(0b0101,), pspace_n=[0b0101])
    assert test.pspace_l == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)
    assert test.pspace_r == (0b0101,)
    assert test.pspace_n == (0b0101,)
    # default pspace_r
    test.assign_pspaces(pspace_l=[0b0101], pspace_r=None, pspace_n=(0b0101,))
    assert test.pspace_l == (0b0101,)
    assert test.pspace_r is None
    assert test.pspace_n == (0b0101,)
    # default pspace_n
    test.assign_pspaces(pspace_l=[0b0101], pspace_r=(0b0101,), pspace_n=None)
    assert test.pspace_l == (0b0101,)
    assert test.pspace_r == (0b0101,)
    assert test.pspace_n is None
    # error checking
    with pytest.raises(TypeError):
        test.assign_pspaces(pspace_l=set([0b0101, 0b1010]))
    with pytest.raises(TypeError):
        test.assign_pspaces(pspace_n=CIWavefunction(2, 4))
    with pytest.raises(ValueError):
        test.assign_pspaces(pspace_r=[0b1101])
    with pytest.raises(ValueError):
        test.assign_pspaces(pspace_r=[0b10001])


def test_num_eqns():
    """Test TwoSidedEnergy.num_eqns."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(1, 5, dtype=float).reshape(2, 2),
        np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2),
    )
    test = TwoSidedEnergy(wfn, ham)
    assert test.num_eqns == 1
    with pytest.raises(TypeError):
        test.assign_pspaces(pspace_n=["0101"])


def test_twosided_energy_objective():
    """Test TwoSidedEnergy.objective.

    The actualy values of the objective is not checked because it is the same as
    BaseSchrodinger.get_energy_two_proj.

    """
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = TwoSidedEnergy(wfn, ham)
    # check assignment
    guess = np.random.rand(6)
    test.objective(guess)
    assert np.allclose(wfn.params, guess)


def test_twosided_energy_gradient():
    """Test TwoSidedEnergy.gradient.

    The actualy values of the gradient is not checked because it is the same as
    BaseSchrodinger.get_energy_two_proj with derivatization.

    """
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = TwoSidedEnergy(wfn, ham)
    # check assignment
    guess = np.random.rand(6)
    test.gradient(guess)
    assert np.allclose(wfn.params, guess)
