"""Test fanpy.eqn.local_energy."""
import numpy as np
import pytest
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.local_energy import LocalEnergy
from fanpy.eqn.utils import ComponentParameterIndices
from fanpy.wfn.ci.base import CIWavefunction


def test_localenergy_init():
    """Test LocalEnergy.__init__."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedMolecularHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = LocalEnergy(wfn, ham, pspace=[0b0101])
    assert test.wfn == wfn
    assert test.ham == ham
    assert test.tmpfile == ""
    indices = ComponentParameterIndices()
    indices[wfn] = np.arange(wfn.nparams)
    assert test.indices_component_params == indices
    assert isinstance(test.step_print, bool) and test.step_print
    assert isinstance(test.step_save, bool) and test.step_save
    assert test.pspace == (0b0101,)


def test_localenergy_assign_pspace():
    """Test LocalEnergy.assign_pspace."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedMolecularHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = LocalEnergy(wfn, ham)
    test.assign_pspace()
    assert test.pspace == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)
    test.assign_pspace((0b0110, 0b0011, 0b1010))
    assert test.pspace == (0b0110, 0b0011, 0b1010)

    with pytest.raises(TypeError):
        test.assign_pspace(set([0b0110, 0b0011, 0b1010]))


def test_localenergy_objective():
    """Test LocalEnergy.objective."""
    wfn = CIWavefunction(4, 10)
    wfn.assign_params(np.random.rand(wfn.nparams))

    one_int = np.random.rand(5, 5)
    one_int = one_int + one_int.T
    two_int = np.random.rand(5, 5, 5, 5)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    ham = RestrictedMolecularHamiltonian(one_int, two_int)

    test = LocalEnergy(wfn, ham, pspace=(0b0001100011,))
    test2 = EnergyOneSideProjection(wfn, ham, refwfn=(0b0001100011,))
    assert np.allclose(test.objective(test.active_params), test2.objective(test2.active_params))

    test = LocalEnergy(wfn, ham, pspace=(0b0001100011, 0b0110001100))
    test2 = EnergyOneSideProjection(wfn, ham, refwfn=(0b0001100011,))
    test3 = EnergyOneSideProjection(wfn, ham, refwfn=(0b0110001100,))
    assert np.allclose(
        test.objective(test.active_params),
        test2.objective(test2.active_params) + test3.objective(test3.active_params),
    )

    test = LocalEnergy(
        wfn, ham, pspace=(0b0001100011, 0b0110001100), step_print=False, step_save=False
    )
    assert np.allclose(
        test.objective(test.active_params),
        test2.objective(test2.active_params) + test3.objective(test3.active_params),
    )


def test_localenergy_gradient():
    """Test LocalEnergy.gradient."""
    nd = pytest.importorskip("numdifftools")
    wfn = CIWavefunction(4, 6)
    wfn.assign_params(np.random.rand(wfn.nparams))

    one_int = np.random.rand(3, 3)
    one_int = one_int + one_int.T
    two_int = np.random.rand(3, 3, 3, 3)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    ham = RestrictedMolecularHamiltonian(one_int, two_int)

    test = LocalEnergy(wfn, ham, pspace=(0b011011, 0b101101))

    def objective(params):
        temp_ham = RestrictedMolecularHamiltonian(one_int.copy(), two_int.copy())
        temp_wfn = CIWavefunction(4, 6)
        temp_obj = LocalEnergy(temp_wfn, temp_ham, pspace=test.pspace, step_print=False)
        return temp_obj.objective(params)

    grad = nd.Gradient(objective)(test.active_params.copy())
    assert np.allclose(grad, test.gradient(test.active_params))

    test = LocalEnergy(wfn, ham, pspace=(0b011011, 0b101101))
    test.step_print = False
    assert np.allclose(grad, test.gradient(test.active_params))
