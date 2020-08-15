"""Test fanpy.eqn.energy_variational."""
import numpy as np
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.eqn.energy_variational import EnergyVariational
from fanpy.eqn.utils import ComponentParameterIndices
from fanpy.wfn.ci.base import CIWavefunction


def test_energyvariational_init():
    """Test EnergyVariational.__init__."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedMolecularHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = EnergyVariational(wfn, ham, pspace=[0b0101])
    assert test.wfn == wfn
    assert test.ham == ham
    assert test.tmpfile == ""
    indices = ComponentParameterIndices()
    indices[wfn] = np.arange(wfn.nparams)
    assert test.indices_component_params == indices
    assert isinstance(test.step_print, bool) and test.step_print
    assert isinstance(test.step_save, bool) and test.step_save
    assert test.pspace_l == (0b0101, )
    assert test.pspace_r == (0b0101, )
    assert test.pspace_n == (0b0101, )


def test_energyvariational_assign_pspace():
    """Test EnergyVariational.assign_pspace."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedMolecularHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = EnergyVariational(wfn, ham)
    test.assign_pspace()
    assert test.pspace_l == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)
    assert test.pspace_r == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)
    assert test.pspace_n == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)
    test.assign_pspace((0b0110, 0b0011, 0b1010))
    assert test.pspace_l == (0b0110, 0b0011, 0b1010)
    assert test.pspace_r == (0b0110, 0b0011, 0b1010)
    assert test.pspace_n == (0b0110, 0b0011, 0b1010)
