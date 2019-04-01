"""Test wfns.ham.base."""
import numpy as np
import pytest
from wfns.ham.base import BaseHamiltonian


class Empty:
    """Empty container class."""
    pass


def test_assign_energy_nuc_nuc():
    """Test BaseHamiltonian.assign_energy_nuc_nuc."""
    # default option
    test = Empty()
    BaseHamiltonian.assign_energy_nuc_nuc(test)
    assert test.energy_nuc_nuc == 0.0

    test = Empty()
    BaseHamiltonian.assign_energy_nuc_nuc(test, None)
    assert test.energy_nuc_nuc == 0.0

    # explicit option
    test = Empty()
    BaseHamiltonian.assign_energy_nuc_nuc(test, 0)
    assert test.energy_nuc_nuc == 0.0

    test = Empty()
    BaseHamiltonian.assign_energy_nuc_nuc(test, 1.5)
    assert test.energy_nuc_nuc == 1.5

    test = Empty()
    BaseHamiltonian.assign_energy_nuc_nuc(test, np.inf)
    assert test.energy_nuc_nuc == np.inf

    # bad option
    test = Empty()
    with pytest.raises(TypeError):
        BaseHamiltonian.assign_energy_nuc_nuc(test, [-2])
    with pytest.raises(TypeError):
        BaseHamiltonian.assign_energy_nuc_nuc(test, '2')


def test_assign_integrals():
    """Test BaseHamiltonian.assign_integrals."""
    test = Empty()
    with pytest.raises(NotImplementedError):
        BaseHamiltonian.assign_integrals(test, None, None)
