"""Test wfns.ham.base."""
import numpy as np
from nose.tools import assert_raises
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
    assert_raises(TypeError, BaseHamiltonian.assign_energy_nuc_nuc, test, [-2])
    assert_raises(TypeError, BaseHamiltonian.assign_energy_nuc_nuc, test, '2')
