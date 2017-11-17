"""Test wfns.ham.senzero."""
import numpy as np
from nose.tools import assert_raises
from wfns.ham.chemical import ChemicalHamiltonian
from wfns.ham.senzero import SeniorityZeroHamiltonian

# FIXME: need more tests for checking integrate_wfn_sd and integrate_sd_sd


def test_assign_orbtype():
    """Test wfns.ham.senzero.assign_orbtype."""
    # sneak around calling initializer
    test = SeniorityZeroHamiltonian.__new__(SeniorityZeroHamiltonian)
    assert_raises(NotImplementedError, SeniorityZeroHamiltonian.assign_orbtype, test, 'generalized')


class TestWavefunction_2e(object):
    """Mock 2-electron wavefunction for testing."""
    def get_overlap(self, sd, deriv=None):
        """Get overlap of wavefunction with Slater determinant."""
        if sd == 0b0101:
            return 1
        elif sd == 0b1010:
            return 2
        elif sd == 0b1100:
            return 3
        return 0


def test_integrate_wfn_sd_2e():
    """Test SeniorityZeroHamiltonian.integrate_wfn_sd with 2 electron wavefunction."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    ham = SeniorityZeroHamiltonian(one_int, two_int, 'restricted')
    test_wfn = TestWavefunction_2e()

    one_energy, coulomb, exchange = ham.integrate_wfn_sd(test_wfn, 0b0101)
    assert one_energy == 1*1 + 1*1
    assert coulomb == 1*5 + 2*8
    assert exchange == 0

    one_energy, coulomb, exchange = ham.integrate_wfn_sd(test_wfn, 0b1010)
    assert one_energy == 2*4 + 2*4
    assert coulomb == 1*17 + 2*20
    assert exchange == 0


class TestWavefunction_4e(object):
    """Mock 4-electron wavefunction for testing."""
    def get_overlap(self, sd, deriv=None):
        """Get overlap of wavefunction with Slater determinant."""
        if sd == 0b011011:
            return 1
        elif sd == 0b101101:
            return 2
        elif sd == 0b110110:
            return 3
        return 0


def test_integrate_wfn_sd_4e():
    """Test SeniorityZeroHamiltonian.integrate_wfn_sd with 4 electron wavefunction."""
    one_int = np.arange(1, 10, dtype=float).reshape(3, 3)
    two_int = np.arange(1, 82, dtype=float).reshape(3, 3, 3, 3)
    ham = SeniorityZeroHamiltonian(one_int, two_int, 'restricted')
    ham_full = ChemicalHamiltonian(one_int, two_int, 'restricted')
    test_wfn = TestWavefunction_4e()

    assert np.allclose(ham.integrate_wfn_sd(test_wfn, 0b011011),
                       ham_full.integrate_wfn_sd(test_wfn, 0b011011))

    assert np.allclose(ham.integrate_wfn_sd(test_wfn, 0b101101),
                       ham_full.integrate_wfn_sd(test_wfn, 0b101101))

    assert np.allclose(ham.integrate_wfn_sd(test_wfn, 0b110110),
                       ham_full.integrate_wfn_sd(test_wfn, 0b110110))
