"""Test wfns.hamiltonian.sen0_hamiltonian."""
import numpy as np
from nose.tools import assert_raises
from wfns.hamiltonian.sen0_hamiltonian import SeniorityZeroHamiltonian


def test_assign_orbtype():
    """Test wfns.hamiltonian.sen0_hamiltonian.assign_orbtype."""
    # sneak around calling initializer
    test = SeniorityZeroHamiltonian.__new__(SeniorityZeroHamiltonian)
    assert_raises(NotImplementedError, SeniorityZeroHamiltonian.assign_orbtype, test, 'generalized')


class TestWavefunction_2e(object):
    """Mock 2-electron wavefunction for testing."""
    def get_overlap(self, sd, deriv=None):
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

    one_energy, coulomb, exchange = ham.integrate_wfn_sd(test_wfn, 0b0101, deriv=None)
    assert one_energy == 1*1 + 1*1
    assert coulomb == 1*5 + 2*8
    assert exchange == 0

    one_energy, coulomb, exchange = ham.integrate_wfn_sd(test_wfn, 0b1010, deriv=None)
    assert one_energy == 2*4 + 2*4
    assert coulomb == 1*17 + 2*20
    assert exchange == 0


class TestWavefunction_4e(object):
    """Mock 4-electron wavefunction for testing."""
    def get_overlap(self, sd, deriv=None):
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
    test_wfn = TestWavefunction_4e()

    one_energy, coulomb, exchange = ham.integrate_wfn_sd(test_wfn, 0b011011, deriv=None)
    assert one_energy == 2*1*1 + 2*1*5
    assert coulomb == 1*1 + 4*1*11 + 1*41 + 3*9 + 2*45
    assert exchange == -2*1*13

    one_energy, coulomb, exchange = ham.integrate_wfn_sd(test_wfn, 0b110110, deriv=None)
    assert one_energy == 2*3*5 + 2*3*9
    assert coulomb == 3*41 + 4*3*51 + 3*81 + 2*37 + 1*73
    assert exchange == -2*3*53
