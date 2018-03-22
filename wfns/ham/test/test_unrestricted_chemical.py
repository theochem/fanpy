"""Test wfns.ham.unrestricted_chemical."""
import numpy as np
from nose.plugins.attrib import attr
from nose.tools import assert_raises
from wfns.ham.unrestricted_chemical import UnrestrictedChemicalHamiltonian
from wfns.tools import find_datafile


class TestWavefunction(object):
    """Mock wavefunction for testing."""
    def get_overlap(self, sd, deriv=None):
        """Get overlap of wavefunction with Slater determinant."""
        if sd == 0b0101:
            return 1
        elif sd == 0b1010:
            return 2
        elif sd == 0b1100:
            return 3
        return 0


def test_set_ref_ints():
    """Test UnrestrictedChemicalHamiltonian.set_ref_ints."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = UnrestrictedChemicalHamiltonian([one_int]*2, [two_int]*3)
    assert np.allclose(test._ref_one_int, [one_int]*2)
    assert np.allclose(test._ref_two_int, [two_int]*3)

    new_one_int = np.random.rand(2, 2)
    new_two_int = np.random.rand(2, 2, 2, 2)
    test.assign_integrals([new_one_int]*2, [new_two_int]*3)
    assert np.allclose(test._ref_one_int, [one_int]*2)
    assert np.allclose(test._ref_two_int, [two_int]*3)

    test.set_ref_ints()
    assert np.allclose(test._ref_one_int, [new_one_int]*2)
    assert np.allclose(test._ref_two_int, [new_two_int]*3)


def test_assign_params():
    """Test UnrestrictedChemicalHamiltonian.assign_params."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)

    test = UnrestrictedChemicalHamiltonian([one_int]*2, [two_int]*3)
    assert_raises(ValueError, test.assign_params, [0, 0])
    assert_raises(ValueError, test.assign_params, np.array([[0], [0]]))
    assert_raises(ValueError, test.assign_params, np.array([0]))

    test.assign_params(np.array([0, 0]))
    assert np.allclose(test.params, np.zeros(1))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    test.assign_params(np.array([10, 0]))
    assert np.allclose(test.params, np.array([10, 0]))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    test.assign_params(np.array([5, 5]))
    assert np.allclose(test.params, np.array([5, 5]))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    # make sure that transformation is independent of the transformations that came before it
    test.assign_params(np.array([10, 0]))
    assert np.allclose(test.params, np.array([10, 0]))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)


def test_integrate_sd_sd_trivial():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(3, 3)
    two_int = np.random.rand(3, 3, 3, 3)
    test = UnrestrictedChemicalHamiltonian([one_int]*2, [two_int]*3)

    assert_raises(NotImplementedError, test.integrate_sd_sd, 0b001001, 0b100100, sign=None, deriv=0)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b100100, sign=0, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b100100, sign=0.5, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b100100, sign=-0.5, deriv=None)

    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b001001)
    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b111000)
    assert (0, two_int[0, 1, 1, 0], 0) == test.integrate_sd_sd(0b110001, 0b101010, sign=1)
    assert (0, -two_int[0, 1, 1, 0], 0) == test.integrate_sd_sd(0b110001, 0b101010, sign=-1)


def test_integrate_sd_sd_h2_631gdp():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.
    Integrals that correspond to restricted orbitals were used.

    """
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    ham = UnrestrictedChemicalHamiltonian([one_int]*2, [two_int]*3)

    ref_ci_matrix = np.load(find_datafile('test/h2_hf_631gdp_cimatrix.npy'))
    ref_pspace = np.load(find_datafile('test/h2_hf_631gdp_civec.npy'))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose(sum(ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_sd_sd_lih_631g_case():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_sd using sd's of LiH HF/6-31G orbitals."""
    one_int = np.load(find_datafile('test/lih_hf_631g_oneint.npy'))
    two_int = np.load(find_datafile('test/lih_hf_631g_twoint.npy'))
    ham = UnrestrictedChemicalHamiltonian([one_int]*2, [two_int]*3)

    sd1 = 0b0000000001100000000111
    sd2 = 0b0000000001100100001001
    assert (0, two_int[1, 2, 3, 8], -two_int[1, 2, 8, 3]) == ham.integrate_sd_sd(sd1, sd2)
    sd1 = 0b0000000000000000000011
    sd2 = 0b0000000000000000000101
    assert (one_int[1, 2],
            two_int[0, 1, 0, 2], -two_int[0, 1, 2, 0]) == ham.integrate_sd_sd(sd1, sd2)
    sd1 = 0b0000000001100000000000
    sd2 = 0b0000000010100000000000
    assert (one_int[1, 2],
            two_int[0, 1, 0, 2], -two_int[0, 1, 2, 0]) == ham.integrate_sd_sd(sd1, sd2)


@attr('slow')
def test_integrate_sd_sd_lih_631g():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_sd using LiH HF/6-31G orbitals.

    Integrals that correspond to restricted orbitals were used.

    """
    one_int = np.load(find_datafile('test/lih_hf_631g_oneint.npy'))
    two_int = np.load(find_datafile('test/lih_hf_631g_twoint.npy'))
    ham = UnrestrictedChemicalHamiltonian([one_int]*2, [two_int]*3)

    ref_ci_matrix = np.load(find_datafile('test/lih_hf_631g_cimatrix.npy'))
    ref_pspace = np.load(find_datafile('test/lih_hf_631g_civec.npy'))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose(sum(ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_sd_sd_particlenum():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_sd and break particle number symmetery."""
    one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)
    ham = UnrestrictedChemicalHamiltonian([one_int]*2, [two_int]*3)
    civec = [0b01, 0b11]

    # \braket{1 | h_{11} | 1}
    assert np.allclose(sum(ham.integrate_sd_sd(civec[0], civec[0])), 1)
    # \braket{12 | H | 1} = 0
    assert np.allclose(sum(ham.integrate_sd_sd(civec[1], civec[0])), 0)
    assert np.allclose(sum(ham.integrate_sd_sd(civec[0], civec[1])), 0)
    # \braket{12 | h_{11} + h_{22} + g_{1212} - g_{1221} | 12}
    assert np.allclose(sum(ham.integrate_sd_sd(civec[1], civec[1])), 4)


def test_integrate_wfn_sd():
    """Test UnrestrictedChemicalHamiltonian.integrate_wfn_sd."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test_ham = UnrestrictedChemicalHamiltonian([one_int]*2, [two_int]*3)
    test_wfn = TestWavefunction()

    one_energy, coulomb, exchange = test_ham.integrate_wfn_sd(test_wfn, 0b0101)
    assert one_energy == 1*1 + 1*1
    assert coulomb == 1*5 + 2*8
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_wfn_sd(test_wfn, 0b1010)
    assert one_energy == 2*4 + 2*4
    assert coulomb == 1*17 + 2*20
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_wfn_sd(test_wfn, 0b0110)
    assert one_energy == 1*3 + 2*2
    # NOTE: results are different from the restricted results b/c integrals are not symmetric
    assert coulomb == 1*13 + 2*16
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_wfn_sd(test_wfn, 0b1100)
    assert one_energy == 1*3 + 3*4
    assert coulomb == 3*10
    assert exchange == -3*11
