"""Test wfns.ham.senzero."""
import numpy as np
from nose.tools import assert_raises
from wfns.backend.slater import get_seniority
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.ham.senzero import SeniorityZeroHamiltonian
from utils import find_datafile


def test_integrate_sd_sd_trivial():
    """Test SeniorityZeroHamiltonian.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(4, 4)
    two_int = np.random.rand(4, 4, 4, 4)
    test = SeniorityZeroHamiltonian(one_int, two_int)

    assert_raises(NotImplementedError,
                  test.integrate_sd_sd, 0b00010001, 0b01000100, sign=None, deriv=0)
    assert_raises(ValueError, test.integrate_sd_sd, 0b00010001, 0b01000100, sign=0, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b00010001, 0b01000100, sign=0.5, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b00010001, 0b01000100, sign=-0.5, deriv=None)

    assert (0, 0, 0) == test.integrate_sd_sd(0b00010001, 0b00100001)
    assert (0, 0, 0) == test.integrate_sd_sd(0b00100001, 0b00010001)
    assert (0, 0, 0) == test.integrate_sd_sd(0b01010101, 0b00010001)
    assert (0, 0, 0) == test.integrate_sd_sd(0b11001100, 0b00110011)
    assert (0, two_int[0, 0, 1, 1], 0) == test.integrate_sd_sd(0b00010001, 0b00100010, sign=1)
    assert (0, -two_int[0, 0, 1, 1], 0) == test.integrate_sd_sd(0b00010001, 0b00100010, sign=-1)


def test_integrate_sd_sd_h2_631gdp():
    """Test SeniorityZeroHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.

    """
    one_int = np.load(find_datafile('data_h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('data_h2_hf_631gdp_twoint.npy'))
    full_ham = RestrictedChemicalHamiltonian(one_int, two_int)
    test_ham = SeniorityZeroHamiltonian(one_int, two_int)

    ref_pspace = np.load(find_datafile('data_h2_hf_631gdp_civec.npy'))

    for i, sd1 in enumerate(ref_pspace):
        sd1 = int(sd1)
        if get_seniority(sd1, one_int.shape[0]) != 0:
            continue
        for j, sd2 in enumerate(ref_pspace):
            sd2 = int(sd2)
            if get_seniority(sd2, one_int.shape[0]) != 0:
                continue
            assert np.allclose(full_ham.integrate_sd_sd(sd1, sd2),
                               test_ham.integrate_sd_sd(sd1, sd2))


def test_integrate_sd_sd_lih_631g_full():
    """Test SeniorityZeroHamiltonian.integrate_sd_sd using LiH HF/6-31G orbitals.

    Compared to all of the CI matrix.

    """
    one_int = np.load(find_datafile('data_lih_hf_631g_oneint.npy'))
    two_int = np.load(find_datafile('data_lih_hf_631g_twoint.npy'))
    full_ham = RestrictedChemicalHamiltonian(one_int, two_int)
    test_ham = SeniorityZeroHamiltonian(one_int, two_int)

    ref_pspace = np.load(find_datafile('data_lih_hf_631g_civec.npy'))

    for i, sd1 in enumerate(ref_pspace):
        sd1 = int(sd1)
        if get_seniority(sd1, one_int.shape[0]) != 0:
            continue
        for j, sd2 in enumerate(ref_pspace):
            sd2 = int(sd2)
            if get_seniority(sd2, one_int.shape[0]) != 0:
                continue
            assert np.allclose(full_ham.integrate_sd_sd(sd1, sd2),
                               test_ham.integrate_sd_sd(sd1, sd2))


def test_integrate_wfn_sd_2e():
    """Test SeniorityZeroHamiltonian.integrate_wfn_sd with 2 electron wavefunction."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    test_wfn = type(
        "Temporary class with desired overlap",
        (object, ),
        {'get_overlap': lambda sd, deriv=None: 1 if sd == 0b0101 else 2 if sd == 0b1010 else 0},
    )

    assert (0, 0, 0) == ham.integrate_wfn_sd(test_wfn, 0b1001)

    one_energy, coulomb, exchange = ham.integrate_wfn_sd(test_wfn, 0b0101)
    assert one_energy == 1*1 + 1*1
    assert coulomb == 1*5 + 2*8
    assert exchange == 0

    one_energy, coulomb, exchange = ham.integrate_wfn_sd(test_wfn, 0b1010)
    assert one_energy == 2*4 + 2*4
    assert coulomb == 1*17 + 2*20
    assert exchange == 0

    assert_raises(ValueError, ham.integrate_wfn_sd, test_wfn, 0b0101, wfn_deriv=0, ham_deriv=0)


def test_integrate_wfn_sd_4e():
    """Test SeniorityZeroHamiltonian.integrate_wfn_sd with 4 electron wavefunction."""
    one_int = np.arange(1, 10, dtype=float).reshape(3, 3)
    two_int = np.arange(1, 82, dtype=float).reshape(3, 3, 3, 3)
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    ham_full = RestrictedChemicalHamiltonian(one_int, two_int)
    test_wfn = type(
        "Temporary class with desired overlap",
        (object, ),
        {
            'get_overlap':
            lambda sd, deriv=None:
            (
                1 if sd == 0b011011 else 2 if sd == 0b101101 else 3 if sd == 0b110110 else 0
            )
        },
    )

    assert np.allclose(ham.integrate_wfn_sd(test_wfn, 0b011011),
                       ham_full.integrate_wfn_sd(test_wfn, 0b011011))

    assert np.allclose(ham.integrate_wfn_sd(test_wfn, 0b101101),
                       ham_full.integrate_wfn_sd(test_wfn, 0b101101))

    assert np.allclose(ham.integrate_wfn_sd(test_wfn, 0b110110),
                       ham_full.integrate_wfn_sd(test_wfn, 0b110110))

    assert_raises(ValueError, ham.integrate_wfn_sd, test_wfn, 0b0101, wfn_deriv=0, ham_deriv=0)
