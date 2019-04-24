"""Test wfns.ham.unrestricted_chemical."""
import itertools as it

import numpy as np
import pytest
from utils import find_datafile
from wfns.backend import slater
from wfns.backend.sd_list import sd_list
from wfns.ham.unrestricted_chemical import UnrestrictedChemicalHamiltonian
from wfns.wfn.ci.base import CIWavefunction


def test_set_ref_ints():
    """Test UnrestrictedChemicalHamiltonian.set_ref_ints."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)
    assert np.allclose(test._ref_one_int, [one_int] * 2)
    assert np.allclose(test._ref_two_int, [two_int] * 3)

    new_one_int = np.random.rand(2, 2)
    new_two_int = np.random.rand(2, 2, 2, 2)
    test.assign_integrals([new_one_int] * 2, [new_two_int] * 3)
    assert np.allclose(test._ref_one_int, [one_int] * 2)
    assert np.allclose(test._ref_two_int, [two_int] * 3)

    test.set_ref_ints()
    assert np.allclose(test._ref_one_int, [new_one_int] * 2)
    assert np.allclose(test._ref_two_int, [new_two_int] * 3)


def test_cache_two_ints():
    """Test UnrestrictedChemicalHamiltonian.cache_two_ints."""
    one_int = [np.arange(1, 5, dtype=float).reshape(2, 2)] * 2
    two_int = [np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)] * 3
    two_int_ijij = np.array([[5, 10], [15, 20]])
    two_int_ijji = np.array([[5, 11], [14, 20]])

    test = UnrestrictedChemicalHamiltonian(one_int, two_int)
    assert np.allclose(test._cached_two_int_0_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_1_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_2_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_0_ijji, two_int_ijji)
    assert np.allclose(test._cached_two_int_2_ijji, two_int_ijji)

    test.two_int = [np.arange(21, 37).reshape(2, 2, 2, 2)] * 3
    new_two_int_ijij = np.array([[21, 26], [31, 36]])
    new_two_int_ijji = np.array([[21, 27], [30, 36]])
    assert np.allclose(test._cached_two_int_0_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_1_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_2_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_0_ijji, two_int_ijji)
    assert np.allclose(test._cached_two_int_2_ijji, two_int_ijji)

    test.cache_two_ints()
    assert np.allclose(test._cached_two_int_0_ijij, new_two_int_ijij)
    assert np.allclose(test._cached_two_int_1_ijij, new_two_int_ijij)
    assert np.allclose(test._cached_two_int_2_ijij, new_two_int_ijij)
    assert np.allclose(test._cached_two_int_0_ijji, new_two_int_ijji)
    assert np.allclose(test._cached_two_int_2_ijji, new_two_int_ijji)


def test_assign_params():
    """Test UnrestrictedChemicalHamiltonian.assign_params."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)

    test = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)
    with pytest.raises(ValueError):
        test.assign_params([0, 0])
    with pytest.raises(ValueError):
        test.assign_params(np.array([[0], [0]]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([0]))

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
    test = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)

    with pytest.raises(ValueError):
        test.integrate_sd_sd(0b001001, 0b100100, sign=0, deriv=None)
    with pytest.raises(ValueError):
        test.integrate_sd_sd(0b001001, 0b100100, sign=0.5, deriv=None)
    with pytest.raises(ValueError):
        test.integrate_sd_sd(0b001001, 0b100100, sign=-0.5, deriv=None)

    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b001001)
    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b111000)
    assert (0, two_int[0, 1, 1, 0], 0) == test.integrate_sd_sd(0b110001, 0b101010, sign=1)
    assert (0, -two_int[0, 1, 1, 0], 0) == test.integrate_sd_sd(0b110001, 0b101010, sign=-1)
    assert (0, -two_int[1, 1, 1, 0] + two_int[0, 1, 0, 0], 0) == test.integrate_sd_sd(
        0b110001, 0b101010, deriv=0
    )


def test_integrate_sd_sd_h2_631gdp():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.
    Integrals that correspond to restricted orbitals were used.

    """
    one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    ham = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)

    ref_ci_matrix = np.load(find_datafile("data_h2_hf_631gdp_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("data_h2_hf_631gdp_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose(sum(ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_sd_sd_lih_631g_case():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_sd using sd's of LiH HF/6-31G orbitals."""
    one_int = np.load(find_datafile("data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_631g_twoint.npy"))
    ham = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)

    sd1 = 0b0000000001100000000111
    sd2 = 0b0000000001100100001001
    assert (0, two_int[1, 2, 3, 8], -two_int[1, 2, 8, 3]) == ham.integrate_sd_sd(sd1, sd2)
    sd1 = 0b0000000000000000000011
    sd2 = 0b0000000000000000000101
    assert (one_int[1, 2], two_int[0, 1, 0, 2], -two_int[0, 1, 2, 0]) == ham.integrate_sd_sd(
        sd1, sd2
    )
    sd1 = 0b0000000001100000000000
    sd2 = 0b0000000010100000000000
    assert (one_int[1, 2], two_int[0, 1, 0, 2], -two_int[0, 1, 2, 0]) == ham.integrate_sd_sd(
        sd1, sd2
    )


def test_integrate_sd_sd_lih_631g_slow():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_sd using LiH HF/6-31G orbitals.

    Integrals that correspond to restricted orbitals were used.

    """
    one_int = np.load(find_datafile("data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_631g_twoint.npy"))
    ham = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)

    ref_ci_matrix = np.load(find_datafile("data_lih_hf_631g_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("data_lih_hf_631g_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose(sum(ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_sd_sd_particlenum():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_sd and break particle number symmetery."""
    one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)
    ham = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)
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
    test_ham = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)
    test_wfn = type(
        "Temporary wavefunction.",
        (object,),
        {
            "get_overlap": lambda sd, deriv=None: 1
            if sd == 0b0101
            else 2
            if sd == 0b1010
            else 3
            if sd == 0b1100
            else 0
        },
    )

    one_energy, coulomb, exchange = test_ham.integrate_wfn_sd(test_wfn, 0b0101)
    assert one_energy == 1 * 1 + 1 * 1
    assert coulomb == 1 * 5 + 2 * 8
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_wfn_sd(test_wfn, 0b1010)
    assert one_energy == 2 * 4 + 2 * 4
    assert coulomb == 1 * 17 + 2 * 20
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_wfn_sd(test_wfn, 0b0110)
    assert one_energy == 1 * 3 + 2 * 2
    # NOTE: results are different from the restricted results b/c integrals are not symmetric
    assert coulomb == 1 * 13 + 2 * 16
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_wfn_sd(test_wfn, 0b1100)
    assert one_energy == 1 * 3 + 3 * 4
    assert coulomb == 3 * 10
    assert exchange == -3 * 11


def test_param_ind_to_rowcol_ind():
    """Test UnrestrictedChemicalHamiltonian.param_ind_to_rowcol_ind."""
    for n in range(1, 20):
        ham = UnrestrictedChemicalHamiltonian(
            [np.random.rand(n, n)] * 2, [np.random.rand(n, n, n, n)] * 3
        )
        for row_ind in range(n):
            for col_ind in range(row_ind + 1, n):
                param_ind = row_ind * n - row_ind * (row_ind + 1) / 2 + col_ind - row_ind - 1
                assert ham.param_ind_to_rowcol_ind(param_ind) == (0, row_ind, col_ind)
                assert ham.param_ind_to_rowcol_ind(param_ind + ham.nparams // 2) == (
                    1,
                    row_ind,
                    col_ind,
                )


def test_integrate_sd_sd_deriv():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sd_deriv."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test_ham = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)

    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, 0.0)
    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, -1)
    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, 2)
    assert test_ham._integrate_sd_sd_deriv(0b0101, 0b0001, 0) == (0, 0, 0)
    assert test_ham._integrate_sd_sd_deriv(0b000111, 0b111000, 0) == (0, 0, 0)


def test_integrate_sd_sd_deriv_fdiff_h2_sto6g():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sd_deriv using H2/STO6G.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.load(find_datafile("data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_sto6g_twoint.npy"))
    test_ham = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)
    epsilon = 1e-8

    for sd1 in [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]:
        for sd2 in [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]:
            for i in range(2):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = UnrestrictedChemicalHamiltonian(
                    [one_int] * 2, [two_int] * 3, params=addition
                )

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=1e-5)


def test_integrate_sd_sd_deriv_fdiff_h4_sto6g_slow():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sd_deriv with using H4/STO6G.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.load(find_datafile("data_h4_square_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_h4_square_hf_sto6g_twoint.npy"))

    test_ham = UnrestrictedChemicalHamiltonian([one_int] * 2, [two_int] * 3)
    epsilon = 1e-8
    sds = sd_list(4, 4, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = UnrestrictedChemicalHamiltonian(
                    [one_int] * 2, [two_int] * 3, params=addition
                )

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)


def test_integrate_sd_sd_deriv_fdiff_random():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sd_deriv using random integrals.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int_a = np.random.rand(4, 4)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(4, 4)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(4, 4, 4, 4)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(4, 4, 4, 4)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(4, 4, 4, 4)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    # check that the in tegrals have the appropriate symmetry
    assert np.allclose(one_int_a, one_int_a.T)
    assert np.allclose(one_int_b, one_int_b.T)
    assert np.allclose(two_int_aaaa, np.einsum("ijkl->jilk", two_int_aaaa))
    assert np.allclose(two_int_aaaa, np.einsum("ijkl->klij", two_int_aaaa))
    assert np.allclose(two_int_abab, np.einsum("ijkl->klij", two_int_abab))
    assert np.allclose(two_int_bbbb, np.einsum("ijkl->jilk", two_int_bbbb))
    assert np.allclose(two_int_bbbb, np.einsum("ijkl->klij", two_int_bbbb))

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )
    epsilon = 1e-8
    sds = sd_list(2, 4, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = UnrestrictedChemicalHamiltonian(
                    [one_int_a, one_int_b],
                    [two_int_aaaa, two_int_abab, two_int_bbbb],
                    params=addition,
                )

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)


def test_integrate_sd_sd_deriv_fdiff_random_small():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sd_deriv using random 1e system.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int_a = np.random.rand(2, 2)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(2, 2)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(2, 2, 2, 2)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(2, 2, 2, 2)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(2, 2, 2, 2)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    # check that the in tegrals have the appropriate symmetry
    assert np.allclose(one_int_a, one_int_a.T)
    assert np.allclose(one_int_b, one_int_b.T)
    assert np.allclose(two_int_aaaa, np.einsum("ijkl->jilk", two_int_aaaa))
    assert np.allclose(two_int_aaaa, np.einsum("ijkl->klij", two_int_aaaa))
    assert np.allclose(two_int_abab, np.einsum("ijkl->klij", two_int_abab))
    assert np.allclose(two_int_bbbb, np.einsum("ijkl->jilk", two_int_bbbb))
    assert np.allclose(two_int_bbbb, np.einsum("ijkl->klij", two_int_bbbb))

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )
    epsilon = 1e-8
    sds = sd_list(1, 2, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = UnrestrictedChemicalHamiltonian(
                    [one_int_a, one_int_b],
                    [two_int_aaaa, two_int_abab, two_int_bbbb],
                    params=addition,
                )

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)


def test_integrate_sd_sds_zero():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_zero against _integrate_sd_sd_zero."""
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3])
    assert np.allclose(
        test_ham._integrate_sd_sds_zero(occ_alpha, occ_beta),
        np.array(test_ham._integrate_sd_sd_zero(occ_alpha, occ_beta)).reshape(3, 1),
    )


def test_integrate_sd_sds_one_alpha():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_one_alpha.

    Compared against UnrestrictedChemicalHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])

    assert np.allclose(
        test_ham._integrate_sd_sds_one_alpha(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                np.array(
                    test_ham._integrate_sd_sd_one((i,), (j,), occ_alpha[occ_alpha != i], occ_beta)
                )
                * slater.sign_excite(0b101101011001, [i], [j])
                for i in occ_alpha.tolist()
                for j in vir_alpha.tolist()
            ]
        ).T,
    )


def test_integrate_sd_sds_one_beta():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_one_beta.

    Compared against UnrestrictedChemicalHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])

    assert np.allclose(
        test_ham._integrate_sd_sds_one_beta(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                np.array(
                    test_ham._integrate_sd_sd_one(
                        (i + 6,), (j + 6,), occ_alpha, occ_beta[occ_beta != i]
                    )
                )
                * slater.sign_excite(0b101101011001, [i + 6], [j + 6])
                for i in occ_beta.tolist()
                for j in vir_beta.tolist()
            ]
        ).T,
    )


def test_integrate_sd_sds_two_aa():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_two_aa.

    Compared against UnrestrictedChemicalHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_two_aa(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_two(diff1, diff2))
                * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                for diff1 in it.combinations(occ_alpha.tolist(), 2)
                for diff2 in it.combinations(vir_alpha.tolist(), 2)
            ]
        ).T[[1, 2]],
    )


def test_integrate_sd_sds_two_ab():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_two_ab.

    Compared against UnrestrictedChemicalHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    vir_beta = np.array([1, 4])

    assert np.allclose(
        test_ham._integrate_sd_sds_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_two(diff1, diff2))
                * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                for diff1 in it.product(occ_alpha.tolist(), (occ_beta + 6).tolist())
                for diff2 in it.product(vir_alpha.tolist(), (vir_beta + 6).tolist())
            ]
        ).T[1],
    )


def test_integrate_sd_sds_two_bb():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_two_bb.

    Compared against UnrestrictedChemicalHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])

    assert np.allclose(
        test_ham._integrate_sd_sds_two_bb(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_two(diff1, diff2))
                * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                for diff1 in it.combinations((occ_beta + 6).tolist(), 2)
                for diff2 in it.combinations((vir_beta + 6).tolist(), 2)
            ]
        ).T[[1, 2]],
    )


def test_integrate_sd_sds_deriv_zero_alpha():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_deriv_zero_alpha.

    Compared with UnrestrictedChemicalHamiltonian._integrate_sd_sd_zero.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_zero_alpha(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    test_ham._integrate_sd_sd_deriv_zero(0, x, y, occ_alpha, occ_beta)
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_zero_beta():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_deriv_zero_beta.

    Compared with UnrestrictedChemicalHamiltonian._integrate_sd_sd_zero.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_zero_beta(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    test_ham._integrate_sd_sd_deriv_zero(1, x, y, occ_alpha, occ_beta)
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_one_aa():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_deriv_one_aa.

    Compared with UnrestrictedChemicalHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])

    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_aa(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one(
                            [i], [j], 0, x, y, occ_alpha[occ_alpha != i], occ_beta
                        )
                    )
                    * slater.sign_excite(0b101101011001, [i], [j])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_alpha.tolist()
                for j in vir_alpha.tolist()
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_one_ab():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_deriv_one_ab.

    Compared with UnrestrictedChemicalHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_ab(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one(
                            [i + 6], [j + 6], 0, x, y, occ_alpha, occ_beta[occ_beta != i]
                        )
                    )
                    * slater.sign_excite(0b101101011001, [i + 6], [j + 6])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_beta.tolist()
                for j in vir_beta.tolist()
            ]
        ).T[1],
    )

    occ_alpha = np.array([0])
    occ_beta = np.array([0])
    vir_beta = np.array([1, 2, 3, 4, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_ab(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one(
                            [i + 6], [j + 6], 0, x, y, occ_alpha, occ_beta[occ_beta != i]
                        )
                    )
                    * slater.sign_excite(0b000001000001, [i + 6], [j + 6])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_beta.tolist()
                for j in vir_beta.tolist()
            ]
        ).T[1],
    )


def test_integrate_sd_sds_deriv_one_ba():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_deriv_one_ba.

    Compared with UnrestrictedChemicalHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_ba(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one(
                            [i], [j], 1, x, y, occ_alpha[occ_alpha != i], occ_beta
                        )
                    )
                    * slater.sign_excite(0b101101011001, [i], [j])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_alpha.tolist()
                for j in vir_alpha.tolist()
            ]
        ).T[1],
    )

    occ_alpha = np.array([0])
    occ_beta = np.array([0])
    vir_alpha = np.array([1, 2, 3, 4, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_ba(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one(
                            [i], [j], 1, x, y, occ_alpha[occ_alpha != i], occ_beta
                        )
                    )
                    * slater.sign_excite(0b000001000001, [i], [j])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_alpha.tolist()
                for j in vir_alpha.tolist()
            ]
        ).T[1],
    )


def test_integrate_sd_sds_deriv_one_bb():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_deriv_one_bb.

    Compared with UnrestrictedChemicalHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_bb(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one(
                            [i + 6], [j + 6], 1, x, y, occ_alpha, occ_beta[occ_beta != i]
                        )
                    )
                    * slater.sign_excite(0b101101011001, [i + 6], [j + 6])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_beta.tolist()
                for j in vir_beta.tolist()
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_two_aaa():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_deriv_two_aaa.

    Compared with UnrestrictedChemicalHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_aaa(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, 0, x, y))
                    * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for diff1 in it.combinations(occ_alpha.tolist(), 2)
                for diff2 in it.combinations(vir_alpha.tolist(), 2)
            ]
        ).T[[1, 2]],
    )


def test_integrate_sd_sds_deriv_two_aab():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_deriv_two_aab.

    Compared with UnrestrictedChemicalHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_aab(occ_alpha, occ_beta, vir_alpha, vir_beta),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, 0, x, y))
                    * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for diff1 in it.product(occ_alpha.tolist(), (occ_beta + 6).tolist())
                for diff2 in it.product(vir_alpha.tolist(), (vir_beta + 6).tolist())
            ]
        ).T[1],
    )


def test_integrate_sd_sds_deriv_two_bab():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_deriv_two_bab.

    Compared with UnrestrictedChemicalHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_bab(occ_alpha, occ_beta, vir_alpha, vir_beta),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, 1, x, y))
                    * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for diff1 in it.product(occ_alpha.tolist(), (occ_beta + 6).tolist())
                for diff2 in it.product(vir_alpha.tolist(), (vir_beta + 6).tolist())
            ]
        ).T[1],
    )


def test_integrate_sd_sds_deriv_two_bbb():
    """Test UnrestrictedChemicalHamiltonian._integrate_sd_sds_deriv_two_bbb.

    Compared with UnrestrictedChemicalHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_bbb(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, 1, x, y))
                    * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for diff1 in it.combinations((occ_beta + 6).tolist(), 2)
                for diff2 in it.combinations((vir_beta + 6).tolist(), 2)
            ]
        ).T[[1, 2]],
    )


def test_integrate_sd_wfn():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_wfn with integrate_wfn_sd."""
    one_int_a = np.random.rand(5, 5)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(5, 5)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(5, 5, 5, 5)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(5, 5, 5, 5)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(5, 5, 5, 5)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    for i in range(1, 4):
        wfn = CIWavefunction(i, 10)
        wfn.assign_params(np.random.rand(*wfn.params_shape))
        for occ_indices in it.combinations(range(10), i):
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, wfn_deriv=None),
                test_ham.integrate_wfn_sd(wfn, slater.create(0, *occ_indices), wfn_deriv=None),
            )
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, wfn_deriv=0),
                test_ham.integrate_wfn_sd(wfn, slater.create(0, *occ_indices), wfn_deriv=0),
            )


def test_integrate_sd_wfn_deriv():
    """Test UnrestrictedChemicalHamiltonian.integrate_sd_wfn_deriv with integrate_wfn_sd."""
    one_int_a = np.random.rand(5, 5)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(5, 5)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(5, 5, 5, 5)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(5, 5, 5, 5)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(5, 5, 5, 5)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedChemicalHamiltonian(
        [one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb]
    )

    wfn = CIWavefunction(4, 10)
    wfn.assign_params(np.random.rand(*wfn.params_shape))
    assert np.allclose(
        test_ham.integrate_sd_wfn_deriv(0b0001101010, wfn, np.arange(20)),
        np.array([test_ham.integrate_wfn_sd(wfn, 0b0001101010, ham_deriv=i) for i in range(20)]).T,
    )

    ham_derivs = np.array([0, 3, 5, 7, 8, 11, 13])
    for i in range(1, 4):
        wfn = CIWavefunction(i, 10)
        wfn.assign_params(np.random.rand(*wfn.params_shape))
        for occ_indices in it.combinations(range(10), i):
            assert np.allclose(
                test_ham.integrate_sd_wfn_deriv(slater.create(0, *occ_indices), wfn, ham_derivs),
                np.array(
                    [
                        test_ham.integrate_wfn_sd(wfn, slater.create(0, *occ_indices), ham_deriv=i)
                        for i in ham_derivs.tolist()
                    ]
                ).T,
            )
