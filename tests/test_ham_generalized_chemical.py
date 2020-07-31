"""Test wfns.ham.generalized_chemical."""
import itertools as it

import numpy as np
import pytest
from utils import find_datafile
from wfns.backend import slater
from wfns.backend.sd_list import sd_list
from wfns.ham.generalized_chemical import GeneralizedChemicalHamiltonian
from wfns.wfn.ci.base import CIWavefunction


def test_init():
    """Test GeneralizedChemicalHamiltonian.__init__."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = GeneralizedChemicalHamiltonian(one_int, two_int)
    assert np.allclose(test.params, np.zeros(6))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)


def test_set_ref_ints():
    """Test GeneralizedChemicalHamiltonian.set_ref_ints."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = GeneralizedChemicalHamiltonian(one_int, two_int)
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    new_one_int = np.random.rand(2, 2)
    new_two_int = np.random.rand(2, 2, 2, 2)
    test.assign_integrals(new_one_int, new_two_int)
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    test.set_ref_ints()
    assert np.allclose(test._ref_one_int, new_one_int)
    assert np.allclose(test._ref_two_int, new_two_int)


def test_cache_two_ints():
    """Test GeneralizedChemicalHamiltonian.cache_two_ints."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    two_int_ijij = np.array([[5, 10], [15, 20]])
    two_int_ijji = np.array([[5, 11], [14, 20]])

    test = GeneralizedChemicalHamiltonian(one_int, two_int)
    assert np.allclose(test._cached_two_int_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_ijji, two_int_ijji)

    test.two_int = np.arange(21, 37).reshape(2, 2, 2, 2)
    new_two_int_ijij = np.array([[21, 26], [31, 36]])
    new_two_int_ijji = np.array([[21, 27], [30, 36]])
    assert np.allclose(test._cached_two_int_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_ijji, two_int_ijji)

    test.cache_two_ints()
    assert np.allclose(test._cached_two_int_ijij, new_two_int_ijij)
    assert np.allclose(test._cached_two_int_ijji, new_two_int_ijji)


def test_assign_params():
    """Test GeneralizedChemicalHamiltonian.assign_params."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)

    test = GeneralizedChemicalHamiltonian(one_int, two_int)
    with pytest.raises(ValueError):
        test.assign_params([0])
    with pytest.raises(ValueError):
        test.assign_params(np.array([[0]]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([0, 1]))

    test.assign_params(np.array([0]))
    assert np.allclose(test.params, np.zeros(1))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    test.assign_params(np.array([10]))
    assert np.allclose(test.params, 10)
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    test.assign_params(np.array([5]))
    assert np.allclose(test.params, 5)
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    # make sure that transformation is independent of the transformations that came before it
    test.assign_params(np.array([10]))
    assert np.allclose(test.params, 10)
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)


def test_integrate_wfn_sd():
    """Test GeneralizedChemicalHamiltonian.integrate_wfn_sd.

    Integrals that correspond to restricted orbitals were used.

    """
    restricted_one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    restricted_two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    one_int = np.zeros((4, 4))
    one_int[:2, :2] = restricted_one_int
    one_int[2:, 2:] = restricted_one_int
    two_int = np.zeros((4, 4, 4, 4))
    two_int[:2, :2, :2, :2] = restricted_two_int
    two_int[:2, 2:, :2, 2:] = restricted_two_int
    two_int[2:, :2, 2:, :2] = restricted_two_int
    two_int[2:, 2:, 2:, 2:] = restricted_two_int

    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)
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
    assert coulomb == 1 * 9 + 2 * 16
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_wfn_sd(test_wfn, 0b1100)
    assert one_energy == 1 * 3 + 3 * 4
    assert coulomb == 3 * 10
    assert exchange == -3 * 11

    with pytest.raises(ValueError):
        test_ham.integrate_wfn_sd(test_wfn, 0b0101, wfn_deriv=0, ham_deriv=0)


def test_integrate_sd_sd_trivial():
    """Test GeneralizedChemicalHamiltonian.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(6, 6)
    two_int = np.random.rand(6, 6, 6, 6)
    test = GeneralizedChemicalHamiltonian(one_int, two_int)

    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b001001)
    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b111000)
    assert (0, two_int[0, 1, 2, 3], -two_int[0, 1, 3, 2]) == test.integrate_sd_sd(
        0b100011, 0b101100
    )
    assert (one_int[0, 0], 0, 0) == test.integrate_sd_sd(0b1, 0b1)
    assert (one_int[0, 1], 0, 0) == test.integrate_sd_sd(0b1, 0b10)
    assert (
        0,
        -two_int[1, 4, 1, 3] + two_int[0, 4, 0, 3],
        two_int[1, 4, 3, 1] - two_int[0, 4, 3, 0],
    ) == test.integrate_sd_sd(0b110001, 0b101010, deriv=0)


def test_integrate_sd_sd_h2_631gdp():
    """Test GenrealizedChemicalHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.
    Integrals that correspond to restricted orbitals were used.

    """
    restricted_one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    one_int = np.zeros((20, 20))
    one_int[:10, :10] = restricted_one_int
    one_int[10:, 10:] = restricted_one_int
    two_int = np.zeros((20, 20, 20, 20))
    two_int[:10, :10, :10, :10] = restricted_two_int
    two_int[:10, 10:, :10, 10:] = restricted_two_int
    two_int[10:, :10, 10:, :10] = restricted_two_int
    two_int[10:, 10:, 10:, 10:] = restricted_two_int

    ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    ref_ci_matrix = np.load(find_datafile("data_h2_hf_631gdp_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("data_h2_hf_631gdp_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose(sum(ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_wfn_sd_h2_631gdp():
    """Test GeneralizedChemicalHamiltonian.integrate_wfn_sd using H2 HF/6-31G** orbitals.

    Compare projected energy with the transformed CI matrix from PySCF.
    Compare projected energy with the transformed integrate_sd_sd.
    Integrals that correspond to restricted orbitals were used.

    """
    restricted_one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    one_int = np.zeros((20, 20))
    one_int[:10, :10] = restricted_one_int
    one_int[10:, 10:] = restricted_one_int
    two_int = np.zeros((20, 20, 20, 20))
    two_int[:10, :10, :10, :10] = restricted_two_int
    two_int[:10, 10:, :10, 10:] = restricted_two_int
    two_int[10:, :10, 10:, :10] = restricted_two_int
    two_int[10:, 10:, 10:, 10:] = restricted_two_int

    ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    ref_ci_matrix = np.load(find_datafile("data_h2_hf_631gdp_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("data_h2_hf_631gdp_civec.npy")).tolist()

    params = np.random.rand(len(ref_pspace))
    wfn = CIWavefunction(2, 10, sd_vec=ref_pspace, params=params)
    for i, sd in enumerate(ref_pspace):
        assert np.allclose(sum(ham.integrate_wfn_sd(wfn, sd)), ref_ci_matrix[i, :].dot(params))
        assert np.allclose(
            sum(ham.integrate_wfn_sd(wfn, sd)),
            sum(sum(ham.integrate_sd_sd(sd, sd1)) * wfn.get_overlap(sd1) for sd1 in ref_pspace),
        )


def test_integrate_wfn_sd_h4_sto6g():
    """Test GeneralizedChemicalHamiltonian.integrate_wfn_sd using H4 HF/STO6G orbitals.

    Compare projected energy with the transformed integrate_sd_sd.
    Integrals that correspond to restricted orbitals were used.

    """
    nelec = 4
    nspin = 8
    sds = sd_list(4, 4, num_limit=None, exc_orders=None)
    wfn = CIWavefunction(nelec, nspin, sd_vec=sds)
    np.random.seed(1000)
    wfn.assign_params(np.random.rand(len(sds)))

    restricted_one_int = np.load(find_datafile("data_h4_square_hf_sto6g_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_h4_square_hf_sto6g_twoint.npy"))
    one_int = np.zeros((8, 8))
    one_int[:4, :4] = restricted_one_int
    one_int[4:, 4:] = restricted_one_int
    two_int = np.zeros((8, 8, 8, 8))
    two_int[:4, :4, :4, :4] = restricted_two_int
    two_int[:4, 4:, :4, 4:] = restricted_two_int
    two_int[4:, :4, 4:, :4] = restricted_two_int
    two_int[4:, 4:, 4:, 4:] = restricted_two_int

    ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    for sd in sds:
        assert np.allclose(
            ham.integrate_wfn_sd(wfn, sd)[0],
            sum(ham.integrate_sd_sd(sd, sd1)[0] * wfn.get_overlap(sd1) for sd1 in sds),
        )
        assert np.allclose(
            ham.integrate_wfn_sd(wfn, sd)[1],
            sum(ham.integrate_sd_sd(sd, sd1)[1] * wfn.get_overlap(sd1) for sd1 in sds),
        )
        assert np.allclose(
            ham.integrate_wfn_sd(wfn, sd)[2],
            sum(ham.integrate_sd_sd(sd, sd1)[2] * wfn.get_overlap(sd1) for sd1 in sds),
        )


def test_integrate_sd_sd_lih_631g_trial_slow():
    """Test GeneralizedChemicalHamiltonian.integrate_sd_sd using LiH HF/6-31G orbitals.

    Integrals that correspond to restricted orbitals were used.

    """
    restricted_one_int = np.load(find_datafile("data_lih_hf_631g_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_lih_hf_631g_twoint.npy"))
    one_int = np.zeros((22, 22))
    one_int[:11, :11] = restricted_one_int
    one_int[11:, 11:] = restricted_one_int
    two_int = np.zeros((22, 22, 22, 22))
    two_int[:11, :11, :11, :11] = restricted_two_int
    two_int[:11, 11:, :11, 11:] = restricted_two_int
    two_int[11:, :11, 11:, :11] = restricted_two_int
    two_int[11:, 11:, 11:, 11:] = restricted_two_int

    ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    ref_ci_matrix = np.load(find_datafile("data_lih_hf_631g_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("data_lih_hf_631g_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose(sum(ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_sd_sd_particlenum():
    """Test GeneralizedChemicalHamiltonian.integrate_sd_sd and break particle number symmetery."""
    restricted_one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    restricted_two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)
    one_int = np.zeros((8, 8))
    one_int[:4, :4] = restricted_one_int
    one_int[4:, 4:] = restricted_one_int
    two_int = np.zeros((8, 8, 8, 8))
    two_int[:4, :4, :4, :4] = restricted_two_int
    two_int[:4, 4:, :4, 4:] = restricted_two_int
    two_int[4:, :4, 4:, :4] = restricted_two_int
    two_int[4:, 4:, 4:, 4:] = restricted_two_int

    ham = GeneralizedChemicalHamiltonian(one_int, two_int)
    civec = [0b01, 0b11]

    # \braket{1 | h_{11} | 1}
    assert np.allclose(sum(ham.integrate_sd_sd(civec[0], civec[0])), 1)
    # \braket{12 | H | 1} = 0
    assert np.allclose(sum(ham.integrate_sd_sd(civec[1], civec[0])), 0)
    assert np.allclose(sum(ham.integrate_sd_sd(civec[0], civec[1])), 0)
    # \braket{12 | h_{11} + h_{22} + g_{1212} - g_{1221} | 12}
    assert np.allclose(sum(ham.integrate_sd_sd(civec[1], civec[1])), 4)


def test_param_ind_to_rowcol_ind():
    """Test GeneralizedChemicalHamiltonian.param_ind_to_rowcol_ind."""
    for n in range(1, 40):
        ham = GeneralizedChemicalHamiltonian(np.random.rand(n, n), np.random.rand(n, n, n, n))
        for row_ind in range(n):
            for col_ind in range(row_ind + 1, n):
                param_ind = row_ind * n - row_ind * (row_ind + 1) / 2 + col_ind - row_ind - 1
                assert ham.param_ind_to_rowcol_ind(param_ind) == (row_ind, col_ind)


def test_integrate_sd_sd_deriv():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sd_deriv for trivial cases."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, 0.0)
    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, -1)
    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, 2)
    assert test_ham._integrate_sd_sd_deriv(0b0101, 0b0001, 0) == (0, 0, 0)
    assert test_ham._integrate_sd_sd_deriv(0b000111, 0b111000, 0) == (0, 0, 0)


def test_integrate_sd_sd_deriv_fdiff_h2_sto6g():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sd_deriv using H2/STO6G.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    restricted_one_int = np.load(find_datafile("data_h4_square_hf_sto6g_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_h4_square_hf_sto6g_twoint.npy"))
    one_int = np.zeros((8, 8))
    one_int[:4, :4] = restricted_one_int
    one_int[4:, 4:] = restricted_one_int
    two_int = np.zeros((8, 8, 8, 8))
    two_int[:4, :4, :4, :4] = restricted_two_int
    two_int[:4, 4:, :4, 4:] = restricted_two_int
    two_int[4:, :4, 4:, :4] = restricted_two_int
    two_int[4:, 4:, 4:, 4:] = restricted_two_int

    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)
    epsilon = 1e-8

    for sd1 in [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]:
        for sd2 in [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = GeneralizedChemicalHamiltonian(one_int, two_int, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)


def test_integrate_sd_sd_deriv_fdiff_h4_sto6g_trial_slow():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sd_deriv using H4-STO6G integrals.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    restricted_one_int = np.load(find_datafile("data_h4_square_hf_sto6g_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_h4_square_hf_sto6g_twoint.npy"))
    one_int = np.zeros((8, 8))
    one_int[:4, :4] = restricted_one_int
    one_int[4:, 4:] = restricted_one_int
    two_int = np.zeros((8, 8, 8, 8))
    two_int[:4, :4, :4, :4] = restricted_two_int
    two_int[:4, 4:, :4, 4:] = restricted_two_int
    two_int[4:, :4, 4:, :4] = restricted_two_int
    two_int[4:, 4:, 4:, 4:] = restricted_two_int

    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)
    epsilon = 1e-8

    sds = sd_list(4, 4, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = GeneralizedChemicalHamiltonian(one_int, two_int, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)


def test_integrate_sd_sd_deriv_fdiff_random():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sd_deriv using random integrals.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T
    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    # check that the integrals have the appropriate symmetry
    assert np.allclose(one_int, one_int.T)
    assert np.allclose(two_int, np.einsum("ijkl->jilk", two_int))
    assert np.allclose(two_int, np.einsum("ijkl->klij", two_int))

    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)
    epsilon = 1e-8
    sds = sd_list(3, 3, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = GeneralizedChemicalHamiltonian(one_int, two_int, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)


def test_integrate_sd_sd_deriv_fdiff_random_small():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sd_deriv using random 1e system.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.random.rand(2, 2)
    one_int = one_int + one_int.T
    two_int = np.random.rand(2, 2, 2, 2)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    # check that the integrals have the appropriate symmetry
    assert np.allclose(one_int, one_int.T)
    assert np.allclose(two_int, np.einsum("ijkl->jilk", two_int))
    assert np.allclose(two_int, np.einsum("ijkl->klij", two_int))

    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)
    epsilon = 1e-8
    sds = sd_list(1, 1, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = GeneralizedChemicalHamiltonian(one_int, two_int, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)


def test_integrate_sd_sds_zero():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sds_zero against _integrate_sd_sd_zero."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_zero(occ_indices),
        np.array(test_ham._integrate_sd_sd_zero(occ_indices)).reshape(3, 1),
    )


def test_integrate_sd_sds_one():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sds_one against _integrate_sd_sd_one."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    vir_indices = np.array([2, 5, 6, 7])
    assert np.allclose(
        test_ham._integrate_sd_sds_one(occ_indices, vir_indices),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_one((i,), (j,), occ_indices[occ_indices != i]))
                * slater.sign_excite(0b11011, [i], [j])
                for i in occ_indices.tolist()
                for j in vir_indices.tolist()
            ]
        ).T,
    )


def test_integrate_sd_sds_two():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sds_two against _integrate_sd_sd_two."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    vir_indices = np.array([2, 5, 6, 7])
    assert np.allclose(
        test_ham._integrate_sd_sds_two(occ_indices, vir_indices),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_two(diff1, diff2))
                * slater.sign_excite(0b11011, diff1, reversed(diff2))
                for diff1 in it.combinations(occ_indices.tolist(), 2)
                for diff2 in it.combinations(vir_indices.tolist(), 2)
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_zero():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sds_deriv_zero w/ _integrate_sd_sd_zero."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    vir_indices = np.array([2, 5, 6, 7])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_zero(occ_indices, vir_indices),
        np.array(
            [
                [
                    test_ham._integrate_sd_sd_deriv_zero(i, j, occ_indices)
                    for i in range(7)
                    for j in range(i + 1, 8)
                ]
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_one():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sds_deriv_one with _integrate_sd_sd_one."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    vir_indices = np.array([2, 5, 6, 7])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one(occ_indices, vir_indices),
        np.transpose(
            np.array(
                [
                    [
                        np.array(
                            test_ham._integrate_sd_sd_deriv_one(
                                (i,), (j,), x, y, occ_indices[occ_indices != i]
                            )
                        )
                        * slater.sign_excite(0b11011, [i], [j])
                        for x in range(8)
                        for y in range(x + 1, 8)
                    ]
                    for i in occ_indices.tolist()
                    for j in vir_indices.tolist()
                ]
            )
        ),
    )


def test_integrate_sd_sds_deriv_two():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sds_deriv_two with _integrate_sd_sd_two."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    vir_indices = np.array([2, 5, 6, 7])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two(occ_indices, vir_indices),
        np.transpose(
            np.array(
                [
                    [
                        np.array(test_ham._integrate_sd_sd_deriv_two(occ, vir, x, y))
                        * slater.sign_excite(0b11011, occ, reversed(vir))
                        for x in (range(8))
                        for y in range(x + 1, 8)
                    ]
                    for occ in it.combinations(occ_indices.tolist(), 2)
                    for vir in it.combinations(vir_indices.tolist(), 2)
                ]
            )
        ),
    )


def test_integrate_sd_wfn():
    """Test GeneralizedChemicalHamiltonian.integrate_sd_wfn with integrate_wfn_sd."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    for n in range(1, 8):
        wfn = CIWavefunction(n, 8)
        wfn.assign_params(np.random.rand(*wfn.params.shape))
        for occ_indices in it.combinations(range(8), n):
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, wfn_deriv=None),
                test_ham.integrate_wfn_sd(wfn, slater.create(0, *occ_indices), wfn_deriv=None),
            )
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, wfn_deriv=0),
                test_ham.integrate_wfn_sd(wfn, slater.create(0, *occ_indices), wfn_deriv=0),
            )


def test_integrate_sd_wfn_deriv():
    """Test GeneralizedChemicalHamiltonian.integrate_sd_wfn_deriv with integrate_wfn_sd."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    wfn = CIWavefunction(4, 8)
    wfn.assign_params(np.random.rand(*wfn.params.shape))
    assert np.allclose(
        test_ham.integrate_sd_wfn_deriv(0b01101010, wfn, np.arange(28)),
        np.array([test_ham.integrate_wfn_sd(wfn, 0b01101010, ham_deriv=i) for i in range(28)]).T,
    )

    ham_derivs = np.array([0, 3, 5, 7, 8, 11, 13])
    for n in range(1, 8):
        wfn = CIWavefunction(n, 8)
        wfn.assign_params(np.random.rand(*wfn.params.shape))
        for occ_indices in it.combinations(range(8), n):
            assert np.allclose(
                test_ham.integrate_sd_wfn_deriv(slater.create(0, *occ_indices), wfn, ham_derivs),
                np.array(
                    [
                        test_ham.integrate_wfn_sd(wfn, slater.create(0, *occ_indices), ham_deriv=i)
                        for i in ham_derivs.tolist()
                    ]
                ).T,
            )

    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn_deriv(0b01101010, wfn, np.arange(28).tolist())
    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn_deriv(0b01101010, wfn, np.arange(28).astype(float))
    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn_deriv(0b01101010, wfn, np.arange(28).reshape(2, 14))
    with pytest.raises(ValueError):
        bad_indices = np.arange(28)
        bad_indices[0] = -1
        test_ham.integrate_sd_wfn_deriv(0b01101010, wfn, bad_indices)
    with pytest.raises(ValueError):
        bad_indices = np.arange(28)
        bad_indices[0] = 28
        test_ham.integrate_sd_wfn_deriv(0b01101010, wfn, bad_indices)
