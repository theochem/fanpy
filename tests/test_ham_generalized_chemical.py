"""Test wfns.ham.generalized_chemical."""
import numpy as np
from nose.plugins.attrib import attr
from nose.tools import assert_raises
from wfns.ham.generalized_chemical import GeneralizedChemicalHamiltonian
from wfns.wfn.ci.base import CIWavefunction
from wfns.tools import find_datafile
from wfns.backend.sd_list import sd_list


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
    assert_raises(ValueError, test.assign_params, [0])
    assert_raises(ValueError, test.assign_params, np.array([[0]]))
    assert_raises(ValueError, test.assign_params, np.array([0, 1]))

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
    assert coulomb == 1*9 + 2*16
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_wfn_sd(test_wfn, 0b1100)
    assert one_energy == 1*3 + 3*4
    assert coulomb == 3*10
    assert exchange == -3*11

    assert_raises(ValueError, test_ham.integrate_wfn_sd, test_wfn, 0b0101, wfn_deriv=0, ham_deriv=0)


def test_integrate_sd_sd_trivial():
    """Test GeneralizedChemicalHamiltonian.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(6, 6)
    two_int = np.random.rand(6, 6, 6, 6)
    test = GeneralizedChemicalHamiltonian(one_int, two_int)

    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b100100, sign=0, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b100100, sign=0.5, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b100100, sign=-0.5, deriv=None)

    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b001001)
    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b111000)
    assert (0, two_int[0, 1, 2, 3], -two_int[0, 1, 3, 2]) == test.integrate_sd_sd(0b100011,
                                                                                  0b101100, sign=1)
    assert (0, -two_int[0, 1, 2, 3], two_int[0, 1, 3, 2]) == test.integrate_sd_sd(0b100011,
                                                                                  0b101100, sign=-1)
    assert (one_int[0, 0], 0, 0) == test.integrate_sd_sd(0b1, 0b1)
    assert (one_int[0, 1], 0, 0) == test.integrate_sd_sd(0b1, 0b10)
    assert ((0,
             -two_int[1, 4, 1, 3] + two_int[0, 4, 0, 3],
             two_int[1, 4, 3, 1] - two_int[0, 4, 3, 0])
            == test.integrate_sd_sd(0b110001, 0b101010, deriv=0))


def test_integrate_sd_sd_h2_631gdp():
    """Test GenrealizedChemicalHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.
    Integrals that correspond to restricted orbitals were used.

    """
    restricted_one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    restricted_two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    one_int = np.zeros((20, 20))
    one_int[:10, :10] = restricted_one_int
    one_int[10:, 10:] = restricted_one_int
    two_int = np.zeros((20, 20, 20, 20))
    two_int[:10, :10, :10, :10] = restricted_two_int
    two_int[:10, 10:, :10, 10:] = restricted_two_int
    two_int[10:, :10, 10:, :10] = restricted_two_int
    two_int[10:, 10:, 10:, 10:] = restricted_two_int

    ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    ref_ci_matrix = np.load(find_datafile('test/h2_hf_631gdp_cimatrix.npy'))
    ref_pspace = np.load(find_datafile('test/h2_hf_631gdp_civec.npy'))

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
    restricted_one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    restricted_two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    one_int = np.zeros((20, 20))
    one_int[:10, :10] = restricted_one_int
    one_int[10:, 10:] = restricted_one_int
    two_int = np.zeros((20, 20, 20, 20))
    two_int[:10, :10, :10, :10] = restricted_two_int
    two_int[:10, 10:, :10, 10:] = restricted_two_int
    two_int[10:, :10, 10:, :10] = restricted_two_int
    two_int[10:, 10:, 10:, 10:] = restricted_two_int

    ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    ref_ci_matrix = np.load(find_datafile('test/h2_hf_631gdp_cimatrix.npy'))
    ref_pspace = np.load(find_datafile('test/h2_hf_631gdp_civec.npy')).tolist()

    params = np.random.rand(len(ref_pspace))
    wfn = CIWavefunction(2, 10, sd_vec=ref_pspace, params=params)
    for i, sd in enumerate(ref_pspace):
        assert np.allclose(sum(ham.integrate_wfn_sd(wfn, sd)), ref_ci_matrix[i, :].dot(params))
        assert np.allclose(sum(ham.integrate_wfn_sd(wfn, sd)),
                           sum(sum(ham.integrate_sd_sd(sd, sd1)) * wfn.get_overlap(sd1)
                               for sd1 in ref_pspace))


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

    restricted_one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    restricted_two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
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
        assert np.allclose(ham.integrate_wfn_sd(wfn, sd)[0],
                           sum(ham.integrate_sd_sd(sd, sd1)[0] * wfn.get_overlap(sd1)
                               for sd1 in sds))
        assert np.allclose(ham.integrate_wfn_sd(wfn, sd)[1],
                           sum(ham.integrate_sd_sd(sd, sd1)[1] * wfn.get_overlap(sd1)
                               for sd1 in sds))
        assert np.allclose(ham.integrate_wfn_sd(wfn, sd)[2],
                           sum(ham.integrate_sd_sd(sd, sd1)[2] * wfn.get_overlap(sd1)
                               for sd1 in sds))


@attr('slow')
def test_integrate_sd_sd_lih_631g():
    """Test GeneralizedChemicalHamiltonian.integrate_sd_sd using LiH HF/6-31G orbitals.

    Integrals that correspond to restricted orbitals were used.

    """
    restricted_one_int = np.load(find_datafile('test/lih_hf_631g_oneint.npy'))
    restricted_two_int = np.load(find_datafile('test/lih_hf_631g_twoint.npy'))
    one_int = np.zeros((22, 22))
    one_int[:11, :11] = restricted_one_int
    one_int[11:, 11:] = restricted_one_int
    two_int = np.zeros((22, 22, 22, 22))
    two_int[:11, :11, :11, :11] = restricted_two_int
    two_int[:11, 11:, :11, 11:] = restricted_two_int
    two_int[11:, :11, 11:, :11] = restricted_two_int
    two_int[11:, 11:, 11:, 11:] = restricted_two_int

    ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    ref_ci_matrix = np.load(find_datafile('test/lih_hf_631g_cimatrix.npy'))
    ref_pspace = np.load(find_datafile('test/lih_hf_631g_civec.npy'))

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
            for col_ind in range(row_ind+1, n):
                param_ind = row_ind * n - row_ind*(row_ind+1)/2 + col_ind - row_ind - 1
                assert ham.param_ind_to_rowcol_ind(param_ind) == (row_ind, col_ind)


def test_integrate_sd_sd_deriv():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sd_deriv for trivial cases."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)

    assert_raises(ValueError, test_ham._integrate_sd_sd_deriv, 0b0101, 0b0101, 0.0)
    assert_raises(ValueError, test_ham._integrate_sd_sd_deriv, 0b0101, 0b0101, -1)
    assert_raises(ValueError, test_ham._integrate_sd_sd_deriv, 0b0101, 0b0101, 2)
    assert test_ham._integrate_sd_sd_deriv(0b0101, 0b0001, 0) == (0, 0, 0)
    assert test_ham._integrate_sd_sd_deriv(0b000111, 0b111000, 0) == (0, 0, 0)


def test_integrate_sd_sd_deriv_fdiff_h2_sto6g():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sd_deriv using H2/STO6G.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    restricted_one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    restricted_two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
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

                finite_diff = (np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                               - np.array(test_ham.integrate_sd_sd(sd1, sd2))) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20*epsilon)


@attr('slow')
def test_integrate_sd_sd_deriv_fdiff_h4_sto6g():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sd_deriv using H4-STO6G integrals.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    restricted_one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    restricted_two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
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

                finite_diff = (np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                               - np.array(test_ham.integrate_sd_sd(sd1, sd2))) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20*epsilon)


def test_integrate_sd_sd_deriv_fdiff_random():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sd_deriv using random integrals.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T
    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum('ijkl->jilk', two_int) + two_int
    two_int = np.einsum('ijkl->klij', two_int) + two_int

    # check that the integrals have the appropriate symmetry
    assert np.allclose(one_int, one_int.T)
    assert np.allclose(two_int, np.einsum('ijkl->jilk', two_int))
    assert np.allclose(two_int, np.einsum('ijkl->klij', two_int))

    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)
    epsilon = 1e-8
    sds = sd_list(3, 3, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = GeneralizedChemicalHamiltonian(one_int, two_int, params=addition)

                finite_diff = (np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                               - np.array(test_ham.integrate_sd_sd(sd1, sd2))) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20*epsilon)


def test_integrate_sd_sd_deriv_fdiff_random_small():
    """Test GeneralizedChemicalHamiltonian._integrate_sd_sd_deriv using random 1e system.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.random.rand(2, 2)
    one_int = one_int + one_int.T
    two_int = np.random.rand(2, 2, 2, 2)
    two_int = np.einsum('ijkl->jilk', two_int) + two_int
    two_int = np.einsum('ijkl->klij', two_int) + two_int

    # check that the integrals have the appropriate symmetry
    assert np.allclose(one_int, one_int.T)
    assert np.allclose(two_int, np.einsum('ijkl->jilk', two_int))
    assert np.allclose(two_int, np.einsum('ijkl->klij', two_int))

    test_ham = GeneralizedChemicalHamiltonian(one_int, two_int)
    epsilon = 1e-8
    sds = sd_list(1, 1, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = GeneralizedChemicalHamiltonian(one_int, two_int, params=addition)

                finite_diff = (np.array(test_ham2.integrate_sd_sd(sd1, sd2))
                               - np.array(test_ham.integrate_sd_sd(sd1, sd2))) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, i)
                assert np.allclose(finite_diff, derivative, atol=20*epsilon)
