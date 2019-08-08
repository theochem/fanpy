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

    assert_raises(NotImplementedError, test.integrate_sd_sd, 0b001001, 0b100100, sign=None, deriv=0)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b100100, sign=0, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b100100, sign=0.5, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b100100, sign=-0.5, deriv=None)

    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b001001)
    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b111000)
    assert (0, two_int[0, 1, 2, 3], -two_int[0, 1, 3, 2]) == test.integrate_sd_sd(0b100011,
                                                                                  0b101100, sign=1)
    assert (0, -two_int[0, 1, 2, 3], two_int[0, 1, 3, 2]) == test.integrate_sd_sd(0b100011,
                                                                                  0b101100, sign=-1)


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
    
    # create "random" parameters
    params = np.array([0.31631948, 0.40947866, 0.68708901, 0.70223529, 0.44619794, 0.42032208,
                       0.80263059, 0.53861757, 0.92350049, 0.06190754, 0.36758345, 0.97039877,
                       0.69910395, 0.89097558, 0.25355272, 0.72552711, 0.20349443, 0.53751182,
                       0.236943, 0.43649216, 0.79802451, 0.30840916, 0.62151663, 0.43964812,
                       0.91332371, 0.6586859, 0.65363145, 0.35845778, 0.52294157, 0.04508731,
                       0.09430966, 0.92161219, 0.0730445, 0.92941735, 0.71701514, 0.62231055,
                       0.75498971, 0.14166991, 0.7938222, 0.72540315, 0.54143928, 0.48258449,
                       0.47647761, 0.44603283, 0.81607006, 0.27435149, 0.06196201, 0.96029634,
                       0.45811813, 0.75789206, 0.6811586, 0.54177793, 0.13283249, 0.25320563,
                       0.44385559, 0.55874991, 0.81386435, 0.31782834, 0.39704509, 0.89016825,
                       0.82541621, 0.10668708, 0.15977588, 0.65121931, 0.56906421, 0.98408367,
                       0.74915901, 0.80239963, 0.22830988, 0.51871345])
    wfn.assign_params(params)

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
