"""Test wfns.hamiltonian.chemical_hamiltonian."""
import numpy as np
from nose.plugins.attrib import attr
from nose.tools import assert_raises
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.tools import find_datafile
from wfns.backend.sd_list import sd_list


class TestWavefunction(object):
    """Mock wavefunction for testing."""
    def get_overlap(self, sd, deriv=None):
        if sd == 0b0101:
            return 1
        elif sd == 0b1010:
            return 2
        elif sd == 0b1100:
            return 3
        return 0


def test_integrate_wfn_sd():
    """Test ChemicalHamiltonian.integrate_wfn_sd."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    hamiltonian = ChemicalHamiltonian(one_int, two_int, 'restricted')
    test_wfn = TestWavefunction()

    one_energy, coulomb, exchange = hamiltonian.integrate_wfn_sd(test_wfn, 0b0101, deriv=None)
    assert one_energy == 1*1 + 1*1
    assert coulomb == 1*5 + 2*8
    assert exchange == 0

    one_energy, coulomb, exchange = hamiltonian.integrate_wfn_sd(test_wfn, 0b1010, deriv=None)
    assert one_energy == 2*4 + 2*4
    assert coulomb == 1*17 + 2*20
    assert exchange == 0

    one_energy, coulomb, exchange = hamiltonian.integrate_wfn_sd(test_wfn, 0b0110, deriv=None)
    assert one_energy == 1*3 + 2*2
    assert coulomb == 1*13 + 2*12
    assert exchange == 0

    one_energy, coulomb, exchange = hamiltonian.integrate_wfn_sd(test_wfn, 0b1100, deriv=None)
    assert one_energy == 1*3 + 3*4
    assert coulomb == 3*10
    assert exchange == -3*11


def test_integrate_sd_sd_h2_631gdp():
    """Test ChemicalHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result
    """
    ''' integrals are geenrated using horton wrapper
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    '''
    one_int = (np.load(find_datafile('test/h2_hf_631gdp_oneint.npy')), )
    two_int = (np.load(find_datafile('test/h2_hf_631gdp_twoint.npy')), )
    ham = ChemicalHamiltonian(one_int, two_int, 'restricted')

    ''' CI matrix is generated using PYSCF wrapper
    ref_ci_matrix, ref_pspace = generate_fci_cimatrix(one_int[0], two_int[0], 2,
                                                      is_chemist_notation=False)
    '''
    ref_ci_matrix = np.load(find_datafile('test/h2_hf_631gdp_cimatrix.npy'))
    ref_pspace = np.load(find_datafile('test/h2_hf_631gdp_civec.npy'))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose(sum(ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_wfn_sd_h2_631gdp():
    """Test ChemicalHamiltonian.integrate_wfn_sd using H2 HF/6-31G** orbitals.

    Compare projected energy with the transformed CI matrix from PySCF
    Compare projected energy with the transformed integrate_sd_sd
    """
    ''' integrals are geenrated using horton wrapper
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    '''
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    ham = ChemicalHamiltonian(one_int, two_int, 'restricted')

    ''' CI matrix is generated using PYSCF wrapper
    ref_ci_matrix, ref_pspace = generate_fci_cimatrix(one_int[0], two_int[0], 2,
                                                      is_chemist_notation=False)
    '''
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
    """Test ChemicalHamiltonian.integrate_wfn_sd using H4 HF/STO6G orbitals.

    Compare projected energy with the transformed integrate_sd_sd
    """
    nelec = 4
    nspin = 8
    sds = sd_list(4, 4, num_limit=None, exc_orders=None)
    wfn = CIWavefunction(nelec, nspin, sd_vec=sds)
    np.random.seed(1000)
    wfn.assign_params(np.random.rand(len(sds)))
    ham = ChemicalHamiltonian(np.abs(np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))),
                              np.abs(np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))),
                              orbtype='restricted')

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
    """Test ChemicalHamiltonian.integrate_sd_sd using LiH HF/6-31G orbitals."""
    ''' integrals are geenrated using horton wrapper
    hf_dict = gaussian_fchk('test/lih_hf_631g.fchk')
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    '''
    one_int = (np.load(find_datafile('test/lih_hf_631g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_631g_twoint.npy')), )
    ham = ChemicalHamiltonian(one_int, two_int, 'restricted')

    ''' CI matrix is generated using PYSCF wrapper
    ref_ci_matrix, ref_pspace = generate_fci_cimatrix(one_int[0], two_int[0], 2,
                                                      is_chemist_notation=False)
    '''
    ref_ci_matrix = np.load(find_datafile('test/lih_hf_631g_cimatrix.npy'))
    ref_pspace = np.load(find_datafile('test/lih_hf_631g_civec.npy'))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose(sum(ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_sd_sd_particlenum():
    """ Tests ChemicalHamiltonian.integrate_sd_sd and break particle number symmetery"""
    one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)
    ham = ChemicalHamiltonian(one_int, two_int, 'restricted')
    civec = [0b01, 0b11]

    # \braket{1 | h_{11} | 1}
    assert np.allclose(sum(ham.integrate_sd_sd(civec[0], civec[0])), 1)
    # \braket{12 | H | 1} = 0
    assert np.allclose(sum(ham.integrate_sd_sd(civec[1], civec[0])), 0)
    assert np.allclose(sum(ham.integrate_sd_sd(civec[0], civec[1])), 0)
    # \braket{12 | h_{11} + h_{22} + g_{1212} - g_{1221} | 12}
    assert np.allclose(sum(ham.integrate_sd_sd(civec[1], civec[1])), 4)
