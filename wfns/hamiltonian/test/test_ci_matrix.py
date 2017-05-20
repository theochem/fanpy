""" Tests wfns.ci.ci_matrix
"""
import numpy as np
from nose.plugins.attrib import attr
from nose.tools import assert_raises
import wfns.hamiltonian.ci_matrix as ci_matrix
from wfns.tools import find_datafile


def test_ci_matrix_h2():
    """Test ci_matrix.ci_matrix using H2 HF/6-31G** orbitals"""
    # read in gaussian fchk file and generate one and two electron integrals (using horton)
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    one_int = (np.load(find_datafile('test/h2_hf_631gdp_oneint.npy')), )
    two_int = (np.load(find_datafile('test/h2_hf_631gdp_twoint.npy')), )
    # reference (from pyscf)
    # ref_ci_matrix, ref_pspace = generate_fci_cimatrix(one_int[0], two_int[0], 2, is_chemist_notation=False)
    ref_ci_matrix = np.load(find_datafile('test/h2_hf_631gdp_cimatrix.npy'))
    ref_pspace = np.load(find_datafile('test/h2_hf_631gdp_civec.npy'))
    # test
    test_ci_matrix = ci_matrix.ci_matrix(one_int, two_int, civec=ref_pspace, dtype=np.float64,
                                         orbtype='restricted')
    assert np.allclose(test_ci_matrix, ref_ci_matrix)

@attr('slow')
def test_ci_matrix_lih():
    """Test ci_matrix.ci_matrix using LiH FCI ci_matrix."""
    # HORTON/Olsen Results
    # hf_dict = gaussian_fchk('test/lih_hf_631g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    one_int = (np.load(find_datafile('test/lih_hf_631g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_631g_twoint.npy')), )
    # reference (from pyscf)
    # ref_ci_matrix, ref_pspace = generate_fci_cimatrix(one_int[0], two_int[0], 4, is_chemist_notation=False)
    ref_ci_matrix = np.load(find_datafile('test/lih_hf_631g_cimatrix.npy'))
    ref_pspace = np.load(find_datafile('test/lih_hf_631g_civec.npy'))
    # test
    test_ci_matrix = ci_matrix.ci_matrix(one_int, two_int, civec=ref_pspace, dtype=np.float64,
                                         orbtype='restricted')
    assert np.allclose(test_ci_matrix, ref_ci_matrix)

    assert np.allclose(test_ci_matrix, ref_ci_matrix)

def test_ci_matrix_enum_break():
    """ Tests ci_matrix.ci_matrix while breaking particle number symmetry
    """
    one_int = (np.arange(16).reshape(4, 4) + 1,)
    two_int = (np.arange(256).reshape(4, 4, 4, 4) + 1, )
    civec = [0b01, 0b11]
    test_ci_matrix = ci_matrix.ci_matrix(one_int, two_int, civec, dtype=np.float64, orbtype='restricted')
    assert np.allclose(test_ci_matrix, np.array([[1, 0], [0, 4]]))
    # first element because \braket{1 | h_{11} | 1}
    # second and third element because they break particle number symmetry
    # last element because \braket{1 2 | h_{11} + h_{22} + g_{1212} - g_{1221} | 1 2}

# FIXME: add ci_matrix tests with unrestricted and generalized orbitals
