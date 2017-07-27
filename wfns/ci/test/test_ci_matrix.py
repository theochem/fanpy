""" Tests wfns.ci.ci_matrix
"""
import numpy as np
from nose.plugins.attrib import attr
from nose.tools import assert_raises
import wfns.ci.ci_matrix as ci_matrix
from wfns.tools import find_datafile


def test_get_one_int_value():
    """
    Tests ci_matrix.get_one_int_value
    """
    one_int = (np.arange(16).reshape(4, 4),)
    # check errors
    assert_raises(ValueError, lambda: ci_matrix.get_one_int_value(one_int, -1, 0, 'restricted'))
    assert_raises(ValueError, lambda: ci_matrix.get_one_int_value(one_int*2, 0, -1, 'unrestricted'))
    assert_raises(ValueError, lambda: ci_matrix.get_one_int_value(one_int, 0, -1, 'generalized'))
    assert_raises(ValueError, lambda: ci_matrix.get_one_int_value(one_int, 8, 0, 'restricted'))
    assert_raises(ValueError, lambda: ci_matrix.get_one_int_value(one_int*2, 0, 8, 'unrestricted'))
    assert_raises(ValueError, lambda: ci_matrix.get_one_int_value(one_int, 0, 4, 'generalized'))
    assert_raises(TypeError, lambda: ci_matrix.get_one_int_value(one_int, 0, 0, 'random type'))
    # restricted
    assert ci_matrix.get_one_int_value(one_int, 0, 0, 'restricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 0, 1, 'restricted') == 1.0
    assert ci_matrix.get_one_int_value(one_int, 0, 4, 'restricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 0, 5, 'restricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 1, 0, 'restricted') == 4.0
    assert ci_matrix.get_one_int_value(one_int, 1, 1, 'restricted') == 5.0
    assert ci_matrix.get_one_int_value(one_int, 1, 4, 'restricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 1, 5, 'restricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 4, 0, 'restricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 4, 1, 'restricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 4, 4, 'restricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 4, 5, 'restricted') == 1.0
    assert ci_matrix.get_one_int_value(one_int, 5, 0, 'restricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 5, 1, 'restricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 5, 4, 'restricted') == 4.0
    assert ci_matrix.get_one_int_value(one_int, 5, 5, 'restricted') == 5.0
    # unrestricted
    one_int = (np.arange(16).reshape(4, 4), np.arange(16, 32).reshape(4, 4))
    assert ci_matrix.get_one_int_value(one_int, 0, 0, 'unrestricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 0, 1, 'unrestricted') == 1.0
    assert ci_matrix.get_one_int_value(one_int, 0, 4, 'unrestricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 1, 0, 'unrestricted') == 4.0
    assert ci_matrix.get_one_int_value(one_int, 1, 1, 'unrestricted') == 5.0
    assert ci_matrix.get_one_int_value(one_int, 1, 4, 'unrestricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 1, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 4, 0, 'unrestricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 4, 4, 'unrestricted') == 16.0
    assert ci_matrix.get_one_int_value(one_int, 4, 5, 'unrestricted') == 17.0
    assert ci_matrix.get_one_int_value(one_int, 5, 0, 'unrestricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 5, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 5, 4, 'unrestricted') == 20.0
    assert ci_matrix.get_one_int_value(one_int, 5, 5, 'unrestricted') == 21.0
    # generalized
    one_int = (np.arange(64).reshape(8, 8),)
    assert ci_matrix.get_one_int_value(one_int, 0, 0, 'generalized') == 0.0
    assert ci_matrix.get_one_int_value(one_int, 0, 1, 'generalized') == 1.0
    assert ci_matrix.get_one_int_value(one_int, 0, 4, 'generalized') == 4.0
    assert ci_matrix.get_one_int_value(one_int, 0, 5, 'generalized') == 5.0
    assert ci_matrix.get_one_int_value(one_int, 1, 0, 'generalized') == 8.0
    assert ci_matrix.get_one_int_value(one_int, 1, 1, 'generalized') == 9.0
    assert ci_matrix.get_one_int_value(one_int, 1, 4, 'generalized') == 12.0
    assert ci_matrix.get_one_int_value(one_int, 1, 5, 'generalized') == 13.0
    assert ci_matrix.get_one_int_value(one_int, 4, 0, 'generalized') == 32.0
    assert ci_matrix.get_one_int_value(one_int, 4, 1, 'generalized') == 33.0
    assert ci_matrix.get_one_int_value(one_int, 4, 4, 'generalized') == 36.0
    assert ci_matrix.get_one_int_value(one_int, 4, 5, 'generalized') == 37.0
    assert ci_matrix.get_one_int_value(one_int, 5, 0, 'generalized') == 40.0
    assert ci_matrix.get_one_int_value(one_int, 5, 1, 'generalized') == 41.0
    assert ci_matrix.get_one_int_value(one_int, 5, 4, 'generalized') == 44.0
    assert ci_matrix.get_one_int_value(one_int, 5, 5, 'generalized') == 45.0


def test_get_two_int_value():
    """
    Tests ci_matrix.get_two_int_value
    """
    two_int = (np.arange(256).reshape(4, 4, 4, 4),)
    # check errors
    assert_raises(ValueError, lambda: ci_matrix.get_two_int_value(two_int, -1, 0, 0, 0, 'restricted'))
    assert_raises(ValueError, lambda: ci_matrix.get_two_int_value(two_int*3, 0, -1, 0, 0, 'unrestricted'))
    assert_raises(ValueError, lambda: ci_matrix.get_two_int_value(two_int, 0, 0, -1, 0, 'generalized'))
    assert_raises(ValueError, lambda: ci_matrix.get_two_int_value(two_int, 8, 0, 0, 0, 'restricted'))
    assert_raises(ValueError, lambda: ci_matrix.get_two_int_value(two_int*3, 0, 8, 0, 0, 'unrestricted'))
    assert_raises(ValueError, lambda: ci_matrix.get_two_int_value(two_int, 0, 0, 4, 0, 'generalized'))
    assert_raises(TypeError, lambda: ci_matrix.get_two_int_value(two_int, 0, 0, 0, 0, 'random type'))
    # restricted
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 0, 1, 'restricted') == 1.0
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 4, 1, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 0, 1, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 0, 1, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 4, 1, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 4, 1, 'restricted') == 1.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 0, 1, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 4, 1, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 0, 5, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 4, 5, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 0, 5, 'restricted') == 1.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 0, 5, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 4, 5, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 4, 5, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 0, 5, 'restricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 4, 5, 'restricted') == 1.0
    # unrestricted
    two_int = (np.arange(256).reshape(4, 4, 4, 4),
         np.arange(256, 512).reshape(4, 4, 4, 4),
         np.arange(512, 768).reshape(4, 4, 4, 4))
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 0, 1, 'unrestricted') == 1.0
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 0, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 0, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 4, 1, 'unrestricted') == 260.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 0, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 4, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 0, 5, 'unrestricted') == 257.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 4, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 4, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 4, 5, 'unrestricted') == 513.0
    # generalized
    two_int = (np.arange(4096).reshape(8, 8, 8, 8),)
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 0, 1, 'generalized') == 1.0
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 4, 1, 'generalized') == 33.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 0, 1, 'generalized') == 257.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 0, 1, 'generalized') == 2049.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 4, 1, 'generalized') == 289.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 4, 1, 'generalized') == 2081.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 0, 1, 'generalized') == 2305.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 4, 1, 'generalized') == 2337.0
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 0, 5, 'generalized') == 5.0
    assert ci_matrix.get_two_int_value(two_int, 0, 0, 4, 5, 'generalized') == 37.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 0, 5, 'generalized') == 261.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 0, 5, 'generalized') == 2053.0
    assert ci_matrix.get_two_int_value(two_int, 0, 4, 4, 5, 'generalized') == 293.0
    assert ci_matrix.get_two_int_value(two_int, 4, 0, 4, 5, 'generalized') == 2085.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 0, 5, 'generalized') == 2309.0
    assert ci_matrix.get_two_int_value(two_int, 4, 4, 4, 5, 'generalized') == 2341.0


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
