""" Tests wfns.wrapper.pyscf
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from nose.tools import assert_raises
import scipy.linalg
from wfns.wrapper.pyscf import hartreefock, generate_fci_cimatrix
from wfns import __file__ as package_path


def check_data_h2_rhf_sto6g(data):
    """ Checks data for h2 rhf sto6g calculation
    """
    assert np.allclose(data['el_energy'], -1.838434259892)
    assert np.allclose(data['nuc_nuc_energy'], 0.713176830593)
    assert np.allclose(data['H'], np.array([[-1.25637540e+00, 0.0000000000000],
                                            [0.0000000000000, -4.80588203e-01]]))
    assert np.allclose(data['G'], np.array([[[[6.74316543e-01, 0.000000000000],
                                              [0.000000000000, 1.81610048e-01]],
                                             [[0.000000000000, 6.64035234e-01],
                                              [1.81610048e-01, 0.000000000000]]],
                                            [[[0.000000000000, 1.81610048e-01],
                                              [6.64035234e-01, 0.000000000000]],
                                             [[1.81610048e-01, 0.000000000000],
                                              [0.000000000000, 6.98855952e-01]]]]))


def check_data_lih_rhf_sto6g(data):
    """ Checks data for LiH rhf sto6g calculation
    """
    assert np.allclose(data['el_energy'], -7.95197153880 - 0.9953176337)
    assert np.allclose(data['nuc_nuc_energy'], 0.9953176337)
    # check types of the integrals
    assert isinstance(data['H'], tuple)
    assert len(data['H']) == 1
    assert isinstance(data['H'][0], np.ndarray)
    assert isinstance(data['G'], tuple)
    assert len(data['G']) == 1
    assert isinstance(data['G'][0], np.ndarray)
    for matrix in data['H'] + data['G']:
        assert np.all(np.array(matrix.shape) == data['H'][0].shape[0])


def test_hartreefock_h2_rhf_sto6g():
    """ Tests HF against LiH STO6G
    """
    # file location specified (absolute)
    hf_dict = hartreefock("{0}/../../../data/test/h2.xyz".format(os.path.dirname(__file__)),
                          "sto-6g")
    check_data_h2_rhf_sto6g(hf_dict)

    # data reference
    hf_dict = hartreefock("test/h2.xyz", "sto-6g")
    check_data_h2_rhf_sto6g(hf_dict)

    # bad file
    assert_raises(ValueError, lambda: hartreefock('magic.xyz', 'sto-6g'))

    # unrestricted
    assert_raises(NotImplementedError, lambda: hartreefock("test/h2.xyz", "sto-6g",
                                                           is_unrestricted=True))


def test_hartreefock():
    """ Tests HF against LiH STO6G
    """
    # file location specified
    hf_dict = hartreefock("{0}/../../../data/test/lih.xyz".format(os.path.dirname(__file__)),
                          "sto-6g")
    check_data_lih_rhf_sto6g(hf_dict)

    hf_dict = hartreefock("test/lih.xyz", "sto-6g")
    check_data_lih_rhf_sto6g(hf_dict)

    # bad file
    assert_raises(ValueError, lambda: hartreefock('magic.xyz', 'sto-6g'))

    # unrestricted
    assert_raises(NotImplementedError, lambda: hartreefock("test/lih.xyz", "sto-6g",
                                                           is_unrestricted=True))


def test_generate_fci_cimatrix_h2_631gs():
    """ Tests generate_fci_cimatrix with H2 6-31G*
    """
    #### H2 ####
    # HF energy: -1.13126983927
    # FCI energy: -1.1651487496
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2.xyz')
    hf_dict = hartreefock(data_path, '6-31gs')

    nelec = 2
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]

    # nelec is number
    ci_matrix, pspace = generate_fci_cimatrix(H[0], G[0], nelec, is_chemist_notation=False)
    ground_energy = scipy.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-1.1651486697)) < 1e-7
    # nelec is tuple
    ci_matrix, pspace = generate_fci_cimatrix(H[0], G[0], (1, 1), is_chemist_notation=False)
    ground_energy = scipy.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-1.1651486697)) < 1e-7
    # invalid nelec
    assert_raises(ValueError,
                  lambda: generate_fci_cimatrix(H[0], G[0], '2', is_chemist_notation=False))
    assert_raises(ValueError,
                  lambda: generate_fci_cimatrix(H[0], G[0], (1, 1, 2), is_chemist_notation=False))
    # chemist notation
    ci_matrix, pspace = generate_fci_cimatrix(H[0], np.einsum('ikjl->ijkl', G[0]), 2,
                                              is_chemist_notation=True)
    ground_energy = scipy.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-1.1651486697)) < 1e-7


def test_generate_fci_cimatrix_lih_sto6g():
    """ Tests generate_fci_cimatrix with LiH STO-6G
    """
    #### LiH ####
    # HF energy: -7.95197153880
    # FCI energy: -7.9723355823
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih.xyz')
    hf_dict = hartreefock(data_path, 'sto-6g')

    nelec = 4
    E_hf = hf_dict["el_energy"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]

    # nelec is int
    ci_matrix, pspace = generate_fci_cimatrix(H[0], G[0], nelec, is_chemist_notation=False)
    ground_energy = scipy.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-7.9723355823)) < 1e-7
    # nelec is tuple
    ci_matrix, pspace = generate_fci_cimatrix(H[0], G[0], (2, 2), is_chemist_notation=False)
    ground_energy = scipy.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-7.9723355823)) < 1e-7
    # invalid nelec
    assert_raises(ValueError,
                  lambda: generate_fci_cimatrix(H[0], G[0], '4', is_chemist_notation=False))
    assert_raises(ValueError,
                  lambda: generate_fci_cimatrix(H[0], G[0], (1, 1, 2), is_chemist_notation=False))
    # chemist notation
    ci_matrix, pspace = generate_fci_cimatrix(H[0], np.einsum('ikjl->ijkl', G[0]), 4,
                                              is_chemist_notation=True)
    ground_energy = scipy.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-7.9723355823)) < 1e-7
