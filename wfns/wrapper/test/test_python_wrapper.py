"""Test wfns.wrapper.horton."""
import numpy as np
from wfns.tools import find_datafile
from wfns.wrapper.python_wrapper import generate_hartreefock_results, generate_fci_results


def check_data_h2_rhf_sto6g(el_energy, nuc_nuc_energy, one_int, two_int):
    """Check data for h2 rhf sto6g calculation."""
    assert np.allclose(el_energy, -1.838434259892)
    assert np.allclose(nuc_nuc_energy, 0.713176830593)
    assert np.allclose(one_int, np.array([[-1.25637540e+00, 0.0000000000000],
                                          [0.0000000000000, -4.80588203e-01]]))
    assert np.allclose(two_int, np.array([[[[6.74316543e-01, 0.000000000000],
                                            [0.000000000000, 1.81610048e-01]],
                                           [[0.000000000000, 6.64035234e-01],
                                            [1.81610048e-01, 0.000000000000]]],
                                          [[[0.000000000000, 1.81610048e-01],
                                            [6.64035234e-01, 0.000000000000]],
                                           [[1.81610048e-01, 0.000000000000],
                                            [0.000000000000, 6.98855952e-01]]]]))


def check_data_h2_uhf_sto6g(el_energy, nuc_nuc_energy, one_int, two_ints):
    """Check data for h2 uhf sto6g calculation."""
    assert np.allclose(el_energy, -1.838434259892)
    assert np.allclose(nuc_nuc_energy, 0.713176830593)
    assert np.allclose(one_int[0], np.array([[-1.25637540e+00, 0.0000000000000],
                                            [0.0000000000000, -4.80588203e-01]]))
    assert np.allclose(one_int[1], np.array([[-1.25637540e+00, 0.0000000000000],
                                            [0.0000000000000, -4.80588203e-01]]))
    assert len(two_ints) == 3
    for two_int in two_ints:
        assert np.allclose(two_int, np.array([[[[6.74316543e-01, 0.000000000000],
                                                [0.000000000000, 1.81610048e-01]],
                                               [[0.000000000000, 6.64035234e-01],
                                                [1.81610048e-01, 0.000000000000]]],
                                              [[[0.000000000000, 1.81610048e-01],
                                                [6.64035234e-01, 0.000000000000]],
                                               [[1.81610048e-01, 0.000000000000],
                                                [0.000000000000, 6.98855952e-01]]]]))


# NOTE: the integrals from PySCF have different sign from HORTON's for some reason
# def check_data_h2_rhf_631gdp(el_energy, nuc_nuc_energy, one_int, two_int):
#     """Check data for LiH rhf sto6g calculation."""
#     assert np.allclose(el_energy, -1.84444667027)
#     assert np.allclose(nuc_nuc_energy, 0.7131768310)

#     # check types of the integrals
#     assert np.allclose(one_int, np.load(find_datafile('test/h2_hf_631gdp_oneint.npy')))
#     assert np.allclose(two_int, np.load(find_datafile('test/h2_hf_631gdp_twoint.npy')))


def check_data_lih_rhf_sto6g(*data):
    """Check data for LiH rhf sto6g calculation."""
    el_energy, nuc_nuc_energy, one_int, two_int = data

    assert np.allclose(el_energy, -7.95197153880 - 0.9953176337)
    assert np.allclose(nuc_nuc_energy, 0.9953176337)

    # check types of the integrals
    assert isinstance(one_int, np.ndarray)
    assert isinstance(two_int, np.ndarray)
    assert np.all(np.array(one_int.shape) == one_int[0].shape[0])
    assert np.all(np.array(two_int.shape) == one_int[0].shape[0])


def test_horton_hartreefock_h2_rhf_sto6g():
    """Test HORTON's hartreefock against H2 HF STO-6G data from Gaussian."""
    hf_data = generate_hartreefock_results('/usr/bin/python2', 'horton_hartreefock.py',
                                           energies_name='energies.npy',
                                           oneint_name='oneint.npy',
                                           twoint_name='twoint.npy',
                                           remove_npyfiles=True,
                                           fn=find_datafile('test/h2.xyz'), basis="sto-6g", nelec=2)
    check_data_h2_rhf_sto6g(*hf_data)


def test_horton_gaussian_fchk_h2_rhf_sto6g():
    """Test HORTON's gaussian_fchk against H2 HF STO-6G data from Gaussian."""
    fchk_data = generate_hartreefock_results('/usr/bin/python2', 'horton_gaussian_fchk.py',
                                             energies_name='energies.npy',
                                             oneint_name='oneint.npy',
                                             twoint_name='twoint.npy',
                                             remove_npyfiles=True,
                                             fchk_file=find_datafile('test/h2_hf_sto6g.fchk'),
                                             horton_internal=False)
    check_data_h2_rhf_sto6g(*fchk_data)


def test_gaussian_fchk_h2_uhf_sto6g():
    """Test HORTON's gaussian_fchk against H2 UHF STO-6G data from Gaussian"""
    fchk_data = generate_hartreefock_results('/usr/bin/python2', 'horton_gaussian_fchk.py',
                                             energies_name='energies.npy',
                                             oneint_name='oneint.npy',
                                             twoint_name='twoint.npy',
                                             remove_npyfiles=True,
                                             fchk_file=find_datafile('test/h2_uhf_sto6g.fchk'),
                                             horton_internal=False)
    check_data_h2_uhf_sto6g(*fchk_data)


def test_pyscf_hartreefock_h2_rhf_sto6g():
    """Test PySCF HF against H2 RHF STO-6G data from Gaussian"""
    hf_data = generate_hartreefock_results('/usr/bin/python2', 'pyscf_hartreefock.py',
                                           energies_name='energies.npy',
                                           oneint_name='oneint.npy',
                                           twoint_name='twoint.npy',
                                           remove_npyfiles=True,
                                           xyz_file=find_datafile('test/h2.xyz'), basis="sto-6g")
    check_data_h2_rhf_sto6g(*hf_data)


def test_pyscf_hartreefock_lih_rhf_sto6g():
    """Test PySCF HF against LiH RHF STO6G data from Gaussian."""
    # file location specified
    hf_data = generate_hartreefock_results('/usr/bin/python2', 'pyscf_hartreefock.py',
                                           energies_name='energies.npy',
                                           oneint_name='oneint.npy',
                                           twoint_name='twoint.npy',
                                           remove_npyfiles=True,
                                           xyz_file=find_datafile('test/lih.xyz'), basis="sto-6g")
    check_data_lih_rhf_sto6g(*hf_data)


def test_generate_fci_cimatrix_h2_631gd():
    """Test PySCF's FCI calculation against H2 FCI 6-31G* data from Gaussian.
    HF energy: -1.13126983927
    FCI energy: -1.1651487496
    """
    hf_data = generate_hartreefock_results('/usr/bin/python2', 'pyscf_hartreefock.py',
                                           energies_name='energies.npy',
                                           oneint_name='oneint.npy',
                                           twoint_name='twoint.npy',
                                           remove_npyfiles=True,
                                           xyz_file=find_datafile('test/h2.xyz'), basis="6-31gs")

    nelec = 2
    el_energy, nuc_nuc, one_int, two_int = hf_data

    # physicist notation
    ci_matrix, pspace = generate_fci_results('/usr/bin/python2',
                                             cimatrix_name='cimatrix.npy',
                                             sds_name='sds.npy',
                                             remove_npyfiles=True,
                                             h1e=one_int,
                                             eri=two_int,
                                             nelec=nelec,
                                             is_chemist_notation=False)
    ground_energy = np.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-1.1651486697)) < 1e-7

    # chemist notation
    ci_matrix, pspace = generate_fci_results('/usr/bin/python2',
                                             cimatrix_name='cimatrix.npy',
                                             sds_name='sds.npy',
                                             remove_npyfiles=True,
                                             h1e=one_int,
                                             eri=np.einsum('ikjl->ijkl', two_int),
                                             nelec=nelec,
                                             is_chemist_notation=True)
    ground_energy = np.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-1.1651486697)) < 1e-7


def test_generate_fci_cimatrix_lih_sto6g():
    """Test generate_fci_cimatrix with LiH STO-6G

    HF energy: -7.95197153880
    FCI energy: -7.9723355823
    """
    hf_data = generate_hartreefock_results('/usr/bin/python2', 'pyscf_hartreefock.py',
                                           energies_name='energies.npy',
                                           oneint_name='oneint.npy',
                                           twoint_name='twoint.npy',
                                           remove_npyfiles=True,
                                           xyz_file=find_datafile('test/lih.xyz'), basis="sto-6g")

    nelec = 4
    el_energy, nuc_nuc, one_int, two_int = hf_data

    # physicist notation
    ci_matrix, pspace = generate_fci_results('/usr/bin/python2',
                                             cimatrix_name='cimatrix.npy',
                                             sds_name='sds.npy',
                                             remove_npyfiles=True,
                                             h1e=one_int,
                                             eri=two_int,
                                             nelec=nelec,
                                             is_chemist_notation=False)
    ground_energy = np.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-7.9723355823)) < 1e-7
    # chemist notation
    ci_matrix, pspace = generate_fci_results('/usr/bin/python2',
                                             cimatrix_name='cimatrix.npy',
                                             sds_name='sds.npy',
                                             remove_npyfiles=True,
                                             h1e=one_int,
                                             eri=np.einsum('ikjl->ijkl', two_int),
                                             nelec=nelec,
                                             is_chemist_notation=True)
    ground_energy = np.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-7.9723355823)) < 1e-7
