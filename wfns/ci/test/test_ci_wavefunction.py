from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import os
import numpy as np
from wfns.wrapper.pyscf import generate_fci_cimatrix
from wfns.wrapper.horton import gaussian_fchk
from wfns.ci.ci_wavefunction import CIWavefunction
from scipy.linalg import eigh


class TestCIWavefunction(CIWavefunction):
    # overwrite to stop initialization
    def __init__(self):
        pass

    nspatial = 4

    @property
    def _nci(self):
        return 4

    def compute_civec(self):
        return [0b1111, 0b10111, 0b11011, 0b11101]

    def compute_ci_matrix(self):
        pass


def test_assign_nci():
    """
    Tests CIWavefunction.assign_nci
    """
    test = TestCIWavefunction()
    # None assigned
    test.nci = None
    test.assign_nci()
    assert test.nci == 4
    test.nci = None
    test.assign_nci(None)
    assert test.nci == 4
    # Int assigned
    test.nci = None
    test.assign_nci(10)
    assert test.nci == 10
    # Other assigned
    assert_raises(TypeError, lambda: test.assign_nci('123'))


def test_assign_spin():
    """
    Tests CIWavefunction.assign_spin
    """
    test = TestCIWavefunction()
    # None assigned
    test.assign_spin()
    assert test.spin is None
    # Int assigned
    test.spin = None
    test.assign_spin(10)
    assert test.spin == 10
    # Other assigned
    assert_raises(TypeError, lambda: test.assign_spin('123'))


def test_assign_civec():
    """
    Tests CIWavefunction.assign_civec
    """
    test = TestCIWavefunction()
    test.assign_nci()
    # None assigned
    test.civec = None
    test.assign_civec()
    assert test.civec == (0b1111, 0b10111, 0b11011, 0b11101)
    test.civec = None
    test.assign_civec(None)
    assert test.civec == (0b1111, 0b10111, 0b11011, 0b11101)
    # tuple assigned
    test.civec = None
    test.assign_civec((0b1111,))
    assert test.civec == (0b1111,)
    # list assigned
    test.civec = None
    test.assign_civec([0b1111, ])
    assert test.civec == (0b1111,)
    # Other assigned
    assert_raises(TypeError, lambda: test.assign_civec(0b1111))
    assert_raises(TypeError, lambda: test.assign_civec((0b1111)))
    assert_raises(TypeError, lambda: test.assign_civec('123'))


def test_dict_sd_coeff():
    """
    Tests CIWavefunction.dict_sd_coeff
    """
    test = TestCIWavefunction()
    test.civec = [0b1111, 0b110011]
    test.sd_coeffs = np.arange(6).reshape(2, 3)
    # ground state
    sd_coeff = test.dict_sd_coeff()
    assert sd_coeff[0b1111] == 0
    assert sd_coeff[0b110011] == 3
    sd_coeff = test.dict_sd_coeff(exc_lvl=0)
    assert sd_coeff[0b1111] == 0
    assert sd_coeff[0b110011] == 3
    # 1st excited
    sd_coeff = test.dict_sd_coeff(exc_lvl=1)
    assert sd_coeff[0b1111] == 1
    assert sd_coeff[0b110011] == 4
    # 2nd excited
    sd_coeff = test.dict_sd_coeff(exc_lvl=2)
    assert sd_coeff[0b1111] == 2
    assert sd_coeff[0b110011] == 5
    # bad excitation
    assert_raises(TypeError, lambda: test.dict_sd_coeff(exc_lvl='2'))
    assert_raises(TypeError, lambda: test.dict_sd_coeff(exc_lvl=2.0))
    assert_raises(ValueError, lambda: test.dict_sd_coeff(exc_lvl=-2))

def test_get_density_matrix_h2_sto6g():
    """
    Tests CIWavefunction.get_density_matrix using H2 system (STO-6G)

    Note
    ----
    One elctron density matrix is obtained using Gaussian
    Everything else is generated using PySCF

    """
    # H2 STO-6G (from PySCF)
    # Electronic Energy : [-1.85908985 -1.25453842 -0.89131832 -0.24166486]
    # Total Energy : [-1.14591301 -0.54136158 -0.17814149  0.47151197]
    # SD Coefficients : [[  9.93594152e-01   1.28646621e-16  -3.29385185e-16  -1.13007352e-01]
    #                    [ -1.11022302e-16  -7.07106781e-01  -7.07106781e-01   2.22044605e-16]
    #                    [ -2.83828762e-16   7.07106781e-01  -7.07106781e-01   3.70478534e-16]
    #                    [ -1.13007352e-01   0.00000000e+00  -4.44089210e-16  -9.93594152e-01]]

    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_sto6g.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    H = hf_dict["H"][0]
    G = hf_dict["G"][0]

    ci_matrix, civec = generate_fci_cimatrix(H, G, nelec, is_chemist_notation=False)
    sd_coeffs = eigh(ci_matrix)[1][:, 0]
    energy = eigh(ci_matrix)[0][0]

    test = TestCIWavefunction()
    test.nspatial = 2
    test.civec = civec
    test.sd_coeffs = sd_coeffs

    density1, density2 = test.get_density_matrix(val_threshold=0)

    # Count number of electrons
    assert abs(np.einsum('ii', density1) - nelec) < 1e-8
    assert abs(np.einsum('iiii', density2) - nelec) < 1e-8

    # Reference from Gaussian
    ref_density1 = [[ 0.197446E+01, -0.163909E-14],
                    [-0.163909E-14,  0.255413E-01]]
    assert np.allclose(density1, ref_density1)

    # Reconstruct FCI energy
    # physicist notation
    density1, density2 = test.get_density_matrix(val_threshold=0, notation='physicist')
    print((np.einsum('ij,ij', H, density1) +
               0.5*np.einsum('ijkl,ijkl', G, density2)), energy)
    assert abs((np.einsum('ij,ij', H, density1) +
               0.5*np.einsum('ijkl,ijkl', G, density2)) - (energy)) < 1e-8
    # chemist notation
    density1, density2 = test.get_density_matrix(val_threshold=0, notation='chemist')
    assert abs((np.einsum('ij,ij', H, density1) +
                0.5*np.einsum('ijkl,iklj', G, density2)) - (energy)) < 1e-8


def test_get_density_matrix_h2_631gdp():
    """
    Tests CIWavefunction.get_density_matrix using H2 system (6-31G**)

    Try to generate numbers from Gaussian and PySCF

    Note
    ----
    One elctron density matrix is obtained using Gaussian
    Everything else is generated using PySCF
    The few coefficients we can get from Gaussian don't match up with PySCF (so the one electron
    density matrix is not tested against Gaussian)
    """
    # H2 6-31G**
    # Electronic Energy : -1.87832559
    # Total Energy : -1.16514876

    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_631gdp.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    H = hf_dict["H"][0]
    G = hf_dict["G"][0]
    ns = H.shape[0]

    ci_matrix, civec = generate_fci_cimatrix(H, G, nelec, is_chemist_notation=False)
    sd_coeffs = eigh(ci_matrix)[1][:, 0]
    energy = eigh(ci_matrix)[0][0]

    test = TestCIWavefunction()
    test.nspatial = H.shape[0]
    test.civec = civec
    test.sd_coeffs = sd_coeffs
    density1, density2 = test.get_density_matrix(val_threshold=0)

    # Count number of electrons
    assert abs(np.einsum('ii', density1) - nelec) < 1e-8
    # print(abs(np.einsum('iiii', density2) - nelec))
    # assert abs(np.einsum('iiii', density2) - nelec) < 1e-8

    # Reference from Gaussian
    # commented out b/c the SD coefficients from Gaussian and PySCF are different
    # ref_density1 = np.zeros(H.shape)
    # ref_density1[np.tril_indices(H.shape[0])] = np.array([0.196959E+01, 0.151608E-14, 0.202432E-01, 0.278696E-03,
    #                                                       0.136404E-16, 0.634765E-02, -0.957626E-15, -0.929509E-05,
    #                                                       -0.826580E-17, 0.213190E-03, -0.464457E-17, 0.263984E-18,
    #                                                       0.294631E-18, 0.440488E-19, 0.157700E-02, -0.142304E-16,
    #                                                       0.359306E-18, 0.592319E-18, -0.291043E-19, -0.232014E-18,
    #                                                       0.157700E-02, -0.112405E-02, 0.129946E-16, -0.674652E-04,
    #                                                       -0.133419E-17, -0.131237E-18, -0.206250E-18, 0.177548E-03,
    #                                                       0.338487E-18, -0.219942E-17, -0.253181E-18, 0.165883E-18,
    #                                                       -0.227609E-20, 0.158935E-17, -0.214759E-19, 0.118809E-03,
    #                                                       0.156062E-17, -0.597619E-18, 0.219858E-18, 0.205982E-19,
    #                                                       0.103419E-17, 0.547492E-19, -0.119113E-18, 0.144748E-17,
    #                                                       0.118809E-03, 0.252367E-15, 0.768633E-05, -0.579179E-17,
    #                                                       0.134128E-03, 0.204890E-20, 0.799816E-20, 0.346587E-18,
    #                                                       0.319358E-18, -0.639014E-18, 0.405324E-04])
    # ref_density1 += np.tril(ref_density1, k=-1).T
    # assert np.allclose(density1, ref_density1)

    # Reconstruct FCI energy
    # physicist notation
    density1, density2 = test.get_density_matrix(val_threshold=0, notation='physicist')
    # density1, density2 = fci.get_density_matrix(val_threshold=0, notation='physicist')
    assert abs((np.einsum('ij,ij', H, density1) +
               0.5*np.einsum('ijkl,ijkl', G, density2)) - (energy)) < 1e-8
    # chemist notation
    density1, density2 = test.get_density_matrix(val_threshold=0, notation='chemist')
    # density1, density2 = fci.get_density_matrix(val_threshold=0, notation='chemist')
    assert abs((np.einsum('ij,ij', H, density1) +
                0.5*np.einsum('ijkl,iklj', G, density2)) - (energy)) < 1e-8

def test_get_density_matrix_lih_sto6g():
    """
    Tests CIWavefunction.get_density_matrix using LiH system (STO6G)

    Try to generate numbers from Gaussian and PySCF

    Note
    ----
    One elctron density matrix is obtained using Gaussian
    Everything else is generated using PySCF
    The few coefficients we can get from Gaussian don't match up with PySCF (so the one electron
    density matrix is not tested against Gaussian)
    """
    # LiH sto-6g
    # Electronic Energy : [-8.99359377 -8.88992694 -8.87276083 ...
    # Total Energy : [-7.99827613 -7.8946093  -7.87744319 ...

    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih_hf_sto6g.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 4
    H = hf_dict["H"][0]
    G = hf_dict["G"][0]

    ci_matrix, civec = generate_fci_cimatrix(H, G, nelec, is_chemist_notation=False)
    sd_coeffs = eigh(ci_matrix)[1][:, 0]
    energy = eigh(ci_matrix)[0][0]

    test = TestCIWavefunction()
    test.nspatial = H.shape[0]
    test.civec = civec
    test.sd_coeffs = sd_coeffs
    density1, density2 = test.get_density_matrix(val_threshold=0)

    # Count number of electrons
    assert abs(np.einsum('ii', density1) - nelec) < 1e-8
    # assert abs(np.einsum('iiii', density2) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    density1, density2 = test.get_density_matrix(val_threshold=0, notation='physicist')
    print((np.einsum('ij,ij', H, density1) +
                0.5*np.einsum('ijkl,ijkl', G, density2)))
    assert abs((np.einsum('ij,ij', H, density1) +
                0.5*np.einsum('ijkl,ijkl', G, density2)) - (energy)) < 1e-8
    # chemist notation
    density1, density2 = test.get_density_matrix(val_threshold=0, notation='chemist')
    assert abs((np.einsum('ij,ij', H, density1) +
                0.5*np.einsum('ijkl,iklj', G, density2)) - (energy)) < 1e-8


def test_get_density_matrix_lih_631g():
    """
    Tests CIWavefunction.get_density_matrix using LiH system (6-31G**)

    Try to generate numbers from Gaussian and PySCF

    Note
    ----
    One elctron density matrix is obtained using Gaussian
    Everything else is generated using PySCF
    The few coefficients we can get from Gaussian don't match up with PySCF (so the one electron
    density matrix is not tested against Gaussian)
    """
    # LiH 6-31G
    # Electronic Energy : [-8.99359377 -8.88992694 -8.87276083 ...
    # Total Energy : [-7.99827613 -7.8946093  -7.87744319 ...

    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih_hf_631g.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 4
    H = hf_dict["H"][0]
    G = hf_dict["G"][0]

    ci_matrix, civec = generate_fci_cimatrix(H, G, nelec, is_chemist_notation=False)
    sd_coeffs = eigh(ci_matrix)[1][:, 0]
    energy = eigh(ci_matrix)[0][0]

    test = TestCIWavefunction()
    test.nspatial = H.shape[0]
    test.civec = civec
    test.sd_coeffs = sd_coeffs
    density1, density2 = test.get_density_matrix(val_threshold=0)

    # Count number of electrons
    assert abs(np.einsum('ii', density1) - nelec) < 1e-8
    # assert abs(np.einsum('iiii', density2) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    density1, density2 = test.get_density_matrix(val_threshold=0, notation='physicist')
    # density1, density2 = fci.get_density_matrix(val_threshold=0, notation='physicist')
    assert abs((np.einsum('ij,ij', H, density1) +
                0.5*np.einsum('ijkl,ijkl', G, density2)) - (energy)) < 1e-8
    # chemist notation
    density1, density2 = test.get_density_matrix(val_threshold=0, notation='chemist')
    # density1, density2 = fci.get_density_matrix(val_threshold=0, notation='chemist')
    assert abs((np.einsum('ij,ij', H, density1) +
                0.5*np.einsum('ijkl,iklj', G, density2)) - (energy)) < 1e-8
