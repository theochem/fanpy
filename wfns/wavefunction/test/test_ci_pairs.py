""" Tests wfns.wavefunction.ci_pairs
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.tools import find_datafile
from wfns.wavefunction.ci_pairs import CIPairs


class TestCIPairs(CIPairs):
    """CIPairs class without initializer."""
    def __init__(self):
        pass


def test_assign_sd_vec():
    """Test CIPairs.assign_sd_vec."""
    test = TestCIPairs()
    test.assign_nelec(6)
    test.assign_nspin(10)
    test.assign_spin(0)
    test.assign_seniority(0)
    test.assign_sd_vec()
    assert test.sd_vec == (0b0011100111, 0b0101101011, 0b1001110011, 0b0110101101, 0b1010110101,
                           0b0111001110, 0b1011010110)
    assert_raises(ValueError, test.assign_sd_vec, (0b0011100111, ))


# FIXME: implement after ap1rog
def test_to_ap1rog():
    """ Tests wfns.wavefunction.ci_pairs.CIPairs.to_ap1rog
    """
    test = CIPairs(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), excs=[0, 1, 2])
    test.sd_coeffs = np.arange(9, 0, -1).reshape(3, 3)
    assert_raises(TypeError, test.to_ap1rog, 1.0)
    assert_raises(ValueError, test.to_ap1rog, -1)
    ap1rog = test.to_ap1rog(exc_lvl=0)
    assert np.allclose(ap1rog.params, np.array([6/9, 3/9, 4]))
    ap1rog = test.to_ap1rog(exc_lvl=1)
    assert np.allclose(ap1rog.params, np.array([5/8, 2/8, 3.875]))
    ap1rog = test.to_ap1rog(exc_lvl=2)
    assert np.allclose(ap1rog.params, np.array([4/7, 1/7, 7*4/7 - 2*1/7]))


# FIXME: implement after solver is implemented
def test_to_ap1rog_h2_sto6g_ground():
    """ Tests wfns.wavefunction.ci_pairs.CIPairs.to_ap1rog using H2 with HF/STO6G orbitals
    """
    nelec = 2

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.71317683129

    cipair = CIPairs(nelec, one_int, two_int, nuc_nuc=nuc_nuc, excs=[0, 1])
    solve(cipair)
    # ground state
    test_ap1rog = cipair.to_ap1rog(exc_lvl=0)
    test_ap1rog.normalize()
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b0101, ))
    proj_solve(ap1rog)
    assert np.allclose(test_ap1rog.params, ap1rog.params)
    # excited state
    test_ap1rog = cipair.to_ap1rog(exc_lvl=1)
    test_ap1rog.normalize()
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b1010, ))
    proj_solve(ap1rog)
    assert np.allclose(test_ap1rog.params, ap1rog.params)


def test_to_ap1rog_lih_sto6g():
    """ Tests wfns.wavefunction.ci_pairs.CIPairs.to_ap1rog with LiH with HF/STO6G orbitals
    """
    nelec = 4

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356

    cipair = CIPairs(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(cipair)
    test_ap1rog = cipair.to_ap1rog(exc_lvl=0)
    test_ap1rog.normalize()
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    proj_solve(ap1rog)
    assert np.all(test_ap1rog.params / ap1rog.params > 0.99)


def test_to_ap1rog_h4_sto6g():
    """ Tests wfns.wavefunction.ci_pairs.CIPairs.to_ap1rog with H4 with HF/STO6G orbitals
    """
    nelec = 4

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h4_square_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
    nuc_nuc = 2.70710678119

    cipair = CIPairs(nelec, one_int, two_int, nuc_nuc=nuc_nuc, excs=[0, 1])
    solve(cipair)
    test_ap1rog = cipair.to_ap1rog(exc_lvl=1)
    #FIXME: SD 0b10101010 cannot be obtained by single pair excitation of 0b01010101 (reference)
    #       this means that one coefficient is discarded when converted to ap1rog
    #       Maybe it will be better to keep 0b00110011 as the reference and modify the normalization
    sd_coeffs = cipair.sd_coeffs[:, 1].flatten()/ cipair.sd_coeffs[1, 1]
    assert np.allclose(sd_coeffs[0], test_ap1rog.params[2])
    assert np.allclose(sd_coeffs[1], 1)
    assert np.allclose(sd_coeffs[2], test_ap1rog.params[3])
