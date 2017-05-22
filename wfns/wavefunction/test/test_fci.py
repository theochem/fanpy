""" Tests wfns.wavefunction.fci
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from nose.plugins.attrib import attr
from wfns.solver.solver_ci import solve
from wfns.wavefunction.fci import FCI
from wfns.tools import find_datafile


class TestFCI(FCI):
    """FCI instance that skips initialization."""
    def __init__(self):
        pass


def test_fci_assign_seniority():
    """Test FCI.assign_seniority."""
    test = TestFCI()
    assert_raises(ValueError, test.assign_seniority, 0)
    assert_raises(ValueError, test.assign_seniority, 1)
    test.assign_seniority(None)
    assert test.seniority is None


def test_fci_assign_sd_vec():
    """Test FCI.assign_sd_vec."""
    test = FCI(2, 4)
    assert_raises(ValueError, test.assign_sd_vec, 1)
    assert_raises(ValueError, test.assign_sd_vec, [0b0101])
    test.assign_sd_vec(None)
    assert test.sd_vec == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)


# FIXME: implement after solver is implemented
def test_fci_h2_631gdp():
    """Test FCI wavefunction for H2 (6-31g**).

    HF energy: -1.13126983927
    FCI energy: -1.1651487496
    """
    nelec = 2

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129

    fci = FCI(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = fci.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + fci.nuc_nuc - (-1.131269841877) < 1e-8)
    # solve
    solve(fci)
    # compare with number from Gaussian
    assert abs(fci.get_energy() - (-1.1651486697)) < 1e-7


def test_fci_lih_sto6g():
    """Test FCI wavefunction for LiH STO-6G.

    HF energy: -7.95197153880
    FCI energy: -7.9723355823
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

    fci = FCI(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = fci.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + fci.nuc_nuc - (-7.95197153880)) < 1e-8
    # check that hamiltonian is symmetric
    assert np.allclose(ci_matrix, ci_matrix.T)
    # solve
    solve(fci)
    # compare with number from Gaussian
    assert abs(fci.get_energy()-(-7.9723355823)) < 1e-7


@attr('slow')
def test_fci_lih_631g():
    """Test FCI wavefunction for LiH 6-31G.

    HF energy: -7.97926894940
    FCI energy: -7.9982761
    """
    nelec = 4

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_631g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_631g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_631g_twoint.npy')), )
    nuc_nuc = 0.995317634356

    fci = FCI(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = fci.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + fci.nuc_nuc - (-7.97926894940)) < 1e-8
    # check that hamiltonian is symmetric
    assert np.allclose(ci_matrix, ci_matrix.T)
    # solve
    solve(fci)
    # compare with number from Gaussian
    assert abs(fci.get_energy()-(-7.9982761)) < 1e-7
