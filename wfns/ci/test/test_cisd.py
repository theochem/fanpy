""" Tests wfns.ci.cisd
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from wfns.ci.solver import solve
from wfns.ci.cisd import CISD
from wfns.sd_list import sd_list
from wfns.tools import find_datafile


def test_generate_civec():
    """ Tests CISD.generate_civec
    """
    test = CISD(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    assert test.generate_civec() == sd_list(test.nelec, test.nspatial, num_limit=None, spin=None,
                                            seniority=None, exc_orders=[1, 2])

def test_cisd_h2_631gdp():
    """ Tests CISD wavefunction using H2 (6-31G**)

    Compared to Gausssian results

    Note
    ----
    You have to be careful with Gaussian and molpro calculations because they freeze core by
    default
    """
    #### H2 ####
    # HF energy: -1.13126983927
    # FCI energy: -1.1651487496
    nelec = 2

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129

    cisd = CISD(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = cisd.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + cisd.nuc_nuc - (-1.131269841877) < 1e-8)
    # solve
    solve(cisd)
    # compare with number from Gaussian
    assert abs(cisd.get_energy() - (-1.1651486697)) < 1e-7


def test_cisd_lih_631g():
    """ Tests CISD wavefunction using LiH (6-31G)

    Compared to Molpro results

    Note
    ----
    You have to be careful with Gaussian and molpro calculations because they freeze core by
    default
    """
    #### LiH ####
    # HF energy: -7.97926895
    # CISD energy: -7.99826182
    nelec = 4

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_631g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_631g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_631g_twoint.npy')), )
    nuc_nuc = 0.995317634356

    cisd = CISD(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc, spin=None)
    ci_matrix = cisd.compute_ci_matrix()

    # compare HF numbers
    assert abs(ci_matrix[0, 0] + cisd.nuc_nuc - (-7.97926895) < 1e-8)
    # solve
    solve(cisd)
    # compare with number from Gaussian
    assert abs(cisd.get_energy() - (-7.99826182)) < 1e-7
