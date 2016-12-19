from __future__ import absolute_import, division, print_function
import numpy as np
from wfns.ci.solver import solve
from wfns.ci.cisd import CISD
from wfns.wrapper.horton import gaussian_fchk


def test_cisd_h2():
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
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')

    nelec = 2
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    cisd = CISD(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = cisd.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + cisd.nuc_nuc - (-1.131269841877) < 1e-8)
    # check that hamiltonian is symmetric
    assert np.allclose(ci_matrix, ci_matrix.T)
    # solve
    solve(cisd)
    # compare with number from Gaussian
    assert abs(cisd.compute_energy() - (-1.1651486697)) < 1e-7

def test_cisd_lih():
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
    hf_dict = gaussian_fchk('test/lih_hf_631g.fchk')

    nelec = 4
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    cisd = CISD(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, spin=None)
    ci_matrix = cisd.compute_ci_matrix()

    # compare HF numbers
    assert abs(ci_matrix[0, 0] + cisd.nuc_nuc - (-7.97926895) < 1e-8)
    # check that hamiltonian is symmetric
    assert np.allclose(ci_matrix, ci_matrix.T)
    # solve
    solve(cisd)

    # compare with number from Gaussian
    print(cisd.compute_energy(), (-7.99826182))
    assert abs(cisd.compute_energy() - (-7.99826182)) < 1e-7
