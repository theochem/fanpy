from __future__ import absolute_import, division, print_function
import numpy as np
from nose.plugins.attrib import attr
from wfns.ci.solver import solve
from wfns.ci.fci import FCI
from wfns.wrapper.horton import gaussian_fchk


def test_fci_h2():
    #### H2 ####
    # HF energy: -1.13126983927
    # FCI energy: -1.1651487496
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')

    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    fci = FCI(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = fci.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + fci.nuc_nuc - (-1.131269841877) < 1e-8)
    # check that hamiltonian is symmetric
    assert np.allclose(ci_matrix, ci_matrix.T)
    # solve
    solve(fci)
    # compare with number from Gaussian
    assert abs(fci.compute_energy() - (-1.1651486697)) < 1e-7

def test_fci_lih_sto6g():
    #### LiH ####
    # HF energy: -7.95197153880
    # FCI energy: -7.9723355823
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')

    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    fci = FCI(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = fci.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + fci.nuc_nuc - (-7.95197153880)) < 1e-8
    # check that hamiltonian is symmetric
    assert np.allclose(ci_matrix, ci_matrix.T)
    # solve
    solve(fci)
    # compare with number from Gaussian
    assert abs(fci.compute_energy()-(-7.9723355823)) < 1e-7


@attr('slow')
def test_fci_lih_631g():
    #### LiH ####
    # HF energy: -7.97926894940
    # FCI energy: -7.9982761
    hf_dict = gaussian_fchk('test/lih_hf_631g.fchk')

    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    fci = FCI(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = fci.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + fci.nuc_nuc - (-7.97926894940)) < 1e-8
    # check that hamiltonian is symmetric
    assert np.allclose(ci_matrix, ci_matrix.T)
    # solve
    solve(fci)
    # compare with number from Gaussian
    print(fci.compute_energy(), -7.97926894940)
    assert abs(fci.compute_energy()-(-7.9982761)) < 1e-7
