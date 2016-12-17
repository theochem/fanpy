from __future__ import absolute_import, division, print_function
import os
import numpy as np
from wfns.ci.solver import solve
from wfns.ci.fci import FCI
from wfns.wrapper.horton import gaussian_fchk


def test_fci_h2():
    #### H2 ####
    # HF energy: -1.13126983927
    # FCI energy: -1.1651487496
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_631gdp.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]

    fci = FCI(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, spin=0)
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
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih_hf_sto6g.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 4
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]

    fci = FCI(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = fci.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + fci.nuc_nuc - (-7.95197153880)) < 1e-8
    # check that hamiltonian is symmetric
    assert np.allclose(ci_matrix, ci_matrix.T)
    # solve
    solve(fci)
    # compare with number from Gaussian
    assert abs(fci.compute_energy()-(-7.9723355823)) < 1e-7


def test_fci_lih_631g():
    #### LiH ####
    # HF energy: -7.97926894940
    # FCI energy: -7.9982761
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih_hf_631g.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 4
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]

    fci = FCI(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, spin=0)
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
