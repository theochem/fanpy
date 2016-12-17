from __future__ import absolute_import, division, print_function

import os
from wfns.ci.solver import solve
from wfns.ci.ci_pairs import CIPairs
from wfns.wrapper.horton import gaussian_fchk


def test_cipairs_wavefunction():
    #### H2 ####
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_631gdp.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    print(nuc_nuc)
    cipairs = CIPairs(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    # compare HF numbers
    print(cipairs.compute_ci_matrix()[0, 0])
    print(cipairs.compute_ci_matrix()[0, 0] + cipairs.nuc_nuc)
    assert abs(cipairs.compute_ci_matrix()[0, 0] + cipairs.nuc_nuc - (-1.131269841877) < 1e-8)
    solve(cipairs)
    print(cipairs.compute_energy())
