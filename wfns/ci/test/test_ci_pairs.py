from __future__ import absolute_import, division, print_function
from wfns.ci.solver import solve
from wfns.ci.ci_pairs import CIPairs
from wfns.wrapper.horton import gaussian_fchk


def test_cipairs_wavefunction():
    #### H2 ####
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')

    nelec = 2
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    print(nuc_nuc)
    cipairs = CIPairs(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    # compare HF numbers
    print(cipairs.compute_ci_matrix()[0, 0])
    print(cipairs.compute_ci_matrix()[0, 0] + cipairs.nuc_nuc)
    assert abs(cipairs.compute_ci_matrix()[0, 0] + cipairs.nuc_nuc - (-1.131269841877) < 1e-8)
    solve(cipairs)
    print(cipairs.compute_energy())
