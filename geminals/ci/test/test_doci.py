from __future__ import absolute_import, division, print_function
import os
from geminals.ci.doci import DOCI
from geminals.hort import gaussian_fchk


def test_doci_wavefunction():
    #### H2 ####
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_631gdp.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    doci = DOCI(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    # compare HF numbers
    print(doci.compute_ci_matrix()[0, 0])
    print(doci.compute_ci_matrix()[0, 0] + doci.nuc_nuc)
    assert abs(doci.compute_ci_matrix()[0, 0] + doci.nuc_nuc - (-1.131269841877) < 1e-8)
    doci()
    print(doci.compute_energy())
