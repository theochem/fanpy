from __future__ import absolute_import, division, print_function
from geminals.ci.doci import DOCI
from geminals.hort import hartreefock


def test_doci_wavefunction():
    #### H2 ####
    nelec = 2
    hf_dict = hartreefock(fn="test/h2.xyz", basis="6-31g**", nelec=nelec)
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
