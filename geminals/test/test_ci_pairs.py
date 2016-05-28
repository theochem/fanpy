from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../')
from ci_pairs import CIPairs
from hort import hartreefock

def test_cipairs_wavefunction():
    #### H2 ####
    nelec = 2
    hf_dict = hartreefock(fn="./h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    cipairs = CIPairs(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    # compare HF numbers
    print(cipairs.compute_ci_matrix()[0,0])
    print(cipairs.compute_ci_matrix()[0,0]+cipairs.nuc_nuc)
    assert abs(cipairs.compute_ci_matrix()[0,0]+cipairs.nuc_nuc-(-1.131269841877) < 1e-8)
    cipairs()
    print(cipairs.compute_energy())
