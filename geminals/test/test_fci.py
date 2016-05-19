from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../')
from fci import FCI
from hort import hartreefock

def test_fci_wavefunction():
    #### H2 ####
    nelec = 2
    hf_dict = hartreefock(fn="./h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    fci = FCI(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    fci()
    # compare with number from Gaussian
    assert abs(fci.compute_energy()-(-1.1651486697)) < 1e-7
    #### LiH ####
    nelec = 4
    hf_dict = hartreefock(fn="./lih.xyz", basis="sto-6g", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    fci = FCI(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    fci()
    print(fci.compute_energy(), -7.9723355823)
    #assert abs(fci.compute_energy()-(-1.1651486697)) < 1e-7

test_fci_wavefunction()
