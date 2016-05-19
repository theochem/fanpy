from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../')
from cisd import CISD
from hort import hartreefock

from nose.tools import assert_raises

def test_cisd_wavefunction():
    #### H2 ####
    nelec = 2
    hf_dict = hartreefock(fn="./h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    cisd = CISD(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    cisd()
    # compare with number from Gaussian
    assert abs(cisd.compute_energy()-(-1.1651486697)) < 1e-7
    #### LiH ####
    nelec = 4
    hf_dict = hartreefock(fn="./lih.xyz", basis="sto-6g", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    cisd = CISD(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    cisd()
    print(cisd.compute_energy(), -7.9721005124)
    assert abs(cisd.compute_energy()-(-7.9721005124)) < 1e-7

test_cisd_wavefunction()
