from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../')
from doci import DOCI
from hort import hartreefock

from nose.tools import assert_raises

def test_doci_wavefunction():
    #### H2 ####
    nelec = 2
    hf_dict = hartreefock(fn="./h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    doci = DOCI(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    doci()
    print(doci.compute_energy())
test_doci_wavefunction()
