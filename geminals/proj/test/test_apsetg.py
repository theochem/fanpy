from __future__ import absolute_import, division, print_function
import numpy as np
np.random.seed(2012)

from geminals.proj.apsetg import APsetG
from geminals.hort import hartreefock

def test_apsetg_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # Old Code Value : -1.86968284431
    # FCI Value :      -1.87832550029
    nelec = 2
    hf_dict = hartreefock(fn="test/h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # Solve with Jacobian using energy as a parameter
    apsetg = APsetG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apsetg()
    print('HF energy', -1.84444667247)
    print('new energy', apsetg.compute_energy())
    print('FCI value', -1.87832550029)
    assert -1.84444667247 > apsetg.compute_energy() > -1.87832550029
    assert abs(apsetg.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-7


def test_apsetg_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # Old Code Value : -8.96353105152
    # FCI Value :      -8.96741814557
    nelec = 4
    hf_dict = hartreefock(fn="test/lih.xyz", basis="sto-6g", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # Compare apsetg energy with old code
    # Solve with Jacobian using energy as a parameter
    apsetg = APsetG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apsetg()
    print('HF energy', -8.9472891719)
    print('new energy', apsetg.compute_energy())
    print('FCI value', -8.96741814557)
    assert -8.9472891719 > apsetg.compute_energy() > -8.96741814557
    print(apsetg.compute_energy(include_nuc=False), 'new code')
    print(-8.96353105152, 'old code')
    assert False
