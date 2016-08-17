from __future__ import absolute_import, division, print_function
import numpy as np
np.random.seed(2012)

from geminals.proj.apfsg import APfsG
from geminals.hort import hartreefock

def test_apfsg_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # FCI Value :      -1.87832550029
    nelec = 2
    hf_dict = hartreefock(fn="test/h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # Solve with Jacobian using energy as a parameter
    apfsg = APfsG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apfsg(jac=None)
    print('HF energy', -1.84444667247)
    print('new energy', apfsg.compute_energy())
    print('FCI value', -1.87832550029)
    assert -1.84444667247 > apfg.computer_energy() > -1.87832550029
    assert abs(apfsg.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-7


def test_apfsg_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # FCI Value :      -8.96741814557
    nelec = 4
    hf_dict = hartreefock(fn="test/lih.xyz", basis="sto-6g", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # Compare apfsg energy with old code
    # Solve with Jacobian using energy as a parameter
    apfsg = APfsG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apfsg(jac=None)
    print('HF energy', -8.9472891719)
    print('new energy', apfsg.compute_energy())
    print('FCI value', -8.96741814557)
    assert -8.9472891719 > apfg.computer_energy() > -8.96741814557
    assert False
