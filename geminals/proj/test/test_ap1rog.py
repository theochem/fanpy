from __future__ import absolute_import, division, print_function
import numpy as np

from geminals.proj.ap1rog import AP1roG
from geminals.hort import hartreefock, ap1rog

def test_ap1rog_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # Old code Value:  -1.86968286065
    # FCI Value :      -1.87832550029
    nelec = 2
    hf_dict = hartreefock(fn="test/h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # see if we can reproduce HF numbers
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    ap1rog.params *= 0.0
    assert abs(ap1rog.compute_energy(include_nuc=False, ref_sds=ap1rog.default_ref_sds)-(-1.84444667247)) < 1e-7
    # Compare AP1roG energy with old code
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    energy = ap1rog.compute_energy(include_nuc=False)
    print("new energy", energy)
    print(ap1rog.params, ap1rog.npair, ap1rog.nspatial-ap1rog.npair)
    assert abs(ap1rog.compute_energy(include_nuc=False)-(-1.86968286065)) < 1e-7


def test_ap1rog_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # Old Code Value : -8.87332409253 WTF?
    # FCI Value :      -8.96741814557
    nelec = 4
    hf_dict = hartreefock(fn="test/lih.xyz", basis="sto-6g", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # Compare AP1ROG energy with old code
    # see if we can reproduce HF numbers
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    ap1rog.params *= 0.0
    assert abs(ap1rog.compute_energy(include_nuc=False, ref_sds=ap1rog.default_ref_sds)-(-8.9472891719)) < 1e-7
    # Compare AP1roG energy with old code
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    ap1rog.params[-1] = E_hf
    ap1rog()
    energy = ap1rog.compute_energy(include_nuc=False)
    print("new energy", energy)
    print(ap1rog.params, ap1rog.npair, ap1rog.nspatial-ap1rog.npair)
    assert abs(energy-(-8.96741814557)) < 1e-7
