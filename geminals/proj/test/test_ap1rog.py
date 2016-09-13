from __future__ import absolute_import, division, print_function
import os
import numpy as np

from geminals.proj.ap1rog import AP1roG
from geminals.hort import gaussian_fchk
from geminals.hort import ap1rog as old_ap1rog

def test_ap1rog_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # Old code Value:  -1.86968286065
    # FCI Value :      -1.87832550029
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_631gdp.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # see if we can reproduce HF numbers
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    ap1rog.params *= 0.0
    ap1rog.cache = {}
    ap1rog.d_cache = {}
    assert abs(ap1rog.compute_energy(include_nuc=False, ref_sds=ap1rog.default_ref_sds)-(-1.84444667247)) < 1e-7
    # old code results
    old_results = old_ap1rog(fn="test/h2.xyz", basis="6-31g**", nelec=nelec)
    old_energy = old_results['energy'] - nuc_nuc
    old_params = np.hstack((old_results['x'], old_energy))
    # Check if AP1roG converges to the same number if we give the old "converged" AP1roG numbers
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, params=old_params)
    ap1rog()
    energy = ap1rog.compute_energy()
    assert abs(ap1rog.compute_energy(include_nuc=False) - old_energy) < 1e-7
    # Check if AP1roG converges to the same number by itself
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    ap1rog()
    energy = ap1rog.compute_energy()
    print('HF energy', -1.84444667247)
    print('new energy', energy)
    print('Old code value', old_energy)
    print('FCI value', -1.87832550029)
    assert abs(ap1rog.compute_energy(include_nuc=False) - old_energy) < 1e-7


def test_ap1rog_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # Old Code Value : -8.87332409253 WTF?
    # FCI Value :      -8.96741814557
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih_hf_sto6g.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 4
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # see if we can reproduce HF numbers
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    ap1rog.params *= 0.0
    ap1rog.cache = {}
    ap1rog.d_cache = {}
    assert abs(ap1rog.compute_energy(include_nuc=False, ref_sds=ap1rog.default_ref_sds)-(-8.9472891719)) < 1e-7
    # old code results
    old_results = old_ap1rog(fn="test/lih.xyz", basis="sto-6g", nelec=nelec)
    old_energy = old_results['energy'] - nuc_nuc
    old_params = np.hstack((old_results['x'], old_energy))
    # Check if AP1roG converges to the same number if we give the old "converged" AP1roG numbers
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, params=old_params)
    ap1rog()
    energy = ap1rog.compute_energy()
    assert abs(ap1rog.compute_energy(include_nuc=False) - old_energy) < 1e-7
    # Check if AP1roG converges to the same number by itself
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    ap1rog()
    energy = ap1rog.compute_energy()
    print('HF energy', -8.9472891719)
    print("new energy", energy)
    print('Old code value', old_energy)
    print('FCI value', -8.96741814557)
    assert abs(ap1rog.compute_energy(include_nuc=False) - old_energy) < 1e-7
