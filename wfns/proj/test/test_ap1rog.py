from __future__ import absolute_import, division, print_function
import os
import numpy as np

from wfns.proj.solver import solve
from wfns.proj.ap1rog import AP1roG
from wfns.wrapper.horton import gaussian_fchk

#FIXME: Need numbers for AP1roG

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
    # Check if AP1roG converges to the same number by itself
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='cma_guess')
    results = solve(ap1rog, solver_type='least squares', jac=True)
    print('HF energy', -1.84444667247)
    print('new energy', ap1rog.compute_energy())
    print('Old code value', -1.86968286065)
    print('FCI value', -1.87832550029)
    assert results.success
    assert -1.84444667247 > ap1rog.compute_energy() > -1.87832550029
    assert False


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
    # Check if AP1roG converges to the same number by itself
    # FIXME: terrible reference
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='cma_guess')
    results = solve(ap1rog, solver_type='least squares', jac=True)
    print('HF energy', -8.9472891719)
    print('new energy', ap1rog.compute_energy())
    print('Old code value', -8.87332409253 )
    print('FCI value', -8.96741814557)
    assert results.success
    assert -8.9472891719 > ap1rog.compute_energy() > -8.96741814557
    assert False
