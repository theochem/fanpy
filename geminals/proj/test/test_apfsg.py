from __future__ import absolute_import, division, print_function
import os
import numpy as np
np.random.seed(2012)

from geminals.proj.solver import solve
from geminals.proj.apfsg import APfsG
from geminals.wrapper.horton import gaussian_fchk

def test_apfsg_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # FCI Value :      -1.87832550029
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_631gdp.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # Reproduce HF energy
    apfsg = APfsG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apfsg.params[:-1] = apfsg.template_coeffs.flatten()
    apfsg.cache = {}
    apfsg.d_cache = {}
    assert abs(apfsg.compute_energy(include_nuc=False, ref_sds=apfsg.default_ref_sds) - (-1.84444667247)) < 1e-7
    # Solve with Jacobian using energy as a parameter
    apfsg = APfsG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    solve(apfsg, jac=False)
    print('HF energy', -1.84444667247)
    print('new energy', apfsg.compute_energy())
    print('FCI value', -1.87832550029)
    print(apfsg.params)
    assert -1.84444667247 > apfsg.compute_energy() > -1.87832550029
    assert abs(apfsg.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-7


# def test_apfsg_wavefunction_lih():
#     #### LiH ####
#     # HF Value :       -8.9472891719
#     # FCI Value :      -8.96741814557
#     data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih_hf_sto6g.fchk')
#     hf_dict = gaussian_fchk(data_path)
#
#     nelec = 4
#     E_hf = hf_dict["energy"]
#     H = hf_dict["H"]
#     G = hf_dict["G"]
#     nuc_nuc = hf_dict["nuc_nuc"]
#     # Reproduce HF energy
#     apfsg = APfsG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
#     apfsg.params[:-1] = apfsg.template_coeffs.flatten()
#     apfsg.cache = {}
#     apfsg.d_cache = {}
#     assert abs(apfsg.compute_energy(include_nuc=False, ref_sds=apfsg.default_ref_sds) - (-8.9472891719)) < 1e-7
#     # Solve with Jacobian using energy as a parameter
#     apfsg = APfsG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
#     solve(apfsg, jac=False)
#     print('HF energy', -8.9472891719)
#     print('new energy', apfsg.compute_energy())
#     print('FCI value', -8.96741814557)
#     assert -8.9472891719 > apfsg.compute_energy() > -8.96741814557
#     assert False
test_apfsg_wavefunction_h2()
