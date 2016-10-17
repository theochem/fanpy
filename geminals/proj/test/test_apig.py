from __future__ import absolute_import, division, print_function
import os
import numpy as np
np.random.seed(2012)

from geminals.proj.apig import APIG
from geminals.proj.solver import solve
from geminals.wrapper.horton import gaussian_fchk

def test_apig_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # Old Code Value : -1.86968284431
    # FCI Value :      -1.87832550029
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_631gdp.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    apig.params[:-1] = apig.template_coeffs.flatten()
    apig.cache = {}
    apig.d_cache = {}
    assert abs(apig.compute_energy(include_nuc=False, ref_sds=apig.default_ref_sds) - (-1.84444667247)) < 1e-7
    # Compare APIG energy with old code
    # Solve with Jacobian using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, dtype=np.float64)
    init_guess = apig.params[:]
    solve(apig)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-7
    # Solve without Jacobian using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apig._solve_least_squares(jac=None)
    # FIXME: the numbers are quite different
    assert abs(apig.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-4

def test_apig_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # Old Code Value : -8.96353105152
    # FCI Value :      -8.96741814557
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih_hf_sto6g.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 4
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    apig.params[:-1] = apig.template_coeffs.flatten()
    apig.cache = {}
    apig.d_cache = {}
    assert abs(apig.compute_energy(include_nuc=False, ref_sds=apig.default_ref_sds) - (-8.9472891719)) < 1e-7
    # Compare APIG energy with old code
    # Solve with Jacobian using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apig()
    print(apig.compute_energy(include_nuc=False), 'new code')
    print(-8.96353105152, 'old code')
    assert abs(apig.compute_energy(include_nuc=False) - (-8.96353105152)) < 1e-7
    # Solve without Jacobian using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apig._solve_least_squares(jac=None)
    # FIXME: the numbers are quite different
    assert abs(apig.compute_energy(include_nuc=False) - (-8.96353105152)) < 1e-4
