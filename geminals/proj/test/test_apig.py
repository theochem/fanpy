from __future__ import absolute_import, division, print_function
import numpy as np
np.random.seed(2012)

from geminals.proj.apig import APIG
from geminals.hort import hartreefock

def test_apig_wavefunction_h2():
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
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    apig.params[:-1] = apig.template_coeffs.flatten()
    assert abs(apig.compute_energy(include_nuc=False, ref_sds=apig.default_ref_sds) - (-1.84444667247)) < 1e-7
    # Compare APIG energy with old code
    # Solve with Jacobian using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, dtype=np.float64)
    init_guess = apig.params[:]
    apig()
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
    nelec = 4
    hf_dict = hartreefock(fn="test/lih.xyz", basis="sto-6g", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    apig.params[:-1] = apig.template_coeffs.flatten()
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
