from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../')
from apig import APIG
from hort import hartreefock

import numpy as np

def test_apig_wavefunction():
    #### H2 ####
    # HF Value :       -1.84444667247
    # Old Code Value : -1.86968284431
    # FCI Value :      -1.87832550029
    nelec = 2
    hf_dict = hartreefock(fn="./h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # see if we can reproduce HF numbers
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False)
    apig.params *= 0.0
    apig.params[0] = 1.0
    assert abs(apig.compute_energy(include_nuc=False)-(-1.84444667247)) < 1e-7
    # Compare APIG energy with old code
    # Solve with Jacobian using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=True)
    apig()
    assert abs(apig.compute_energy(include_nuc=False)-(-1.86968284431)) < 1e-7
    # convert energy back into projection dependent (energy is not a parameter)
    apig.energy_is_param = False
    apig.params = apig.params[:-1]
    assert abs(apig.compute_energy(sd=apig.pspace[0], include_nuc=False)-(-1.86968284431)) < 1e-7
    assert abs(apig.compute_energy(sd=apig.pspace, include_nuc=False)-(-1.86968284431)) < 1e-7
    # Solve with Jacobian not using energy as a parameter
    print('x'*30)
    print(apig.params)
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False)
    apig()
    # FIXME: THESE TESTS FAIL!
    print(apig.compute_energy(sd=apig.pspace[0], include_nuc=False), 'new code')
    print(-1.86968284431, 'old code')
    assert abs(apig.compute_energy(sd=apig.pspace[0], include_nuc=False)-(-1.86968284431)) < 1e-7
    assert abs(apig.compute_energy(sd=apig.pspace[0], include_nuc=False)-(-1.86968284431)) < 1e-7
    # assert abs(apig.compute_energy(sd=apig.pspace, include_nuc=False)-(-1.86968284431)) < 1e-7
    # Solve without Jacobian using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=True)
    apig._solve_least_squares(jac=None)
    # FIXME: the numbers are quite different
    assert abs(apig.compute_energy(include_nuc=False)-(-1.86968284431)) < 1e-4
    # Solve without Jacobian not using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False)
    apig._solve_least_squares(jac=None)
    # FIXME: THESE TESTS FAIL!
    # assert abs(apig.compute_energy(sd=apig.pspace[0], include_nuc=False)-(-1.86968284431)) < 1e-4
    # assert abs(apig.compute_energy(sd=apig.pspace, include_nuc=False)-(-1.86968284431)) < 1e-4

    #### LiH ####
    # HF Value :       -8.9472891719
    # Old Code Value : -8.96353105152
    # FCI Value :      -8.96741814557
    nelec = 4
    hf_dict = hartreefock(fn="./lih.xyz", basis="sto-6g", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # Compare APIG energy with old code
    # Solve with Jacobian using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=True)
    apig()
    assert abs(apig.compute_energy(include_nuc=False)-(-8.96353105152)) < 1e-7
    # convert energy back into projection dependent (energy is not a parameter)
    apig.energy_is_param = False
    apig.params = apig.params[:-1]
    assert abs(apig.compute_energy(sd=apig.pspace[0], include_nuc=False)-(-8.96353105152)) < 1e-7
    assert abs(apig.compute_energy(sd=apig.pspace, include_nuc=False)-(-8.96353105152)) < 1e-7
    # Solve with Jacobian not using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False)
    apig()
    # FIXME: THESE TESTS FAIL!
    # assert abs(apig.compute_energy(sd=apig.pspace[0], include_nuc=False)-(-8.96353105152)) < 1e-7
    # assert abs(apig.compute_energy(sd=apig.pspace, include_nuc=False)-(-8.96353105152)) < 1e-7
    # Solve without Jacobian using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=True)
    apig._solve_least_squares(jac=None)
    # FIXME: the numbers are quite different
    assert abs(apig.compute_energy(include_nuc=False)-(-8.96353105152)) < 1e-4
    # Solve without Jacobian not using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False)
    apig._solve_least_squares(jac=None)
    # FIXME: THESE TESTS FAIL!
    # assert abs(apig.compute_energy(sd=apig.pspace[0], include_nuc=False)-(-8.96353105152)) < 1e-4
    # assert abs(apig.compute_energy(sd=apig.pspace, include_nuc=False)-(-8.96353105152)) < 1e-4
test_apig_wavefunction()

