from __future__ import absolute_import, division, print_function
import os
import numpy as np
from nose.plugins.attrib import attr

from wfns.proj.apig import APIG
from wfns.proj.solver import solve
from wfns.wrapper.horton import gaussian_fchk

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
    solve(apig, solver_type='cma_guess')
    results = solve(apig, solver_type='least squares', jac=True)
    assert results.success
    print('HF Energy: -1.84444667247')
    print('APIG Energy: {0}'.format(apig.compute_energy(include_nuc=False)))
    print('APIG Energy (old code): -1.86968284431')
    print('FCI Energy: -1.87832550029')
    assert abs(apig.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-7
    # Solve without Jacobian using energy as a parameter
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='cma_guess')
    results = solve(apig, solver_type='least squares', jac=False)
    assert results.success
    print('HF Energy: -1.84444667247')
    print('APIG Energy: {0}'.format(apig.compute_energy(include_nuc=False)))
    print('APIG Energy (old code): -1.86968284431')
    print('FCI Energy: -1.87832550029')
    # Note: least squares solver without a jacobian isn't very good
    assert abs(apig.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-4


@attr('slow')
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
    apig = APIG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apig.normalize()
    # guess with cma
    solve(apig, solver_type='cma_guess')
    # finish solving with least squares
    results = solve(apig, solver_type='least squares', jac=True)
    assert results.success
    print('HF Energy: -8.9472891719')
    print('APIG Energy: {0}'.format(apig.compute_energy(include_nuc=False)))
    print('APIG Energy (old code): -8.96353105152')
    print('FCI Energy: -8.96741814557')
    assert abs(apig.compute_energy(include_nuc=False) - (-8.96353105152)) < 1e-6
