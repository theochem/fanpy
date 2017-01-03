from __future__ import absolute_import, division, print_function
from nose.plugins.attrib import attr
from wfns.proj.solver import solve
from wfns.proj.apsetg import APsetG
from wfns.wrapper.horton import gaussian_fchk


@attr('slow')
def test_apsetg_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # Old Code Value : -1.86968284431
    # FCI Value :      -1.87832550029
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')

    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    # Solve with Jacobian using energy as a parameter
    apsetg = APsetG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(apsetg, solver_type='cma_guess')
    results = solve(apsetg, solver_type='least squares', jac=True)
    print('HF energy', -1.84444667247)
    print('new energy', apsetg.compute_energy())
    print('FCI value', -1.87832550029)
    assert results.success
    assert -1.84444667247 > apsetg.compute_energy() > -1.87832550029
    assert abs(apsetg.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-7


@attr('slow')
def test_apsetg_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # Old Code Value : -8.96353105152
    # FCI Value :      -8.96741814557
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')

    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    # Compare apsetg energy with old code
    # Solve with Jacobian using energy as a parameter
    apsetg = APsetG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(apsetg, solver_type='cma_guess')
    results = solve(apsetg, solver_type='least squares', jac=True)
    print('HF energy', -8.9472891719)
    print('new energy', apsetg.compute_energy())
    print('FCI value', -8.96741814557)
    assert results.success
    assert -8.9472891719 > apsetg.compute_energy() > -8.96741814557
    assert False
