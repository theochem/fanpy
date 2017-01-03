from __future__ import absolute_import, division, print_function
from wfns.proj.solver import solve
from wfns.proj.ap1rog import AP1roG
from wfns.wrapper.horton import gaussian_fchk

#FIXME: Need numbers for AP1roG

def test_ap1rog_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # Old code Value:  -1.86968286065
    # FCI Value :      -1.87832550029
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')

    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    # see if we can reproduce HF numbers
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    ap1rog.params *= 0.0
    ap1rog.cache = {}
    ap1rog.d_cache = {}
    assert abs(ap1rog.compute_energy(include_nuc=False, ref_sds=ap1rog.default_ref_sds)-(-1.84444667247)) < 1e-7
    # Check if AP1roG converges to the same number by itself
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
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
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')

    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    # see if we can reproduce HF numbers
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    ap1rog.params *= 0.0
    ap1rog.cache = {}
    ap1rog.d_cache = {}
    assert abs(ap1rog.compute_energy(include_nuc=False, ref_sds=ap1rog.default_ref_sds)-(-8.9472891719)) < 1e-7
    # Check if AP1roG converges to the same number by itself
    # FIXME: terrible reference
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='cma_guess')
    results = solve(ap1rog, solver_type='least squares', jac=True)
    print('HF energy', -8.9472891719)
    print('new energy', ap1rog.compute_energy())
    print('Old code value', -8.87332409253 )
    print('FCI value', -8.96741814557)
    assert results.success
    assert -8.9472891719 > ap1rog.compute_energy() > -8.96741814557
    assert False
