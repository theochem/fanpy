from __future__ import absolute_import, division, print_function
from nose.plugins.attrib import attr
from wfns.proj.solver import solve
from wfns.proj.apr2g import APr2G
from wfns.proj.ap1rog import AP1roG
from wfns.wrapper.horton import gaussian_fchk


def test_apr2g_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # Old code Value:  -1.86968286065
    # FCI Value :      -1.87832550029
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')

    nelec = 2
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    # AP1roG as an initial guess
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='cma_guess')
    solve(ap1rog, solver_type='least squares', jac=True)
    # Check if apr2g converges to a reasonable number
    apr2g = APr2G(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apr2g.assign_params(params=ap1rog.params, params_type='ap1rog')
    apr2g.normalize()
    solve(apr2g, solver_type='cma_guess')
    results = solve(apr2g, solver_type='least squares', jac=False)
    print('HF energy', -1.84444667247)
    print('AP1roG energy', ap1rog.compute_energy())
    print('APr2G energy', apr2g.compute_energy())
    print('FCI value', -1.87832550029)
    assert results.success
    assert -1.84444667247 > apr2g.compute_energy() > -1.87832550029
    assert False


@attr('slow')
def test_apr2g_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # FCI Value :      -8.96741814557
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')

    nelec = 4
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    # AP1roG as an initial guess
    ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='cma_guess')
    solve(ap1rog, solver_type='least squares', jac=True)
    # Check if apr2g converges to a reasonable number
    apr2g = APr2G(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apr2g.assign_params(params=ap1rog.params, params_type='ap1rog')
    apr2g.normalize()
    solve(apr2g, solver_type='cma_guess')
    results = solve(apr2g, solver_type='least squares', jac=False)
    print('HF energy', -8.9472891719)
    print('AP1roG energy', ap1rog.compute_energy())
    print('APr2G energy', apr2g.compute_energy())
    print('FCI value', -8.96741814557)
    assert results.success
    assert -8.9472891719 > apr2g.compute_energy() > -8.96741814557
    assert False

# def test_apr2g_wavefunction_li2():
#     #### Li2 ####
#     # HF Value :
#     # FCI Value :
#     hf_dict = gaussian_fchk('test/lih_hf_631g.fchk')
#
#     nelec = 6
#     H = hf_dict["H"]
#     G = hf_dict["G"]
#     nuc_nuc = hf_dict["nuc_nuc_energy"]
#     # AP1roG as an initial guess
#     ap1rog = AP1roG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
#     solve(ap1rog)
#     # Check if apr2g converges to a reasonable number
#     apr2g = APr2G(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
#     apr2g.assign_params(ap1rog_params=ap1rog.params)
#     solve(apr2g, jac=False)
#     energy = apr2g.compute_energy(include_nuc=False)
#     print('HF energy', '?')
#     print('AP1roG energy', ap1rog.compute_energy())
#     print('APr2G energy', energy)
#     print('FCI value', '?')
#    # assert abs(energy - (-14.796070)) < 1e-7
#     assert False
