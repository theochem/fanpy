from __future__ import absolute_import, division, print_function
import numpy as np
from nose.plugins.attrib import attr
from wfns.solver.solver import solve
from wfns.wfn.apfsg import APfsG
from wfns.tools import find_datafile

@attr('slow')
def test_apfsg_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # FCI Value :      -1.87832550029
    nelec = 2
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129
    # Reproduce HF energy
    apfsg = APfsG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    apfsg.params[:-1] = apfsg.template_coeffs.flatten()
    apfsg.cache = {}
    apfsg.d_cache = {}
    assert abs(apfsg.compute_energy(include_nuc=False, ref_sds=apfsg.default_ref_sds) - (-1.84444667247)) < 1e-7
    # Solve with Jacobian using energy as a parameter
    apfsg = APfsG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(apfsg, use_jac=False)
    print('HF energy', -1.84444667247)
    print('new energy', apfsg.compute_energy())
    print('FCI value', -1.87832550029)
    print(apfsg.params)
    assert -1.84444667247 > apfsg.compute_energy() > -1.87832550029
    assert abs(apfsg.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-7


@attr('slow')
def test_apfsg_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # FCI Value :      -8.96741814557
    nelec = 4
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356
    # Reproduce HF energy
    apfsg = APfsG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    apfsg.params[:-1] = apfsg.template_coeffs.flatten()
    apfsg.cache = {}
    apfsg.d_cache = {}
    assert abs(apfsg.compute_energy(include_nuc=False, ref_sds=apfsg.default_ref_sds) - (-8.9472891719)) < 1e-7
    # Solve with Jacobian using energy as a parameter
    apfsg = APfsG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(apfsg, use_jac=False)
    print('HF energy', -8.9472891719)
    print('new energy', apfsg.compute_energy())
    print('FCI value', -8.96741814557)
    assert -8.9472891719 > apfsg.compute_energy() > -8.96741814557
    assert False
