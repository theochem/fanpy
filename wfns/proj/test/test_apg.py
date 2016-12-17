from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import os
import numpy as np

from wfns.proj.proj_wavefunction import ProjectionWavefunction
from wfns.proj.solver import solve
from wfns.proj.apg import APG
from wfns.wrapper.horton import gaussian_fchk

def test_assign_adjacency():
    """
    Tests APGWavefunction.assign_adjacency
    """
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_631gdp.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # default adjacenecy
    apg = APG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, adjacency=None)
    apg.assign_adjacency(adjacency=None)
    nspin = apg.nspin
    adjacency = np.identity(nspin, dtype=bool)
    adjacency = -adjacency
    assert np.allclose(apg.adjacency, adjacency)
    # give adjacency
    adjacency = np.zeros((nspin, nspin), dtype=bool)
    adjacency[[0,1],[1,0]] = True
    adjacency[[2,4],[4,2]] = True
    apg.assign_adjacency(adjacency=adjacency)
    assert np.allclose(apg.adjacency, adjacency)
    # bad adjacency
    test = np.identity(nspin, dtype=bool)
    test = -test
    test = adjacency.tolist()
    assert_raises(TypeError, lambda:apg.assign_adjacency(test))
    test = np.identity(nspin, dtype=bool)
    test = -test
    test = test.astype(int)
    assert_raises(TypeError, lambda:apg.assign_adjacency(test))
    test = np.zeros((nspin, nspin+1), dtype=bool)
    assert_raises(ValueError, lambda:apg.assign_adjacency(test))
    test = np.zeros((nspin, nspin), dtype=bool)
    test[[0,1],[1,2]] = True
    assert_raises(ValueError, lambda:apg.assign_adjacency(test))
    test = np.zeros((nspin, nspin), dtype=bool)
    test[[0,1],[1,0]] = True
    test[0,0] = True
    assert_raises(ValueError, lambda:apg.assign_adjacency(test))

def test_apg_wavefunction_h2():
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
    # see if we can reproduce HF numbers
    apg = APG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apg.params *= 0.0
    apg.cache = {}
    apg.d_cache = {}
    apg.params[apg.dict_orbpair_gem[(0, apg.nspatial)]] = 1.0
    assert abs(apg.compute_energy(include_nuc=False, ref_sds=apg.default_ref_sds) - (-1.84444667247)) < 1e-7
    # Compare APG energy with old code
    # Solve with Jacobian using energy as a parameter
    apg = APG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apg.params[-1] = -1.87832550029
    solve(apg, solver_type='cma_guess')
    results = solve(apg, solver_type='least squares', jac=True)
    print('HF energy', -1.84444667247)
    print('APG energy', apg.compute_energy())
    print('FCI value', -1.87832550029)
    assert results.success
    assert abs(apg.compute_energy(include_nuc=False) - (-1.87832550029)) < 1e-7


def test_apg_wavefunction_lih():
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
    # Compare apg energy with old code
    # Solve with Jacobian using energy as a parameter
    apg = APG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    solve(apg, solver_type='cma_guess')
    results = solve(apg, solver_type='least squares', jac=True)
    print('HF energy', -8.9472891719)
    print('APG energy', apg.compute_energy())
    print('FCI value', -8.96741814557)
    assert results.success
    assert -8.9472891719 > apg.computer_energy() > -8.96741814557
    assert False
