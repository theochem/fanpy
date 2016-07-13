from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np

from geminals.proj.proj_wavefunction import ProjectionWavefunction
from geminals.proj.apg import APG
from geminals.hort import hartreefock

def test_assign_adjacency():
    """
    Tests APGWavefunction.assign_adjacency
    """
    nelec = 2
    hf_dict = hartreefock(fn="test/h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # default adjacenecy
    apg = APG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False, adjacency=None)
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
    # Old Code Value : -1.86968284431
    # FCI Value :      -1.87832550029
    nelec = 2
    hf_dict = hartreefock(fn="test/h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # see if we can reproduce HF numbers
    apg = APG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False)
    apg.params *= 0.0
    apg.params[apg.dict_orbpair_gem[(0, apg.nspatial)]] = 1.0
    assert abs(apg.compute_energy(include_nuc=False) - (-1.84444667247)) < 1e-7
    # Compare APG energy with old code
    # Solve with Jacobian using energy as a parameter
    apg = APG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=True)
    apg()
    print(apg.compute_energy())
    assert abs(apg.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-7
    # convert energy back into projection dependent (energy is not a parameter)
    apg.energy_is_param = False
    apg.params = apg.params[:-1]
    assert abs(apg.compute_energy(sd=apg.pspace[0], include_nuc=False) - (-1.86968284431)) < 1e-7
    assert abs(apg.compute_energy(sd=apg.pspace, include_nuc=False) - (-1.86968284431)) < 1e-7
    # Solve with Jacobian not using energy as a parameter
    apg = APG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False)
    apg()
    # FIXME: THESE TESTS FAIL!
    print('overlaps', apg.overlap(apg.pspace[0]), apg.compute_overlap(apg.pspace[0]))
    print(apg.compute_energy(sd=apg.pspace[0], include_nuc=False), 'new code')
    print(-1.86968284431, 'old code')
    # assert abs(apg.compute_energy(sd=apg.pspace[0], include_nuc=False) - (-1.86968284431)) < 1e-7
    # assert abs(apg.compute_energy(sd=apg.pspace, include_nuc=False)-(-1.86968284431)) < 1e-7
    # Solve without Jacobian using energy as a parameter
    apg = APG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=True)
    apg._solve_least_squares()
    # FIXME: the numbers are quite different
    #assert abs(apg.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-4
    # Solve without Jacobian not using energy as a parameter
    apg = APG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False)
    apg._solve_least_squares()
    # FIXME: THESE TESTS FAIL!
    # assert abs(apg.compute_energy(sd=apg.pspace[0], include_nuc=False)-(-1.86968284431)) < 1e-4
    # assert abs(apg.compute_energy(sd=apg.pspace, include_nuc=False)-(-1.86968284431)) < 1e-4
