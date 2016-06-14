from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np

from geminals.proj.proj_wavefunction import ProjectionWavefunction
from geminals.proj.apg import APG, ind_from_orbs_to_gem, ind_from_gem_to_orbs, generate_pairing_scheme
from geminals.hort import hartreefock

def test_ind_converter():
    nspin = 100
    counter = 0
    for i in range(nspin):
        # if C_{p;ij} \neq C_{p;ji}
        for j in range(nspin):
        # # if C_{p;ij} = C_{p;ji}
        # for j in range(i):
            if i == j:
                assert_raises(ValueError, lambda: ind_from_orbs_to_gem(i, j, nspin))
            else:
                gem_ind = ind_from_orbs_to_gem(i, j, nspin)
                assert gem_ind == counter
                assert ind_from_gem_to_orbs(gem_ind, nspin) == (i,j)
                counter += 1

def test_generate_pairing_scheme():
    occ_indices = range(4)
    for i in generate_pairing_scheme(occ_indices):
        print(i)

    # schemes = tuple(tuple(sorted(i, key=lambda x:x[0])) for i in generate_pairing_scheme(occ_indices))
    # print(schemes)
    # print(len(schemes))
    # print(len(set(schemes)))

class TestAPGWavefunction(APG):
    def compute_hamiltonian():
        pass
    def normalize():
        pass

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
    apg = TestAPGWavefunction(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False, adjacency=None)
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
