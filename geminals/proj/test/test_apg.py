from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np

from geminals.proj.proj_wavefunction import ProjectionWavefunction
from geminals.proj.apg import APG, generate_pairing_scheme
from geminals.hort import hartreefock

def test_generate_pairing_scheme():
    occ_indices = [0,1,3,4]
    answer = [ [[0,1], [3,4]],
               [[0,3], [1,4]],
               [[0,4], [1,3]]]
    assert answer == list(generate_pairing_scheme(occ_indices))
    occ_indices = [0,1,3,4,6,7]
    answer = [ [[0,1], [3,4], [6,7]],
               [[0,1], [3,6], [4,7]],
               [[0,1], [3,7], [4,6]],
               [[0,3], [1,4], [6,7]],
               [[0,3], [1,6], [4,7]],
               [[0,3], [1,7], [4,6]],
               [[0,4], [1,3], [6,7]],
               [[0,4], [1,6], [3,7]],
               [[0,4], [1,7], [3,6]],
               [[0,6], [1,3], [4,7]],
               [[0,7], [1,3], [4,6]],
               [[0,6], [1,4], [3,7]],
               [[0,7], [1,4], [3,6]],
               [[0,6], [1,7], [3,4]],
               [[0,7], [1,6], [3,4]],
    ]
    assert answer == [sorted(i, key=lambda x: x[0]) for i in sorted(generate_pairing_scheme(occ_indices), key=lambda x:x[0])]

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
