from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.linalg
import os
from geminals.wrapper.pyscf import hartreefock, generate_ci_matrix

def test_hartreefock():
    """ Tests geminals.wrapper.pyscf.hartreefock

    """
    hf_dict = hartreefock("test/lih.xyz", "sto-6g")
    E_hf, H, G, nuc_nuc = hf_dict["energy"], hf_dict["H"], hf_dict["G"], hf_dict["nuc_nuc"]

    # compare energies with Gaussian results
    # Total energy : -7.95197153880 Hartree
    # Nuclear repulsion energy : 0.9953176337 Hartree
    print(abs(E_hf - (-7.95197153880 - 0.9953176337)))
    assert abs(E_hf - (-7.95197153880 - 0.9953176337)) < 1e-8
    assert abs(nuc_nuc - (0.9953176337)) < 1e-8
    # check types of the integrals
    assert isinstance(H, tuple)
    for i in H:
        assert isinstance(i, np.ndarray)
    assert isinstance(G, tuple)
    for i in G:
        assert isinstance(i, np.ndarray)
    for matrix in H+G:
        assert np.all(np.array(matrix.shape) == H[0].shape[0])

def test_generate_ci_matrix():
    """ Tests geminals.wrapper.pyscf.generate_ci_matrix

    Code is tested by seeing if CI matrix can be used to create the FCI energies of H2 and LiH
    """
    #### H2 ####
    # HF energy: -1.13126983927
    # FCI energy: -1.1651487496
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2.xyz')
    hf_dict = hartreefock(data_path, '6-31gs')

    nelec = 2
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]

    ci_matrix, pspace = generate_ci_matrix(H[0], G[0], nelec)
    ground_energy = scipy.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-1.1651486697)) < 1e-7

    #### LiH ####
    # HF energy: -7.95197153880
    # FCI energy: -7.9723355823
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih.xyz')
    hf_dict = hartreefock(data_path, 'sto-6g')

    nelec = 4
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]

    ci_matrix, pspace = generate_ci_matrix(H[0], G[0], nelec)
    ground_energy = scipy.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-7.9723355823)) < 1e-7
