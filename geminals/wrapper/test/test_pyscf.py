from __future__ import absolute_import, division, print_function

import numpy as np

from geminals.wrapper.pyscf import hartreefock

def test_hartreefock():
    """ Tests geminals.wrapper.pyscf.hartreefock

    """
    hf_dict = hartreefock("test/lih.xyz", "sto-6g")
    E_hf, H, G, nuc_nuc = hf_dict["energy"], hf_dict["H"], hf_dict["G"], hf_dict["nuc_nuc"]

    # compare energies with Gaussian results
    # Total energy : -7.95197153880 Hartree
    # Nuclear repulsion energy : 0.9953176337 Hartree
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
