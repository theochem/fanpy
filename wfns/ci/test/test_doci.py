from __future__ import absolute_import, division, print_function
import os
import numpy as np
from wfns.ci.solver import solve
from wfns.ci.doci import DOCI
from wfns.wrapper.horton import gaussian_fchk


def test_doci_h2():
    """ Tests DOCI wavefunction for H2 (STO-6G) agaisnt Peter's orbital optimized DOCI

    NOTE
    ----
    Optimized orbitals are read in from Peter's code
    """
    #### H2 ####
    # HF energy: -1.13126983927
    # OO DOCI energy: -1.884948574812363
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h4_square_hf_sto6g.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 4
    E_hf = hf_dict["energy"]
    H = hf_dict["H"][0]
    G = hf_dict["G"][0]
    nuc_nuc = hf_dict["nuc_nuc"]

    # compare HF numbers
    doci = DOCI(nelec=nelec, H=(H,), G=(G,), nuc_nuc=nuc_nuc)
    assert abs(doci.compute_ci_matrix()[0, 0] + doci.nuc_nuc - (-1.131269841877) < 1e-8)

    T = np.array([[0.707106752870, -0.000004484084, 0.000006172115, -0.707106809462, ],
                  [0.707106809472, -0.000004868924, -0.000006704609, 0.707106752852, ],
                  [0.000004942751, 0.707106849959, 0.707106712365, 0.000006630781, ],
                  [0.000004410256, 0.707106712383, -0.707106849949, -0.000006245943, ],]
    )
    H = T.T.dot(H).dot(T)
    G = np.einsum('ijkl,ia->ajkl', G, T)
    G = np.einsum('ajkl,jb->abkl', G, T)
    G = np.einsum('abkl,kc->abcl', G, T)
    G = np.einsum('abcl,ld->abcd', G, T)

    doci = DOCI(nelec=nelec, H=(H,), G=(G,), nuc_nuc=nuc_nuc)
    solve(doci)
    assert abs(doci.compute_energy() - (-1.884948574812363)) < 1e-7

# FIXME: need other tests
