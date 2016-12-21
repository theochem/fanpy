from __future__ import absolute_import, division, print_function
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
    hf_dict = gaussian_fchk('test/h4_square_hf_sto6g.fchk')

    nelec = 4
    one_int = hf_dict["one_int"][0]
    two_int = hf_dict["two_int"][0]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    # compare HF numbers
    doci = DOCI(nelec=nelec, one_int=(one_int,), two_int=(two_int,), nuc_nuc=nuc_nuc)
    assert abs(doci.compute_ci_matrix()[0, 0] + doci.nuc_nuc - (-1.131269841877) < 1e-8)

    T = np.array([[0.707106752870, -0.000004484084, 0.000006172115, -0.707106809462, ],
                  [0.707106809472, -0.000004868924, -0.000006704609, 0.707106752852, ],
                  [0.000004942751, 0.707106849959, 0.707106712365, 0.000006630781, ],
                  [0.000004410256, 0.707106712383, -0.707106849949, -0.000006245943, ],]
    )
    one_int = T.T.dot(one_int).dot(T)
    two_int = np.einsum('ijkl,ia->ajkl', two_int, T)
    two_int = np.einsum('ajkl,jb->abkl', two_int, T)
    two_int = np.einsum('abkl,kc->abcl', two_int, T)
    two_int = np.einsum('abcl,ld->abcd', two_int, T)

    doci = DOCI(nelec=nelec, one_int=(one_int,), two_int=(two_int,), nuc_nuc=nuc_nuc)
    solve(doci)
    assert abs(doci.compute_energy() - (-1.884948574812363)) < 1e-7

# FIXME: need other tests
