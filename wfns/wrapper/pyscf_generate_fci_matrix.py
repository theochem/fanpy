"""Script for generating one- and two-electron integrals using PySCF.

Functions
---------
generate_fci_cimatrix(h1e, eri, nelec, is_chemist_notation=False)
    Generate the FCI Hamiltonian CI matrix.

"""
import sys
import ctypes
import numpy as np
from pyscf.lib import load_library, hermi_triu
from pyscf.fci import cistring

LIBFCI = load_library("libfci")


def generate_fci_cimatrix(h1e, eri, nelec, is_chemist_notation=False):
    """Construct the FCI CI Hamiltonian matrix using PySCF.

    Parameters
    ----------
    h1e : np.ndarray(K, K)
        One electron integrals.
    eri : np.ndarray(K, K, K, K)
        Two electron integrals.
    nelec : int
        Number of electrons.
    is_chemist_notation : bool
        Flag to set the notation for the two electron integrals.
        By default, it is assumed that the Physicist's notation is used for the two electron
        integrals.

    Returns
    -------
    ci_matrix : np.ndarray(M, M)
        CI Hamiltonian matrix.
    pspace : list(M)
        List of the Slater determinants (in bitstring) that corresponds to each row/column of the
        `ci_matrix`.

    Raises
    ------
    ValueError
        If number of electrons is invalid.

    """
    if not is_chemist_notation:
        eri = np.einsum("ijkl->ikjl", eri)
    # adapted/copied from pyscf.fci.direct_spin1.make_hdiag
    # number of spatial orbitals
    norb = h1e.shape[0]
    # number of electrons
    if isinstance(nelec, (int, np.number)):
        # beta
        nelecb = nelec // 2
        # alpha
        neleca = nelec - nelecb
    elif isinstance(nelec, (tuple, list)) and len(nelec) == 2:
        neleca, nelecb = nelec
    else:
        raise ValueError("Unsupported electron number, {0}".format(nelec))
    # integrals
    h1e = np.asarray(h1e, order="C")
    eri = np.asarray(eri, order="C")
    # Construct some sort of lookup table to link different bit string occupations
    # to one another. i.e. From one bit string, and several indices that describes
    # certain excitation, we can get the other bit string
    # NOTE: PySCF treats alpha and the beta bits separately
    occslista = np.asarray(cistring._gen_occslst(range(norb), neleca))
    occslistb = np.asarray(cistring._gen_occslst(range(norb), nelecb))
    # number of Slater determinants
    na = len(occslista)  # number of "alpha" Slater determinants
    nb = len(occslistb)  # number of "beta" Slater determinants
    num_sd = na * nb  # number of Slater determinants in total

    # Diagonal of CI Hamiltonian matrix
    hdiag = np.empty(num_sd)
    # Coulomb integrals
    jdiag = np.asarray(np.einsum("iijj->ij", eri), order="C")
    # Exchange integrals
    kdiag = np.asarray(np.einsum("ijji->ij", eri), order="C")
    # Fucking Magic
    LIBFCI.FCImake_hdiag_uhf(
        hdiag.ctypes.data_as(ctypes.c_void_p),
        h1e.ctypes.data_as(ctypes.c_void_p),
        h1e.ctypes.data_as(ctypes.c_void_p),
        jdiag.ctypes.data_as(ctypes.c_void_p),
        jdiag.ctypes.data_as(ctypes.c_void_p),
        jdiag.ctypes.data_as(ctypes.c_void_p),
        kdiag.ctypes.data_as(ctypes.c_void_p),
        kdiag.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(norb),
        ctypes.c_int(na),
        ctypes.c_int(nb),
        ctypes.c_int(neleca),
        ctypes.c_int(nelecb),
        occslista.ctypes.data_as(ctypes.c_void_p),
        occslistb.ctypes.data_as(ctypes.c_void_p),
    )

    # adapted/copied from pyscf.fci.direct_spin1.pspace
    # PySCF has a fancy indicing of Slater determinants (bitstrings to consecutive integers)
    addr = np.arange(hdiag.size)
    # again, separate the alpha and the beta parts
    addra, addrb = divmod(addr, nb)
    # bit strings for the alpha and beta parts
    stra = cistring.addrs2str(norb, neleca, addra)
    strb = cistring.addrs2str(norb, nelecb, addrb)
    # number of slater determinants
    ci_matrix = np.zeros((num_sd, num_sd))
    # More Fucking Magic
    LIBFCI.FCIpspace_h0tril(
        ci_matrix.ctypes.data_as(ctypes.c_void_p),
        h1e.ctypes.data_as(ctypes.c_void_p),
        eri.ctypes.data_as(ctypes.c_void_p),
        stra.ctypes.data_as(ctypes.c_void_p),
        strb.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(norb),
        ctypes.c_int(num_sd),
    )

    for i in range(num_sd):
        ci_matrix[i, i] = hdiag[addr[i]]
    ci_matrix = hermi_triu(ci_matrix)

    # convert  PySCF Slater determinant notation to one that we use
    pspace = []
    for i in addr:
        # beta b/c the modulus corresponds to the "beta slater determinant" index
        addra, addrb = divmod(i, nb)
        alpha_bit = cistring.addr2str(norb, neleca, addra)
        beta_bit = cistring.addr2str(norb, nelecb, addrb)
        # hard code in slater.combine_spin
        pspace.append(alpha_bit | (beta_bit << norb))

    return ci_matrix, pspace


if __name__ == "__main__":
    # extract keyword from command line
    kwargs = {key: val for key, val in zip(sys.argv[3::2], sys.argv[4::2])}
    # change data types
    if "h1e" in kwargs:
        kwargs["h1e"] = np.load(kwargs["h1e"])
    if "eri" in kwargs:
        kwargs["eri"] = np.load(kwargs["eri"])
    if "nelec" in kwargs:
        kwargs["nelec"] = int(kwargs["nelec"])
    if "is_chemist_notation" in kwargs:
        kwargs["is_chemist_notation"] = kwargs["is_chemist_notation"] == "True"

    ci_matrix, pspace = generate_fci_cimatrix(**kwargs)
    np.save(sys.argv[1], ci_matrix)
    np.save(sys.argv[2], pspace)
