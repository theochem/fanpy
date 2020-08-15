"""Script for utilizing PySCF.

Functions
---------
hartreefock(xyz_file, basis, is_unrestricted=False)
    Runs HF in PySCF and generates the corresponding one- and two-electron integrals.

"""
# pylint: disable=C0103
import ctypes
import os

import numpy as np

from pyscf import ao2mo, gto, scf
from pyscf.fci import cistring
from pyscf.lib import hermi_triu, load_library

LIBFCI = load_library("libfci")


def hartreefock(xyz_file, basis, is_unrestricted=False):
    """Run HF using PySCF.

    Parameters
    ----------
    xyz_file : str
        XYZ file location.
        Units are in Angstrom.
    basis : str
        Basis set available in PySCF.
    is_unrestricted : bool
        Flag to run unrestricted HF.
        Default is restricted HF.

    Returns
    -------
    result : dict
        "hf_energy"
            The electronic energy.
        "nuc_nuc"
            The nuclear repulsion energy.
        "one_int"
            The tuple of the one-electron interal.
        "two_int"
            The tuple of the two-electron integral in Physicist's notation.

    Raises
    ------
    ValueError
        If given xyz file does not exist.
    NotImplementedError
        If calculation is unrestricted or generalized.

    """
    # check xyz file
    cwd = os.path.dirname(__file__)
    if os.path.isfile(os.path.join(cwd, xyz_file)):
        xyz_file = os.path.join(cwd, xyz_file)
    elif not os.path.isfile(xyz_file):  # pragma: no branch
        raise ValueError("Given xyz_file does not exist")

    # get coordinates
    with open(xyz_file, "r") as f:
        lines = [i.strip() for i in f.readlines()[2:]]
        atoms = ";".join(lines)

    # get mol
    mol = gto.M(atom=atoms, basis=basis, parse_arg=False, unit="angstrom")

    # get hf
    if is_unrestricted:
        raise NotImplementedError(
            "Unrestricted or Generalized orbitals are not supported in this" " PySCF wrapper (yet)."
        )
    hf = scf.RHF(mol)
    # run hf
    hf.scf()
    # energies
    energy_nuc = hf.energy_nuc()
    energy_tot = hf.kernel()  # HF is solved here
    energy_elec = energy_tot - energy_nuc
    # mo_coeffs
    mo_coeff = hf.mo_coeff
    # Get integrals (See pyscf.gto.moleintor.getints_by_shell for other types of integrals)
    # get 1e integral
    one_int_ab = mol.intor("cint1e_nuc_sph") + mol.intor("cint1e_kin_sph")
    one_int = mo_coeff.T.dot(one_int_ab).dot(mo_coeff)
    # get 2e integral
    eri = ao2mo.full(mol, mo_coeff, verbose=0, intor="cint2e_sph")
    two_int = ao2mo.restore(1, eri, mol.nao_nr())
    # NOTE: PySCF uses Chemist's notation
    two_int = np.einsum("ijkl->ikjl", two_int)
    # results
    result = {
        "hf_energy": energy_elec,
        "nuc_nuc": energy_nuc,
        "one_int": one_int,
        "two_int": two_int,
    }
    return result


def fci_cimatrix(h1e, eri, nelec, is_chemist_notation=False):
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
