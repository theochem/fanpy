""" Wrapper of PySCF

At this moment, we mainly need the integrals (in MO basis) and the energies (electron, nuclear
nuclear repulsion) to construct a wavefunction. Here, we use PySCF to get these values.

Functions
---------
hartreefock(xyz_file, basis, is_unrestricted=False)
    Runs HF in PySCF
generate_fci_cimatrix(h1e, eri, nelec, is_chemist_notation=False)
    Generates the FCI Hamiltonian CI matrix
"""
from __future__ import absolute_import, division, print_function
import os
import ctypes
import numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.lib import load_library, hermi_triu
from pyscf.fci import cistring
from .. import __file__ as package_path
from .. import slater
from ..tools import find_datafile

__all__ = []

LIBFCI = load_library('libfci')

def hartreefock(xyz_file, basis, is_unrestricted=False):
    """ Runs HF using PySCF

    Parameters
    ----------
    xyz_file : str
        XYZ file location
    basis : str
        Basis set available in PySCF
    is_unrestricted : bool
        Flag to run unrestricted HF
        Default is restricted HF

    Returns
    -------
    result : dict
        "el_energy"
            electronic energy
        "nuc_nuc_energy"
            nuclear repulsion energy
        "one_int"
            tuple of the one-electron interal
        "two_int"
            tuple of the two-electron integral in Physicist's notation

    Raises
    ------
    ValueError
        If given xyz file does not exist
    NotImplementedError
        If calculation is unrestricted or generalized
    """
    # check xyz file
    cwd = os.path.dirname(__file__)
    if os.path.isfile(xyz_file):
        pass
    elif os.path.isfile(os.path.join(cwd, xyz_file)):
        xyz_file = os.path.join(cwd, xyz_file)
    else:
        try:
            xyz_file = find_datafile(xyz_file)
        except IOError:
            raise ValueError('Given xyz_file does not exist')

   # get coordinates
    with open(xyz_file, 'r') as f:
        lines = [i.strip() for i in f.readlines()[2:]]
        atoms = ';'.join(lines)

    # get mol
    mol = gto.M(atom=atoms, basis=basis, parse_arg=False, unit='angstrom')

    # get hf
    if is_unrestricted:
        raise NotImplementedError('Unrestricted or Generalized orbitals are not supported in this'
                                  ' PySCF wrapper (yet).')
    hf = scf.RHF(mol)
    # run hf
    hf.scf()
    # energies
    E_nuc = hf.energy_nuc()
    E_tot = hf.kernel() # HF is solved here
    E_elec = E_tot - E_nuc
    # mo_coeffs
    mo_coeff = hf.mo_coeff
    # Get integrals (See pyscf.gto.moleintor.getints_by_shell for other types of integrals)
    # get 1e integral
    one_int_ab = mol.intor_symmetric('cint1e_kin_sph') + mol.intor_symmetric('cint1e_nuc_sph')
    one_int = mo_coeff.T.dot(one_int_ab).dot(mo_coeff)
    # get 2e integral
    eri = ao2mo.full(mol, mo_coeff, verbose=0, intor='cint2e_sph')
    two_int = ao2mo.restore(1, eri, mol.nao_nr())
    # NOTE: PySCF uses Chemist's notation
    two_int = np.einsum('ijkl->ikjl', two_int)
    # results
    result = {'el_energy' : E_elec,
              'nuc_nuc_energy' : E_nuc,
              'one_int' : (one_int,),
              'two_int' : (two_int,)}
    return result


def generate_fci_cimatrix(h1e, eri, nelec, is_chemist_notation=False):
    """ Constructs the FCI CI Hamiltonian matrix using PySCF

    Parameters
    ----------
    h1e : np.ndarray(K, K)
        One electron integrals
    eri : np.ndarray(K, K, K, K)
        Two electron integrals
    nelec : int
        Number of electrons
    is_chemist_notation : bool
        Flag to set the notation for the two electron integrals
        By default, it is assumed that the Physicist's notation is used for the
        two electron integrals

    Returns
    -------
    ci_matrix : np.ndarray(M, M)
        CI Hamiltonian matrix
    pspace : list(M)
        List of the Slater determinants (in bitstring) that corresponds to each
        row/column of the ci_matrix

    Raises
    ------
    ValueError
        If number of electrons is invalid
    """
    if not is_chemist_notation:
        eri = np.einsum('ijkl->ikjl', eri)
    # adapted/copied from pyscf.fci.direct_spin1.make_hdiag
    # number of spatial orbitals
    norb = h1e.shape[0]
    # number of electrons
    if isinstance(nelec, (int, np.number)):
        # beta
        nelecb = nelec//2
        # alpha
        neleca = nelec - nelecb
    elif isinstance(nelec, (tuple, list)) and len(nelec) == 2:
        neleca, nelecb = nelec
    else:
        raise ValueError('Unsupported electron number, {0}'.format(nelec))
    # integrals
    h1e = np.ascontiguousarray(h1e)
    eri = np.ascontiguousarray(eri)
    # Construct some sort of lookup table to link different bit string occupations
    # to one another. i.e. From one bit string, and several indices that describes
    # certain excitation, we can get the other bit string
    # NOTE: PySCF treats alpha and the beta bits separately
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    # number of Slater determinants
    na = link_indexa.shape[0] # number of "alpha" Slater determinants
    nb = link_indexb.shape[0] # number of "beta" Slater determinants
    num_sd = na*nb # number of Slater determinants in total

    occslista = np.asarray(link_indexa[:, :neleca, 0], order='C')
    occslistb = np.asarray(link_indexb[:, :nelecb, 0], order='C')
    # Diagonal of CI Hamiltonian matrix
    hdiag = np.empty(num_sd)
    # Coulomb integrals
    jdiag = np.asarray(np.einsum('iijj->ij', eri), order='C')
    # Exchange integrals
    kdiag = np.asarray(np.einsum('ijji->ij', eri), order='C')
    # Fucking Magic
    LIBFCI.FCImake_hdiag_uhf(hdiag.ctypes.data_as(ctypes.c_void_p),
                             h1e.ctypes.data_as(ctypes.c_void_p),
                             h1e.ctypes.data_as(ctypes.c_void_p),
                             jdiag.ctypes.data_as(ctypes.c_void_p),
                             jdiag.ctypes.data_as(ctypes.c_void_p),
                             jdiag.ctypes.data_as(ctypes.c_void_p),
                             kdiag.ctypes.data_as(ctypes.c_void_p),
                             kdiag.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(norb),
                             ctypes.c_int(na), ctypes.c_int(nb),
                             ctypes.c_int(neleca), ctypes.c_int(nelecb),
                             occslista.ctypes.data_as(ctypes.c_void_p),
                             occslistb.ctypes.data_as(ctypes.c_void_p))
    hdiag = np.asarray(hdiag)

    # adapted/copied from pyscf.fci.direct_spin1.pspace
    # PySCF has a fancy indicing of Slater determinants (bitstrings to consecutive integers)
    addr = np.arange(hdiag.size)
    # again, separate the alpha and the bet aparts
    addra, addrb = divmod(addr, nb)
    # bit strings for the alpha and beta parts
    stra = np.array([cistring.addr2str(norb, neleca, ia) for ia in addra], dtype=np.uint64)
    strb = np.array([cistring.addr2str(norb, nelecb, ib) for ib in addrb], dtype=np.uint64)
    # number of slater determinants
    ci_matrix = np.zeros((num_sd, num_sd))
    # More Fucking Magic
    LIBFCI.FCIpspace_h0tril(ci_matrix.ctypes.data_as(ctypes.c_void_p),
                            h1e.ctypes.data_as(ctypes.c_void_p),
                            eri.ctypes.data_as(ctypes.c_void_p),
                            stra.ctypes.data_as(ctypes.c_void_p),
                            strb.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(num_sd))

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
        pspace.append(slater.combine_spin(alpha_bit, beta_bit, norb))

    return ci_matrix, pspace
