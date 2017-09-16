"""Script for generating one and two electron integrals using PySCF.

Functions
---------
hartreefock(xyz_file, basis, is_unrestricted=False)
    Runs HF in PySCF.

"""
from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.lib import load_library

__all__ = []

LIBFCI = load_library('libfci')


def hartreefock(xyz_file, basis, is_unrestricted=False):
    """Run HF using PySCF.

    Parameters
    ----------
    xyz_file : str
        XYZ file location.
    basis : str
        Basis set available in PySCF.
    is_unrestricted : bool
        Flag to run unrestricted HF.
        Default is restricted HF.

    Returns
    -------
    result : dict
        "el_energy"
            The electronic energy.
        "nuc_nuc_energy"
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
    if os.path.isfile(xyz_file):
        pass
    elif os.path.isfile(os.path.join(cwd, xyz_file)):
        xyz_file = os.path.join(cwd, xyz_file)
    else:
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
    E_tot = hf.kernel()  # HF is solved here
    E_elec = E_tot - E_nuc
    # mo_coeffs
    mo_coeff = hf.mo_coeff
    # Get integrals (See pyscf.gto.moleintor.getints_by_shell for other types of integrals)
    # get 1e integral
    one_int_ab = mol.intor('cint1e_nuc_sph') + mol.intor('cint1e_kin_sph')
    one_int = mo_coeff.T.dot(one_int_ab).dot(mo_coeff)
    # get 2e integral
    eri = ao2mo.full(mol, mo_coeff, verbose=0, intor='cint2e_sph')
    two_int = ao2mo.restore(1, eri, mol.nao_nr())
    # NOTE: PySCF uses Chemist's notation
    two_int = np.einsum('ijkl->ikjl', two_int)
    # results
    result = {'el_energy': E_elec,
              'nuc_nuc_energy': E_nuc,
              'one_int': (one_int,),
              'two_int': (two_int,)}
    return result


if __name__ == '__main__':
    # extract keyword from command line
    kwargs = {key: val for key, val in zip(sys.argv[4::2], sys.argv[5::2])}
    # change data types
    if 'is_unrestricted' in kwargs:
        kwargs['is_unrestricted'] = bool(kwargs['is_unrestricted'])

    data = hartreefock(**kwargs)
    np.save(sys.argv[1], np.array([data['el_energy'], data['nuc_nuc_energy']]))
    np.save(sys.argv[2], data['one_int'][0])
    np.save(sys.argv[3], data['two_int'][0])
