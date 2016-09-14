from __future__ import absolute_import, division, print_function
import os
from pyscf import gto, scf, ao2mo

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
        "energy", electronic energy
        "nuc_nuc", nuclear repulsion energy
        "H", tuple of the one-electron Hamiltonian;
        "G", tuple of the two-electron Hamiltonian;

    """
    # check xyz file
    cwd = os.path.dirname(__file__)
    data_dir = os.path.join(cwd, '../../data')
    if os.path.isfile(xyz_file):
        pass
    elif os.path.isfile(os.path.join(cwd, xyz_file)):
        xyz_file = os.path.join(cwd, xyz_file)
    elif os.path.isfile(os.path.join(data_dir, xyz_file)):
        xyz_file = os.path.join(data_dir, xyz_file)
    else:
        raise ValueError('Given xyz_file does not exist')
   # get coordinates
    with open(xyz_file, 'r') as f:
        lines = [i.strip() for i in f.readlines()[2:]]
        atoms = ';'.join(lines)
    # get mol
    mol = gto.M(atom=atoms, basis=basis)
    # get hf
    if is_unrestricted:
        raise NotImplementedError('Unrestricted or Generalized orbitals are not supported in this PySCF wrapper (yet).')
    else:
        hf = scf.RHF(mol)
    # run hf
    # hf.scf()
    # energies
    E_nuc = hf.energy_nuc()
    E_tot = hf.kernel() # HF is solved here
    E_elec = E_tot - E_nuc
    # mo_coeffs
    mo_coeff = hf.mo_coeff
    # get 1e integral
    H_ab = hf.get_hcore(mol)
    H = mo_coeff.T.dot(H_ab).dot(mo_coeff)
    # get 2e integral
    eri = ao2mo.full(mol, mo_coeff, verbose=0)
    G = ao2mo.restore(1, eri, mol.nao_nr())
    # results
    result = {'energy' : E_elec,
              'nuc_nuc' : E_nuc,
              'H' : (H,),
              'G' : (G,)}
    return result


