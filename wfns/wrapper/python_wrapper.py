"""Wraps appropriate Python versions around current Python version (3.6+).

Because HORTON and PySCF do not support many versions of Python, some sort of work around is needed.
Since only certain arrays are needed from HORTON and PySCF, these arrays are extract and saved to
disk as a npy file.
Then, user can load these files to have access to these numbers.

Note
----
This is only a temporary solution. We can make things back-compatible to certain versions of HORTON
and PySCF, or we can wait until these modules catch up to the latest Python versions. For now, this
module can act as a temporary hack to access these modules.
"""
import os
from subprocess import call
import numpy as np


dirname = os.path.dirname(os.path.abspath(__file__))


def generate_hartreefock_results(python_name, calctype, energies_name='energies.npy',
                                 oneint_name='oneint.npy', twoint_name='twoint.npy',
                                 remove_npyfiles=False, **kwargs):
    """Generate results of Hartree Fock calculation.

    Results include: electronic energy, nuclear-nuclear repulsion energy, one electron integrals,
    and two electron integrals.

    Parameters
    ----------
    python_name : str
        Name of the python to be used in the shell
    calctype : {'horton_hartreefock.py', 'horton_gaussian_fchk.py', 'pyscf_hartreefock.py'}
        Name of the python script to be used
    energies_name : {str, 'energies.npy}
        Name of the file to be generated that contains the electronic and nuclear-nuclear repulsion
        energy
        First entry is the electronic energy
        Second entry is the nuclear-nuclear repulsion energy
    oneint_name : {str, 'oneint.npy}
        Name of the file to be generated that contains the one electron integrals
        If two dimensional matrix, then the orbitals are restricted
        If three dimensional matrix, then the orbitals are unrestricted
    twoint_name : {str, 'twoint.npy}
        Name of the file to be generated that contains the two electron integrals
        If four dimensional matrix, then the orbitals are restricted
        If five dimensional matrix, then the orbitals are unrestricted
    remove_npyfiles : bool
        Option to remove generated numpy files
        True will remove numpy files
    kwargs
        Keyword arguments for the script

    Returns
    -------
    el_energy : float
        Electronic energy
    nuc_nuc_energy : float
        Nuclear-nuclear repulsion energy
    oneint : np.ndarray, tuple of np.ndarray
        One electron integrals
        If numpy array, then orbitals are restricted
        If tuple of numpy arrays, then orbitals are unrestricted
    twoint : np.ndarray, tuple of np.ndarray
        Two electron integrals
        If numpy array, then orbitals are restricted
        If tuple of numpy arrays, then orbitals are unrestricted
    """
    # turn keywords to pair of key and value
    kwargs = [str(i) for item in kwargs.items() for i in item]
    # call script with appropriate python
    call([python_name, os.path.join(dirname, calctype), energies_name, oneint_name, twoint_name,
          *kwargs])
    el_energy, nuc_nuc_energy = np.load(energies_name)
    oneint = np.load(oneint_name)
    if oneint.ndim == 3:
        oneint = tuple(oneint)
    twoint = np.load(twoint_name)
    if twoint.ndim == 5:
        twoint = tuple(twoint)

    if remove_npyfiles:
        os.remove(energies_name)
        os.remove(oneint_name)
        os.remove(twoint_name)

    return el_energy, nuc_nuc_energy, oneint, twoint


def generate_fci_results(python_name, cimatrix_name='cimatrix.npy', sds_name='sds.npy',
                         remove_npyfiles=False, **kwargs):
    """Generate results of FCI calculation (from PySCF).

    Results include: ci matrix and pspace

    Parameters
    ----------
    python_name : str
        Name of the python to be used in the shell
    calctype : {}
        Name of the python script to be used
    cimatrix_name : {str, 'cimatrix.npy}
        Name of the file to be generated that contains the ci matrix coefficients
    sds_name : {str, 'sds.npy}
        Name of the file to be generated that contains the binary Slater determinant values
    remove_npyfiles : bool
        Option to remove generated numpy files
        True will remove numpy files
    kwargs
        Keyword arguments for the script
        'h1e' : str
            Name of the numpy file that contains the one electron integrals
        'eri' : str
            Name of the numpy file that contains the two electron integrals
        'nelec' : int
            Number of electrons

    Returns
    -------
    cimatrix : np.ndarray
        CI matrix
    sds : list of ints
        List of binary Slater determinant values
    """
    # convert integrals
    if isinstance(kwargs['h1e'], np.ndarray):
        np.save('temp_h1e.npy', kwargs['h1e'])
        kwargs['h1e'] = 'temp_h1e.npy'
    if isinstance(kwargs['eri'], np.ndarray):
        np.save('temp_eri.npy', kwargs['eri'])
        kwargs['eri'] = 'temp_eri.npy'
    # turn keywords to pair of key and value
    kwargs = [str(i) for item in kwargs.items() for i in item]
    # call script with appropriate python
    call([python_name, os.path.join(dirname, 'pyscf_generate_fci_matrix.py'), cimatrix_name,
          sds_name, *kwargs])

    cimatrix = np.load(cimatrix_name)
    sds = np.load(sds_name).tolist()

    if remove_npyfiles:
        os.remove(cimatrix_name)
        os.remove(sds_name)

    return cimatrix, sds
