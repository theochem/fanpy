"""Eigenvalue/vector solver for CI wavefunctions."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.linalg
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian

__all__ = []


# TODO: implement Davidson method
# FIXME: incredibly slow implementation
# FIXME: change return value
def eigen_solve(wfn, ham, exc_lvl=0):
    """Optimize the CI wavefunction with the given Hamiltonian using eigenvalue decomposition.

    Generates the CI Hamiltonian matrix and diagonalizes it using eigenvalue decomposition.

    Parameters
    ----------
    ci_wfn : CIWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited
        state)
    ham : ChemicalHamiltonian
        Hamiltonian that defines the system under study
    exc_lvl : int
        Level of excitation of the CI wavefunction

    Returns
    -------
    Energy of the wavefunction

    Raises
    ------
    TypeError
        If wavefunction is not an instance (or instance of a child) of CIWavefunction
        If Hamiltonian is not an instance (or instance of a child) of ChemicalHamiltonian
        If exc_lvl is not a positive integer
    ValueError
        If wavefunction and Hamiltonian do not have the same data type
        If wavefunction and Hamiltonian do not have the same number of spin orbitals
    """
    # Preprocess variables
    if not isinstance(wfn, CIWavefunction):
        raise TypeError('Given wavefunction is not an instance of BaseWavefunction (or its '
                        'child).')
    if not isinstance(ham, ChemicalHamiltonian):
        raise TypeError('Given Hamiltonian is not an instance of BaseWavefunction (or its '
                        'child).')
    if wfn.dtype != ham.dtype:
        raise ValueError('Wavefunction and Hamiltonian do not have the same data type.')
    if wfn.nspin != ham.nspin:
        raise ValueError('Wavefunction and Hamiltonian do not have the same number of '
                         'spin orbitals')
    if not (isinstance(exc_lvl, int) and exc_lvl >= 0):
        raise TypeError('Given exc_lvl is not a positive integer')

    ci_matrix = np.zeros((wfn.nsd, wfn.nsd), dtype=wfn.dtype)
    for i, sd1 in enumerate(wfn.sd_vec):
        for j, sd2 in enumerate(wfn.sd_vec[i:]):
            ci_matrix[i, i+j] += sum(ham.integrate_sd_sd(sd1, sd2))
    ci_matrix += ci_matrix.T - np.diag(ci_matrix.diagonal())

    eigval, eigvec = scipy.linalg.eigh(ci_matrix, lower=True, overwrite_a=True, turbo=False, type=1)
    del ci_matrix

    wfn.params[:] = eigvec[:, exc_lvl].flat
    energy = eigval[exc_lvl]

    return energy
