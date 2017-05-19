""" Computes Hamiltonian of wavefunctions projected against a Slater determinant

..math::
    \braket{\Phi | H | \Psi}

Functions
---------
hamiltonian(wfn, slater_d, orbtype, deriv=None)
    Computes expectation value of the wavefunction with Hamiltonian projected against a Slater
    determinant
sen0_hamiltonian(wfn, slater_d, orbtype, deriv=None)
    Computes expectation value of the seniority zero wavefunction with Hamiltonian projected against
    a Slater determinant
"""
from itertools import combinations
from ..backend import slater
from .ci_matrix import get_one_int_value, get_two_int_value

__all__ = []

def sen0_hamiltonian(wfn, slater_d, orbtype, deriv=None):
    """ Compute the seniority zero Hamiltonian of the wavefunction projected against `slater_d`.

    Seniority zero means that there are no unpaired electrons

    Parameters
    ----------
    wfn : Wavefunction Instance
        Instance of Wavefunction class
        Needs to have the following in __dict__:
            nspin, nspatial, one_int, two_int, overlap
    slater_d : int
        The Slater Determinant against which to project.
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital
    deriv : int, None
        Index of the parameter with which to derivatize against
        Defeault is no derivatization

    Returns
    -------
    one_electron : float
        One electron energy
    coulomb : float
        Coulomb energy
    exchange : float
        Exchange energy

    Raises
    ------
    ValueError
        If Slater determinant does not have seniority zero
    TypeError
        If orbital type is generalized
    """
    if slater.get_seniority(slater_d, wfn.nspatial) != 0:
        raise ValueError('Given Slater determinant, {0}, does not have seniority zero'
                         ''.format(bin(slater_d)))
    if orbtype == 'generalized':
        raise TypeError('Generalized orbitals are not supported with seniority zero Hamiltonian')

    spatial_sd = slater.split_spin(slater_d, wfn.nspatial)[0]
    occ_spatial_indices = slater.occ_indices(spatial_sd)
    vir_spatial_indices = slater.vir_indices(spatial_sd, wfn.nspatial)

    one_electron = 0.0
    coulomb = 0.0
    exchange = 0.0

    # sum over zeroth order excitation
    coeff = wfn.get_overlap(slater_d, deriv=deriv)
    for counter, i in enumerate(occ_spatial_indices):
        one_electron += 2*coeff*get_one_int_value(wfn.one_int, i, i, orbtype=orbtype)
        coulomb += coeff*get_two_int_value(wfn.two_int, i, i, i, i, orbtype=orbtype)
        for j in occ_spatial_indices[counter+1:]:
            coulomb += 4*coeff*get_two_int_value(wfn.two_int, i, j, i, j, orbtype=orbtype)
            exchange -= 2*coeff*get_two_int_value(wfn.two_int, i, j, j, i, orbtype=orbtype)

    # sum over pair wise excitation (seniority zero)
    for i in occ_spatial_indices:
        for a in vir_spatial_indices:
            j = i + wfn.nspatial
            b = a + wfn.nspatial
            coeff = wfn.get_overlap(slater.excite(slater_d, i, j, a, b), deriv=deriv)
            coulomb += coeff*get_two_int_value(wfn.two_int, i, j, a, b, orbtype=orbtype)
            exchange -= coeff*get_two_int_value(wfn.two_int, i, j, b, a, orbtype=orbtype)

    return one_electron, coulomb, exchange
