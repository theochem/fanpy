""" Functions used to obtain the Hamiltonian CI matrix

Functions
---------
get_one_int_value(one_int_matrices, i, k, orbtype)
    Gets value of the one-electron hamiltonian integral with orbitals `i` and `k`
get_two_int_value(two_int_matrices, i, j, k, l, orbtype)
    Gets value of the two-electron hamiltonian integral with orbitals `i`, `j`, `k` and `l`
ci_matrix(one_int, two_int, civec, dtype, orbtype)
    Returns CI Hamiltonian matrix in the arbitrary Slater (orthonormal) determinant basis
"""
import numpy as np
from ..backend import slater

__all__ = ['ci_matrix']


# FIXME: This should move to somewhere in wfns b/c wfns.proj.proj_hamiltonian also uses it
def get_one_int_value(one_int_matrices, i, k, orbtype):
    """ Gets value of the one-electron hamiltonian integral with orbitals `i` and `k`

    ..math::
        \big< \phi_i \big | \hat{h} | \phi_k \big>

    Parameters
    ----------
    one_int_matrices : tuple of np.ndarray(K, K)
        One electron integral matrices
        If 1-tuple, then restricted or generalized orbitals
        If 2-tuple, then unrestricted orbitals
    i : int
        Index of the spin orbital
    k : int
        Index of the spin orbital
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital

    Returns
    -------
    h_ik : float
        Value of the one electron hamiltonian

    Raises
    ------
    ValueError
        If indices are less than zero
        If indices are greater than the number of spin orbitals
    TypeError
        If orbtype is not one of 'restricted', 'unrestricted', 'generalized'
    """
    # NOTE: NECESARY?
    if any(index < 0 for index in [i, k]):
        raise ValueError('Indices cannot be negative')
    elif any(index >= one_int_matrices[0].shape[0] * (2 - int(orbtype == 'generalized'))
             for index in [i, k]):
        raise ValueError('Indices cannot be greater than the number of spin orbitals')

    if orbtype == 'restricted':
        # Assume that one_int_matrices are expressed wrt spatial orbitals
        nspatial = one_int_matrices[0].shape[0]
        I = slater.spatial_index(i, nspatial)
        K = slater.spatial_index(k, nspatial)
        # if spins are the same
        if slater.is_alpha(i, nspatial) == slater.is_alpha(k, nspatial):
            return one_int_matrices[0][I, K]
    elif orbtype == 'unrestricted':
        # Assume that one_int_matrices is expressed wrt spatial orbitals
        nspatial = one_int_matrices[0].shape[0]
        I = slater.spatial_index(i, nspatial)
        K = slater.spatial_index(k, nspatial)
        # if spins are both alpha
        if slater.is_alpha(i, nspatial) and slater.is_alpha(k, nspatial):
            return one_int_matrices[0][I, K]
        # if spins are both beta
        elif not slater.is_alpha(i, nspatial) and not slater.is_alpha(k, nspatial):
            return one_int_matrices[1][I, K]
    elif orbtype == 'generalized':
        return one_int_matrices[0][i, k]
    else:
        raise TypeError, 'Unknown orbital type, {0}'.format(orbtype)
    return 0.0


# FIXME: This should move to somewhere in wfns b/c wfns.proj.proj_hamiltonian also uses it
def get_two_int_value(two_int_matrices, i, j, k, l, orbtype):
    """ Gets value of the two-electron hamiltonian integral with orbitals `i`, `j`, `k`, and `l`

    ..math::
        \big< \theta \big | \hat{g} a_i a_j a^\dagger_k a^\dagger_l | \big> =
        \big< \phi_i \phi_j \big | \hat{g} | \phi_k \phi_l \big>

    Parameters
    ----------
    two_int_matrices : tuple of np.ndarray(K, K, K, K)
        Two electron integral matrices
        If 1-tuple, then restricted or generalized orbitals
        If 3-tuple, then unrestricted orbitals
    i : int
        Index of the spin orbital
    j : int
        Index of the spin orbital
    k : int
        Index of the spin orbital
    l : int
        Index of the spin orbital
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital

    Returns
    -------
    g_ijkl : float
        Value of the one electron hamiltonian

    Raises
    ------
    ValueError
        If indices are less than zero
        If indices are greater than the number of spin orbitals
    TypeError
        If orbtype is not one of 'restricted', 'unrestricted', 'generalized'
    """
    # NOTE: NECESARY?
    if any(index < 0 for index in [i, j, k, l]):
        raise ValueError('Indices cannot be negative')
    elif any(index >= two_int_matrices[0].shape[0] * (2 - int(orbtype == 'generalized'))
             for index in [i, j, k, l]):
        raise ValueError('Indices cannot be greater than the number of spin orbitals')

    if orbtype == 'restricted':
        # Assume that two_int_matrices is expressed wrt spatial orbitals
        nspatial = two_int_matrices[0].shape[0]
        # if i and k have same spin and j and l have same spin
        if (slater.is_alpha(i, nspatial) == slater.is_alpha(k, nspatial)
            and slater.is_alpha(j, nspatial) == slater.is_alpha(l, nspatial)):
            I = slater.spatial_index(i, nspatial)
            J = slater.spatial_index(j, nspatial)
            K = slater.spatial_index(k, nspatial)
            L = slater.spatial_index(l, nspatial)
            return two_int_matrices[0][I, J, K, L]
    elif orbtype == 'unrestricted':
        # Assume that one_int_matrices is expressed wrt spin orbitals
        nspatial = two_int_matrices[0].shape[0]
        # number of spatial orbitals is number of spin orbitals divided by 2
        I = slater.spatial_index(i, nspatial)
        J = slater.spatial_index(j, nspatial)
        K = slater.spatial_index(k, nspatial)
        L = slater.spatial_index(l, nspatial)
        # if alpha alpha alpha alpha
        if (slater.is_alpha(i, nspatial) and slater.is_alpha(j, nspatial) and
                slater.is_alpha(k, nspatial) and slater.is_alpha(l, nspatial)):
            return two_int_matrices[0][I, J, K, L]
        # if alpha beta alpha beta
        elif (slater.is_alpha(i, nspatial) and not slater.is_alpha(j, nspatial) and
              slater.is_alpha(k, nspatial) and not slater.is_alpha(l, nspatial)):
            return two_int_matrices[1][I, J, K, L]
        # if beta alpha beta alpha
        elif (not slater.is_alpha(i, nspatial) and slater.is_alpha(j, nspatial) and
              not slater.is_alpha(k, nspatial) and slater.is_alpha(l, nspatial)):
            # take the appropraite transposition to get the beta alpha beta alpha form
            return two_int_matrices[1][J, I, L, K]
        # if beta beta beta beta
        elif (not slater.is_alpha(i, nspatial) and not slater.is_alpha(j, nspatial) and
              not slater.is_alpha(k, nspatial) and not slater.is_alpha(l, nspatial)):
            return two_int_matrices[2][I, J, K, L]
    elif orbtype == 'generalized':
        return two_int_matrices[0][i, j, k, l]
    else:
        raise TypeError, 'Unknown orbital type, {0}'.format(orbtype)
    return 0.0


def ci_matrix(one_int, two_int, civec, dtype, orbtype):
    """ Returns Hamiltonian matrix in the arbitrary Slater (orthogonal) determinant basis

    ..math::
        one_int_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

    Parameters
    ----------
    one_int : 1- or 2-tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        1-tuple for spatial (restricted) and generalized orbitals
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
    two_int : 1- or 3-tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        In physicist's notation
        1-tuple for spatial (restricted) and generalized orbitals
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components)
    civec : iterable of {int, gmpy2.mpz}
        List of integers that describes the occupation of a Slater determinant as a bitstring
    dtype : {np.float64, np.complex128}
        Numpy data type
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital

    Returns
    -------
    matrix : np.ndarray(K, K)
    """
    #FIXME: need to allow one_int and two_int that are numpy arrays
    nci = len(civec)
    matrix = np.zeros((nci, ) * 2, dtype=dtype)

    # Loop only over upper triangular
    for nsd0, sd0 in enumerate(civec):
        for nsd1, sd1 in enumerate(civec[nsd0:]):
            nsd1 += nsd0
            diff_sd0, diff_sd1 = slater.diff(sd0, sd1)
            shared_indices = slater.occ_indices(slater.shared(sd0, sd1))

            # moving all the shared orbitals toward one another (in the middle)
            num_transpositions_0 = sum(len([j for j in shared_indices if j < i]) for i in diff_sd0)
            num_transpositions_1 = sum(len([j for j in shared_indices if j < i]) for i in diff_sd1)
            num_transpositions = num_transpositions_0 + num_transpositions_1
            sign = (-1)**num_transpositions

            if len(diff_sd0) != len(diff_sd1):
                continue
            else:
                diff_order = len(diff_sd0)

            # two sd's are the same
            if diff_order == 0:
                for count, i in enumerate(shared_indices):
                    matrix[nsd0, nsd1] += get_one_int_value(one_int, i, i,
                                                            orbtype=orbtype) * sign
                    for j in shared_indices[count + 1:]:
                        matrix[nsd0, nsd1] += get_two_int_value(two_int, i, j, i, j,
                                                                orbtype=orbtype) * sign
                        matrix[nsd0, nsd1] -= get_two_int_value(two_int, i, j, j, i,
                                                                orbtype=orbtype) * sign

            # two sd's are different by single excitation
            elif diff_order == 1:
                i, = diff_sd0
                a, = diff_sd1
                matrix[nsd0, nsd1] += get_one_int_value(one_int, i, a, orbtype=orbtype) * sign
                for j in shared_indices:
                    matrix[nsd0, nsd1] += get_two_int_value(two_int, i, j, a, j,
                                                            orbtype=orbtype) * sign
                    matrix[nsd0, nsd1] -= get_two_int_value(two_int, i, j, j, a,
                                                            orbtype=orbtype) * sign

            # two sd's are different by double excitation
            elif diff_order == 2:
                i, j = diff_sd0
                a, b = diff_sd1
                matrix[nsd0, nsd1] += get_two_int_value(two_int, i, j, a, b,
                                                        orbtype=orbtype) * sign
                matrix[nsd0, nsd1] -= get_two_int_value(two_int, i, j, b, a,
                                                        orbtype=orbtype) * sign

    # Make it Hermitian (symmetric)
    matrix[:, :] = np.triu(matrix) + np.triu(matrix, 1).T

    return matrix
