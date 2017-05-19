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
