""" Functions used to obtain the density matrix Functions
---------
add_one_density(matrices, i, j, val, orbtype)
    Adds value to the one electron density matrix appropriately
add_two_density(matrices, i, j, k, l, val, orbtype)
    Adds value to the two electron density matrix appropriately
density_matrix(sd_coeffs, civec, nspatial, is_chemist_notation=False, val_threshold=0,
               orbtype='restricted')
    Returns the one and two electron density matrices
"""
import numpy as np
from .. import slater

__all__ = ['density_matrix']


def add_one_density(matrices, spin_i, spin_j, val, orbtype):
    """ Adds some value to the appropriate density matrix element

    ..math::
        \braket{\Phi_1 | a_i a_j^\dagger | \Phi_2}

    Parameters
    ----------
    matrices : list of np.ndarray
        List of one electron density matrices
        If 1-list, then restricted or generalized orbitals
        If 2-list, then unrestricted orbitals (alpha-alpha and beta-beta components)
    spin_i : int
        Spin orbital index of the density matrix
    spin_j : int
        Spin orbital index of the density matrix
    val : float
        Value that will be added to the density matrix
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital

    Raises
    ------
    TypeError
        If matrices are not a list of numpy arrays
        If matrices are not two dimensional
        If matrices are not square
    ValueError
        If restricted or generalized orbitals and number of matrices is not one
        If unrestricted orbitals and number of matrices is not two
        If orbital type is not one of 'restricted', 'unrestricted', 'generalized'
    """
    if not (isinstance(matrices, list) and all(isinstance(i, np.ndarray) for i in matrices)):
        raise TypeError('Matrices must be given as a list of numpy arrays')

    if any(len(i.shape) != 2 for i in matrices):
        raise TypeError('All matrices must be two dimensional')
    if any(j != matrices[0].shape[0] for i in matrices for j in i.shape):
        raise TypeError('All matrices must be square')

    if orbtype in ['restricted', 'generalized'] and len(matrices) != 1:
        raise ValueError('Density matrix must be given as a list of one numpy array for'
                         ' restricted and generalized orbitals')
    elif orbtype in ['unrestricted'] and len(matrices) != 2:
        raise ValueError('Density matrix must be given as a list of two numpy arrays for'
                         ' unrestricted orbitals')

    if orbtype == 'restricted':
        nspatial = matrices[0].shape[0]
        spatial_i = slater.spatial_index(spin_i, nspatial)
        spatial_j = slater.spatial_index(spin_j, nspatial)
        if slater.is_alpha(spin_i, nspatial) == slater.is_alpha(spin_j, nspatial):
            # CHECK: is there a factor of 2 missing here?
            matrices[0][spatial_i, spatial_j] += val

    elif orbtype == 'unrestricted':
        nspatial = matrices[0].shape[0]
        spatial_i = slater.spatial_index(spin_i, nspatial)
        spatial_j = slater.spatial_index(spin_j, nspatial)
        # if spins are both alpha
        if slater.is_alpha(spin_i, nspatial) and slater.is_alpha(spin_j, nspatial):
            matrices[0][spatial_i, spatial_j] += val
        # if spins are both beta
        elif not slater.is_alpha(spin_i, nspatial) and not slater.is_alpha(spin_j, nspatial):
            matrices[1][spatial_i, spatial_j] += val

    elif orbtype == 'generalized':
        matrices[0][spin_i, spin_j] += val

    else:
        raise ValueError('Unsupported orbital type')


def add_two_density(matrices, spin_i, spin_j, spin_k, spin_l, val, orbtype):
    """ Adds some value to the appropriate one electron density matrix element

    ..math::
        \braket{\Phi_1 | a_i a_j a_k^\dagger a_l^\dagger | \Phi_2}

    Parameters
    ----------
    matrices : list of np.ndarray
        List of two electron density matrices
        If 1-list, then restricted or generalized orbitals
        If 3-list, then unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components)
    spin_i : int
        Spin orbital index of the density matrix
    spin_j : int
        Spin orbital index of the density matrix
    spin_k : int
        Spin orbital index of the density matrix
    spin_l : int
        Spin orbital index of the density matrix
    val : float
        Value that will be added to the density matrix
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital

    Raises
    ------
    TypeError
        If matrices are not a list of numpy arrays
    ValueError
        If restricted or generalized orbitals and number of matrices is not one
        If unrestricted orbitals and number of matrices is not two
        If orbital type is not one of 'restricted', 'unrestricted', 'generalized'

    Note
    ----
    Assumes that the spin orbital indices are given with the physicist's notation
    """
    if not (isinstance(matrices, list) and all(isinstance(i, np.ndarray) for i in matrices)):
        raise TypeError('Matrices must be given as a list of numpy arrays')

    if any(len(i.shape) != 4 for i in matrices):
        raise TypeError('All matrices must be four dimensional')
    if any(j != matrices[0].shape[0] for i in matrices for j in i.shape):
        raise TypeError('All matrices should have the same dimension along all of the axes')

    if orbtype in ['restricted', 'generalized'] and len(matrices) != 1:
        raise ValueError('Density matrix must be given as a list of one numpy array for'
                         ' restricted and generalized orbitals')
    elif orbtype in ['unrestricted'] and len(matrices) != 3:
        raise ValueError('Density matrix must be given as a list of three numpy arrays for'
                         ' unrestricted orbitals')

    if orbtype == 'restricted':
        nspatial = matrices[0].shape[0]
        spatial_i = slater.spatial_index(spin_i, nspatial)
        spatial_j = slater.spatial_index(spin_j, nspatial)
        spatial_k = slater.spatial_index(spin_k, nspatial)
        spatial_l = slater.spatial_index(spin_l, nspatial)
        # if i and k have same spin and j and l have same spin
        if (slater.is_alpha(spin_i, nspatial) == slater.is_alpha(spin_k, nspatial) and
                slater.is_alpha(spin_j, nspatial) == slater.is_alpha(spin_l, nspatial)):
            matrices[0][spatial_i, spatial_j, spatial_k, spatial_l] += val

    elif orbtype == 'unrestricted':
        nspatial = matrices[0].shape[0]
        spatial_i = slater.spatial_index(spin_i, nspatial)
        spatial_j = slater.spatial_index(spin_j, nspatial)
        spatial_k = slater.spatial_index(spin_k, nspatial)
        spatial_l = slater.spatial_index(spin_l, nspatial)
        # if all spins are alpha
        if (slater.is_alpha(spin_i, nspatial) and slater.is_alpha(spin_k, nspatial) and
                slater.is_alpha(spin_j, nspatial) and slater.is_alpha(spin_l, nspatial)):
            matrices[0][spatial_i, spatial_j, spatial_k, spatial_l] += val
        # if alpha beta alpha beta
        elif (slater.is_alpha(spin_i, nspatial) and slater.is_alpha(spin_k, nspatial) and
              not slater.is_alpha(spin_j, nspatial) and not slater.is_alpha(spin_l, nspatial)):
            matrices[1][spatial_i, spatial_j, spatial_k, spatial_l] += val
        # if all spins are beta
        elif (not slater.is_alpha(spin_i, nspatial) and not slater.is_alpha(spin_k, nspatial) and
              not slater.is_alpha(spin_j, nspatial) and not slater.is_alpha(spin_l, nspatial)):
            matrices[2][spatial_i, spatial_j, spatial_k, spatial_l] += val

    elif orbtype == 'generalized':
        matrices[0][spin_i, spin_j, spin_k, spin_l] += val

    else:
        raise ValueError('Unsupported orbital type')

# TODO: generalize to arbitrary order density matrix

def density_matrix(sd_coeffs, civec, nspatial, is_chemist_notation=False, val_threshold=0,
                   orbtype='restricted'):
    """ Returns the first and second order density matrices

    Second order density matrix uses the Physicist's notation:
    ..math::
        \Gamma_{ijkl} = < \Psi | a_i^\dagger a_k^\dagger a_l a_j | \Psi >
    Chemist's notation is also implemented
    ..math::
        \Gamma_{ijkl} = < \Psi | a_i^\dagger a_j^\dagger a_k a_l | \Psi >

    Paramaters
    ----------
    sd_coeffs : list of float
        Slater determinant coefficients
    civec : list of int/gmpy2.mpz
        Slater determinant
    nspatial : int
        Number of spatial orbitals
    is_chemist_notation : bool
        True if chemist's notation
        False if physicist's notation
        Default is Physicist's notation
    val_threshold : float
        Threshold for truncating the density matrice entries
        Skips all of the Slater determinants whose maximal sum of contributions to density matrices
        is less than threshold value
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital

    Returns
    -------
    one_densities : tuple of np.ndarray
        One electron density matrix
        For spatial and generalized orbitals, 1-tuple of np.ndarray
        For unretricted spin orbitals, 2-tuple of np.ndarray
    two_densities : tuple of np.ndarray
        Two electron density matrix
        For spatial and generalized orbitals, 1-tuple of np.ndarray
        For unrestricted orbitals, 3-tuple of np.ndarray

    Raises
    ------
    TypeError
        If the orbital type is not one of 'restricted', 'unrestricted', 'generalized'
    """
    # TODO: generalize to arbitrary order density matrix
    # sort coefficients and sd's by the magintude of coefficient (useful for truncating)
    sorted_x, sorted_sd = zip(*sorted(zip(sd_coeffs, civec), key=lambda x: abs(x[0]), reverse=True))
    num_sds = len(sorted_sd)

    # initiate output
    one_densities = []
    two_densities = []
    if orbtype == 'restricted':
        one_densities = [np.zeros((nspatial, )*2)]
        two_densities = [np.zeros((nspatial, )*4)]
    elif orbtype == 'unrestricted':
        one_densities = [np.zeros((nspatial, )*2) for i in range(2)]
        two_densities = [np.zeros((nspatial, )*4) for i in range(3)]
    elif orbtype == 'generalized':
        one_densities = [np.zeros((2*nspatial, )*2)]
        two_densities = [np.zeros((2*nspatial, )*4)]
    else:
        raise TypeError('Unsupported orbital type')

    for count1, sd1 in enumerate(sorted_sd):
        # truncation condition
        if (sorted_x[count1] * (num_sds - count1))**2 < val_threshold:
            break
        for count2, sd2 in enumerate(sorted_sd[count1:]):
            # increatement counter (because enumerate)
            count2 += count1
            # truncation condition
            if abs(sorted_x[count1] * sorted_x[count2]) * (num_sds - count1)**2 < val_threshold:
                break

            # orbitals that are not shared by the two determinants
            left_diff, right_diff = slater.diff(sd1, sd2)
            shared_indices = slater.occ_indices(slater.shared(sd1, sd2))

            # moving all the shared orbitals toward one another (in the middle)
            num_transpositions_0 = sum(len([j for j in shared_indices if j < i]) for i in
                                       left_diff)
            num_transpositions_1 = sum(len([j for j in shared_indices if j < i]) for i in
                                       right_diff)
            num_transpositions = num_transpositions_0 + num_transpositions_1
            sign = (-1)**num_transpositions

            # contributing value
            val = sorted_x[count1] * sorted_x[count2] * sign

            # check for number symmetery
            if len(right_diff) != len(left_diff):
                continue

            # if they're the same
            elif len(left_diff) == 0:
                for ind, i in enumerate(shared_indices):
                    add_one_density(one_densities, i, i, val, orbtype=orbtype)
                    for j in shared_indices[ind+1:]:
                        add_two_density(two_densities, i, j, i, j, val, orbtype=orbtype)
                        add_two_density(two_densities, j, i, j, i, val, orbtype=orbtype)
                        add_two_density(two_densities, i, j, j, i, -val, orbtype=orbtype)
                        add_two_density(two_densities, j, i, i, j, -val, orbtype=orbtype)
            # if single excitation
            elif len(left_diff) == 1:
                i, = left_diff
                k, = right_diff
                add_one_density(one_densities, i, k, val, orbtype=orbtype)
                add_one_density(one_densities, k, i, val, orbtype=orbtype)
                for j in shared_indices:
                    add_two_density(two_densities, i, j, k, j, val, orbtype=orbtype)
                    add_two_density(two_densities, j, i, j, k, val, orbtype=orbtype)
                    add_two_density(two_densities, k, j, i, j, val, orbtype=orbtype)
                    add_two_density(two_densities, j, k, j, i, val, orbtype=orbtype)
                    add_two_density(two_densities, i, j, j, k, -val, orbtype=orbtype)
                    add_two_density(two_densities, j, i, k, j, -val, orbtype=orbtype)
                    add_two_density(two_densities, j, k, i, j, -val, orbtype=orbtype)
                    add_two_density(two_densities, k, j, j, i, -val, orbtype=orbtype)
            # if double excitation
            elif len(left_diff) == 2:
                i, j = left_diff
                k, l = right_diff
                add_two_density(two_densities, i, j, k, l, val, orbtype=orbtype)
                add_two_density(two_densities, j, i, l, k, val, orbtype=orbtype)
                add_two_density(two_densities, k, l, i, j, val, orbtype=orbtype)
                add_two_density(two_densities, l, k, j, i, val, orbtype=orbtype)
                add_two_density(two_densities, i, j, l, k, -val, orbtype=orbtype)
                add_two_density(two_densities, j, i, k, l, -val, orbtype=orbtype)
                add_two_density(two_densities, l, k, i, j, -val, orbtype=orbtype)
                add_two_density(two_densities, k, l, j, i, -val, orbtype=orbtype)
    # change notation if necessary
    if is_chemist_notation:
        two_densities = [np.einsum('ijkl->iklj', i) for i in two_densities]
    return tuple(one_densities), tuple(two_densities)
