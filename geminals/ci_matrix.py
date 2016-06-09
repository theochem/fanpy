import numpy as np
from . import slater


def is_alpha(index, nspatial):
    """ Checks if index belongs to an alpha spin orbital

    Parameter
    ---------
    index : int
        Index of the spin orbital in the Slater determinant
    nspatial : int
        Number of spatial orbitals

    Returns
    -------
    True if alpha orbital
    False if beta orbital

    Note
    ----
    Erratic behaviour of nspatial <= 0
    Erratic behaviour of i < 0 or i > 2*nspatial
    """
    # assert nspatial > 0, 'Number of spatial orbitals must be greater than 0'
    # assert 0 <= i < 2*nspatial, 'Index must be between 0 and 2*nspatial-1'
    return bool(index < nspatial)


def spatial_index(index, nspatial):
    """ Returns index of the spatial orbital corresponding to the specified spin orbital.

    Parameter
    ---------
    index : int
        Index of spin orbital in the Slater determinant
    nspatial : int
        Number of spatial orbitals

    Returns
    -------
    ind_spatial : int
        Index of spatial orbital

    Note
    ----
    Erratic behaviour of nspatial <= 0
    Erratic behaviour of i < 0 or i > 2*nspatial
    """
    # assert nspatial > 0, 'Number of spatial orbitals must be greater than 0'
    # assert 0 <= i < 2*nspatial, 'Index must be between 0 and 2*nspatial-1'
    if is_alpha(index, nspatial):
        return index
    else:
        return index - nspatial


def get_H_element(H_matrices, i, k, orb_type):
    """ Returns value of the one-electron hamiltonian matrix for spin orbitals `i` and `k`.

    ..math::
        \big< \phi_i \big | \hat{h} | \phi_k \big>

    Parameters
    ----------
    H_matrices : tuple of np.ndarray(K, K)
        Tuple of one-electron hamiltonian matrices
        More than one if orbital is unrestricted
    i : int
        Index of spin orbital
    k : int
        Index of spin orbital
    orb_type : str, options={'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of orbital

    Returns
    -------
    h_ik : float
        Value the one-electron hamiltonian matrix

    Note
    ----
    Erratic behaviour if i or k (or their spatial index analog) is negative or
    is larger than H_matrices dimensions
    """
    # In restricted & unrestricted cases, it is assumed that H_matrices are expressed wrt
    # spatial orbitals, and in generalized case wrt spin orbitals.

    if orb_type == 'restricted':
        # one-electron hamiltonain matrix is the same for alpha & beta spin orbitals
        nspatial = H_matrices[0].shape[0]   # number of spatial orbitals
        if is_alpha(i, nspatial) == is_alpha(k, nspatial):
            # orbital i & k have same spins
            I = spatial_index(i, nspatial)
            K = spatial_index(k, nspatial)
            return H_matrices[0][I, K]

    elif orb_type == 'unrestricted':
        # one-electron hamiltonain matrix is different for alpha & beta spin orbitals
        nspatial = H_matrices[0].shape[0]   # number of spatial orbitals
        I = spatial_index(i, nspatial)
        K = spatial_index(k, nspatial)
        if is_alpha(i, nspatial) and is_alpha(k, nspatial):
            # orbital i & k are both alpha
            return H_matrices[0][I, K]
        elif not is_alpha(i, nspatial) and not is_alpha(k, nspatial):
            # orbital i & k are both beta
            return H_matrices[1][I, K]

    elif orb_type == 'generalized':
        # one-electron hamiltonain matrix is expressed wrt spatial orbitals
        return H_matrices[0][i, k]

    else:
        raise AssertionError('Unknown orbital type, {0}'.format(orb_type))

    # orbital i & k have different spins (restricted & unrestricted cases)
    return 0.0


def get_G_element(G_matrices, i, j, k, l, orb_type):
    """ Returns value of two-electron hamiltonian matrix for spin orbitals `i`, `j`, `k`, and `l`.

    # TODO: check ordering
    ..math::
        \big< \theta \big | \hat{g} a_i a_j a^\dagger_k a^\dagger_l | \big> =
        \big< \phi_i \phi_j \big | \hat{g} | \phi_k \phi_l \big>

    Parameters
    ----------
    G_matrices : tuple of np.ndarray(K, K, K, K)
        Tuple of two-electron hamiltonian matrices
        More than one if orbital is unrestricted
    i : int
        Index of the spin orbital
    j : int
        Index of the spin orbital
    k : int
        Index of the spin orbital
    l : int
        Index of the spin orbital
    orb_type : str, options={'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital

    Returns
    -------
    g_ijkl : float
        Value of the two-electron hamiltonian matrix
    """
    # In restricted & unrestricted cases, it is assumed that G_matrices are expressed wrt
    # spatial orbitals, and in generalized case wrt spin orbitals.

    if orb_type == 'restricted':
        nspatial = G_matrices[0].shape[0]   # number of spatial orbitals
        if is_alpha(i, nspatial) == is_alpha(k, nspatial) and is_alpha(j, nspatial) == is_alpha(l, nspatial):
            # orbitals i & k have the same spin, and orbitals j & l have the same spin
            I = spatial_index(i, nspatial)
            J = spatial_index(j, nspatial)
            K = spatial_index(k, nspatial)
            L = spatial_index(l, nspatial)
            return G_matrices[0][I, J, K, L]

    elif orb_type == 'unrestricted':
        nspatial = G_matrices[0].shape[0]   # number of spatial orbitals
        I = spatial_index(i, nspatial)
        J = spatial_index(j, nspatial)
        K = spatial_index(k, nspatial)
        L = spatial_index(l, nspatial)
        if is_alpha(i, nspatial) and is_alpha(j, nspatial) and is_alpha(k, nspatial) and is_alpha(l, nspatial):
            # orbitals i, j, k & l are all alpha
            return G_matrices[0][I, J, K, L]
        elif is_alpha(i, nspatial) and not is_alpha(j, nspatial) and is_alpha(k, nspatial) and not is_alpha(l, nspatial):
            # orbitals i, j, k & l are alpha, beta, alpha & beta
            return G_matrices[1][I, J, K, L]
        elif not is_alpha(i, nspatial) and is_alpha(j, nspatial) and not is_alpha(k, nspatial) and is_alpha(l, nspatial):
            # orbitals i, j, k & l are beta, alpha, beta & alpha
            # FIXME: check the transposition
            return np.einsum('IJKL->JILK', G_matrices[1])[I, J, K, L]
        elif not is_alpha(i, nspatial) and not is_alpha(j, nspatial) and not is_alpha(k, nspatial) and not is_alpha(l, nspatial):
            # orbitals i, j, k & l are all beta
            return G_matrices[2][I, J, K, L]

    elif orb_type == 'generalized':
        # two-electron hamiltonain matrix is expressed wrt spatial orbitals
        return G_matrices[0][i, j, k, l]

    else:
        raise AssertionError('Unknown orbital type, {0}'.format(orb_type))

    return 0.0


def make_ci_matrix(H_matrices, G_matrices, civec, dtype, orb_type):
    """ Returns Hamiltonian matrix in the arbitrary Slater (orthogonal) determinant basis

    ..math::
        H_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

    Parameters
    ----------
    self : Wavefunction Instance
        Instance of Wavefunction class
        Needs to have the following in __dict__:
            nci, H, G, civec, dtype
    orb_type : {'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital

    Returns
    -------
    matrix : np.ndarray(K, K)
    """
    # Hamiltonian matrix
    ci_matrix = np.zeros((len(civec),) * 2, dtype=dtype)

    # Loop only over upper triangular elements
    for index0, sd0 in enumerate(civec):
        for index1, sd1 in enumerate(civec[index0:]):
            index1 += index0
            diff_sd0, diff_sd1 = slater.diff_indices(sd0, sd1)
            shared_indices = slater.shared_indices(sd0, sd1)

            if len(diff_sd0) != len(diff_sd1):
                continue
            else:
                diff_order = len(diff_sd0)

            if diff_order == 2:
                # Slater determinants are different by double excitation
                i, j = diff_sd0
                k, l = diff_sd1
                ci_matrix[index0, index1] += get_G_element(G_matrices, i, j, k, l, orb_type=orb_type)
                ci_matrix[index0, index1] -= get_G_element(G_matrices, i, j, l, k, orb_type=orb_type)

            elif diff_order == 1:
                # Slater determinants are different by single excitation
                i, = diff_sd0
                k, = diff_sd1
                ci_matrix[index0, index1] += get_H_element(H_matrices, i, k, orb_type=orb_type)
                for j in shared_indices:
                    if j != i:
                        ci_matrix[index0, index1] += get_G_element(G_matrices, i, j, k, j, orb_type=orb_type)
                        ci_matrix[index0, index1] -= get_G_element(G_matrices, i, j, j, k, orb_type=orb_type)

            elif diff_order == 0:
                # Slater determinants are the same
                for ic, i in enumerate(shared_indices):
                    ci_matrix[index0, index1] += get_H_element(H_matrices, i, i, orb_type=orb_type)
                    for j in shared_indices[ic + 1:]:
                        ci_matrix[index0, index1] += get_G_element(G_matrices, i, j, i, j, orb_type=orb_type)
                        ci_matrix[index0, index1] -= get_G_element(G_matrices, i, j, j, i, orb_type=orb_type)

    # Make it Hermitian
    ci_matrix[:, :] = np.triu(ci_matrix) + np.triu(ci_matrix, 1).T

    return ci_matrix


def make_doci_matrix(H_matrices, G_matrices, civec, dtype, nspatial, orb_type):
    """ Returns Hamiltonian matrix in the doci Slater determinant basis

    ..math::
        H_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

    Parameters
    ----------
    self : Wavefunction Instance
        Instance of Wavefunction class
        Needs to have the following in __dict__:
            nci, nspatial, H, G, civec, dtype
    orb_type : {'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital

    Returns
    -------
    matrix : np.ndarray(K, K)
    """
    # Hamiltonian matrix
    doci_matrix = np.zeros((len(civec),) * 2, dtype=dtype)

    # Loop only over upper triangular
    for index0, sd0 in enumerate(civec):
        for index1, sd1 in enumerate(civec[index0:]):
            index1 += index0
            diff_sd0, diff_sd1 = slater.diff_indices(sd0, sd1)
            shared_indices = slater.shared_indices(sd0, sd1)

            if len(diff_sd0) != len(diff_sd1):
                continue
            else:
                diff_order = len(diff_sd0)

            assert diff_order % 2 == 0, 'One (or both) of the Slater determinants, {0} or {1},'
            'are not DOCI Slater determinants'.format(bin(sd0), bin(sd1))

            # Slater determinants are different by double excitation
            if diff_order == 2:
                i, j = diff_sd0
                k, l = diff_sd1
                if spatial_index(i, nspatial) == spatial_index(k, nspatial) and spatial_index(j, nspatial) == spatial_index(l, nspatial):
                    doci_matrix[index0, index1] += get_G_element(G_matrices, i, j, k, l, orb_type=orb_type)
                elif spatial_index(i, nspatial) == spatial_index(l, nspatial) and spatial_index(j, nspatial) == spatial_index(k, nspatial):
                    doci_matrix[index0, index1] -= get_G_element(G_matrices, i, j, l, k, orb_type=orb_type)
                else:
                    assert True, 'One (or both) of the Slater determinants, {0} or {1},'
                    ' are not DOCI Slater determinants'.format(bin(sd0), bin(sd1))

            # Slater determinants are the same
            elif diff_order == 0:
                for ic, i in enumerate(shared_indices):
                    doci_matrix[index0, index1] += get_H_element(H_matrices, i, i, orb_type=orb_type)
                    for j in shared_indices[ic + 1:]:
                        doci_matrix[index0, index1] += get_G_element(G_matrices, i, j, i, j, orb_type=orb_type)
                        doci_matrix[index0, index1] -= get_G_element(G_matrices, i, j, j, i, orb_type=orb_type)

    # Make it Hermitian
    doci_matrix[:, :] = np.triu(doci_matrix) + np.tril(doci_matrix, -1)

    return doci_matrix
