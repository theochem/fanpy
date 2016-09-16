import numpy as np
from .. import slater


def is_alpha(i, nspatial):
    """ Checks if index `i` belongs to an alpha spin orbital

    Parameter
    ---------
    i : int
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
    if i < nspatial:
        return True
    else:
        return False


def spatial_index(i, nspatial):
    """ Returns the index of the spatial orbital that corresponds to the
    spin orbital `i`

    Parameter
    ---------
    i : int
        Index of the spin orbital in the Slater determinant
    nspatial : int
        Number of spatial orbitals

    Returns
    -------
    ind_spatial : int
        Index of the spatial orbital that corresponds to the spin orbital `i`

    Note
    ----
    Erratic behaviour of nspatial <= 0
    Erratic behaviour of i < 0 or i > 2*nspatial
    """
    # assert nspatial > 0, 'Number of spatial orbitals must be greater than 0'
    # assert 0 <= i < 2*nspatial, 'Index must be between 0 and 2*nspatial-1'
    if is_alpha(i, nspatial):
        return i
    else:
        return i - nspatial


def get_H_value(H_matrices, i, k, orb_type):
    """ Gets value of the one-electron hamiltonian integral with orbitals `i` and `k`

    ..math::
        \big< \phi_i \big | \hat{h} | \phi_k \big>

    Parameters
    ----------
    H_matrices : tuple of np.ndarray(K, K)
        One electron integral matrices
        More than one if orbital is unrestricted
    i : int
        Index of the spin orbital
    k : int
        Index of the spin orbital
    orb_type : {'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital

    Returns
    -------
    h_ik : float
        Value of the one electron hamiltonian

    Note
    ----
    Erratic behaviour if i or k (or their spatial index analog) is negative or
    is larger than H_matrices
    """
    if orb_type == 'restricted':
        # Assume that H_matrices are expressed wrt spatial orbitals
        ns = H_matrices[0].shape[0]
        # if spins are the same
        if is_alpha(i, ns) == is_alpha(k, ns):
            I = spatial_index(i, ns)
            K = spatial_index(k, ns)
            return H_matrices[0][I, K]
    elif orb_type == 'unrestricted':
        # Assume that H_matrices is expressed wrt spin orbitals
        ns = H_matrices[0].shape[0]
        # number of spatial orbitals is number of spin orbitals divided by 2
        I = spatial_index(i, ns)
        K = spatial_index(k, ns)
        if is_alpha(i, ns) and is_alpha(k, ns):
            return H_matrices[0][I, K]
        elif not is_alpha(i, ns) and not is_alpha(k, ns):
            return H_matrices[1][I, K]
    elif orb_type == 'generalized':
        return H_matrices[0][i, k]
    else:
        raise AssertionError, 'Unknown orbital type, {0}'.format(orb_type)
    return 0.0


def get_G_value(G_matrices, i, j, k, l, orb_type):
    """ Gets value of the two-electron hamiltonian integral with orbitals `i`, `j`, `k`, and `l`

    # TODO: check ordering
    ..math::
        \big< \theta \big | \hat{g} a_i a_j a^\dagger_k a^\dagger_l | \big> =
        \big< \phi_i \phi_j \big | \hat{g} | \phi_k \phi_l \big>

    Parameters
    ----------
    G_matrices : tuple of np.ndarray(K, K, K, K)
        Two electron integral matrices
        More than one if orbital is unrestricted
    i : int
        Index of the spin orbital
    j : int
        Index of the spin orbital
    k : int
        Index of the spin orbital
    l : int
        Index of the spin orbital
    orb_type : {'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital

    Returns
    -------
    g_ijkl : float
        Value of the one electron hamiltonian
    """
    if orb_type == 'restricted':
        # Assume that G_matrices is expressed wrt spatial orbitals
        ns = G_matrices[0].shape[0]
        if is_alpha(i, ns) == is_alpha(k, ns) and is_alpha(j, ns) == is_alpha(l, ns):
            I = spatial_index(i, ns)
            J = spatial_index(j, ns)
            K = spatial_index(k, ns)
            L = spatial_index(l, ns)
            return G_matrices[0][I, J, K, L]
    elif orb_type == 'unrestricted':
        # Assume that H_matrices is expressed wrt spin orbitals
        ns = G_matrices[0].shape[0]
        # number of spatial orbitals is number of spin orbitals divided by 2
        I = spatial_index(i, ns)
        J = spatial_index(j, ns)
        K = spatial_index(k, ns)
        L = spatial_index(l, ns)
        if is_alpha(i, ns) and is_alpha(j, ns) and is_alpha(k, ns) and is_alpha(l, ns):
            return G_matrices[0][I, J, K, L]
        elif is_alpha(i, ns) and not is_alpha(j, ns) and is_alpha(k, ns) and not is_alpha(l, ns):
            return G_matrices[1][I, J, K, L]
        elif not is_alpha(i, ns) and is_alpha(j, ns) and not is_alpha(k, ns) and is_alpha(l, ns):
            # FIXME: check the transposition
            # take the appropraite transpose to get the beta alpha beta alpha form
            return np.einsum('IJKL->JILK', G_matrices[1])[I, J, K, L]
        elif not is_alpha(i, ns) and not is_alpha(j, ns) and not is_alpha(k, ns) and not is_alpha(l, ns):
            return G_matrices[2][I, J, K, L]
    elif orb_type == 'generalized':
        return G_matrices[0][i, j, k, l]
    return 0.0


def ci_matrix(self, orb_type):
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
    H = self.H
    G = self.G

    ci_matrix = np.zeros((self.nci,) * 2, dtype=self.dtype)

    # Loop only over upper triangular
    for nsd0, sd0 in enumerate(self.civec):
        for nsd1, sd1 in enumerate(self.civec[nsd0:]):
            nsd1 += nsd0
            diff_sd0, diff_sd1 = slater.diff(sd0, sd1)
            shared_indices = slater.occ_indices(slater.shared(sd0, sd1))

            # moving all the shared orbitals toward one another (in the middle)
            num_transpositions_0 = sum(len([j for j in shared_indices if j<i]) for i in diff_sd0)
            num_transpositions_1 = sum(len([j for j in shared_indices if j<i]) for i in diff_sd1)
            num_transpositions = num_transpositions_0 + num_transpositions_1
            sign = (-1)**num_transpositions

            if len(diff_sd0) != len(diff_sd1):
                continue
            else:
                diff_order = len(diff_sd0)

            # two sd's are the same
            if diff_order == 0:
                for ic, i in enumerate(shared_indices):
                    ci_matrix[nsd0, nsd1] += get_H_value(H, i, i, orb_type=orb_type) * sign
                    for j in shared_indices[ic + 1:]:
                        ci_matrix[nsd0, nsd1] += get_G_value(G, i, j, i, j, orb_type=orb_type) * sign
                        ci_matrix[nsd0, nsd1] -= get_G_value(G, i, j, j, i, orb_type=orb_type) * sign

            # two sd's are different by single excitation
            elif diff_order == 1:
                i, = diff_sd0
                a, = diff_sd1
                ci_matrix[nsd0, nsd1] += get_H_value(H, i, a, orb_type=orb_type) * sign
                for num_trans, j in enumerate(shared_indices):
                    ci_matrix[nsd0, nsd1] += get_G_value(G, i, j, a, j, orb_type=orb_type) * sign
                    ci_matrix[nsd0, nsd1] -= get_G_value(G, i, j, j, a, orb_type=orb_type) * sign

            # two sd's are different by double excitation
            elif diff_order == 2:
                i, j = diff_sd0
                a, b = diff_sd1
                ci_matrix[nsd0, nsd1] += get_G_value(G, i, j, a, b, orb_type=orb_type) * sign
                ci_matrix[nsd0, nsd1] -= get_G_value(G, i, j, b, a, orb_type=orb_type) * sign

    # Make it Hermitian
    ci_matrix[:, :] = np.triu(ci_matrix) + np.triu(ci_matrix, 1).T

    return ci_matrix


def doci_matrix(self, orb_type):
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
    H = self.H
    G = self.G
    ns = self.nspatial

    doci_matrix = np.zeros((self.nci,) * 2, dtype=self.dtype)

    # Loop only over upper triangular
    for nsd0, sd0 in enumerate(self.civec):
        for nsd1, sd1 in enumerate(self.civec[nsd0:]):
            nsd1 += nsd0
            diff_sd0, diff_sd1 = slater.diff(sd0, sd1)
            shared_indices = slater.occ_indices(slater.shared(sd0, sd1))

            if len(diff_sd0) != len(diff_sd1):
                continue
            else:
                diff_order = len(diff_sd0)

            assert diff_order % 2 == 0, 'One (or both) of the Slater determinants, {0} or {1},'
            'are not DOCI Slater determinants'.format(bin(sd0), bin(sd1))

            # two sd's are different by double excitation
            if diff_order == 2:
                i, j = diff_sd0
                k, l = diff_sd1
                if spatial_index(i, ns) == spatial_index(k, ns) and spatial_index(j, ns) == spatial_index(l, ns):
                    doci_matrix[nsd0, nsd1] += get_G_value(G, i, j, k, l, orb_type=orb_type)
                elif spatial_index(i, ns) == spatial_index(l, ns) and spatial_index(j, ns) == spatial_index(k, ns):
                    doci_matrix[nsd0, nsd1] -= get_G_value(G, i, j, l, k, orb_type=orb_type)
                else:
                    assert True, 'One (or both) of the Slater determinants, {0} or {1},'
                    ' are not DOCI Slater determinants'.format(bin(sd0), bin(sd1))

            # two sd's are the same
            elif diff_order == 0:
                for ic, i in enumerate(shared_indices):
                    doci_matrix[nsd0, nsd1] += get_H_value(H, i, i, orb_type=orb_type)
                    for j in shared_indices[ic + 1:]:
                        doci_matrix[nsd0, nsd1] += get_G_value(G, i, j, i, j, orb_type=orb_type)
                        doci_matrix[nsd0, nsd1] -= get_G_value(G, i, j, j, i, orb_type=orb_type)

    # Make it Hermitian
    doci_matrix[:, :] = np.triu(doci_matrix) + np.tril(doci_matrix, -1)

    return doci_matrix
