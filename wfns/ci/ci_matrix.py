import numpy as np
from .. import slater


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
        if slater.is_alpha(i, ns) == slater.is_alpha(k, ns):
            I = slater.spatial_index(i, ns)
            K = slater.spatial_index(k, ns)
            return H_matrices[0][I, K]
    elif orb_type == 'unrestricted':
        # Assume that H_matrices is expressed wrt spin orbitals
        ns = H_matrices[0].shape[0]
        # number of spatial orbitals is number of spin orbitals divided by 2
        I = slater.spatial_index(i, ns)
        K = slater.spatial_index(k, ns)
        if slater.is_alpha(i, ns) and slater.is_alpha(k, ns):
            return H_matrices[0][I, K]
        elif not slater.is_alpha(i, ns) and not slater.is_alpha(k, ns):
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
        if slater.is_alpha(i, ns) == slater.is_alpha(k, ns) and slater.is_alpha(j, ns) == slater.is_alpha(l, ns):
            I = slater.spatial_index(i, ns)
            J = slater.spatial_index(j, ns)
            K = slater.spatial_index(k, ns)
            L = slater.spatial_index(l, ns)
            return G_matrices[0][I, J, K, L]
    elif orb_type == 'unrestricted':
        # Assume that H_matrices is expressed wrt spin orbitals
        ns = G_matrices[0].shape[0]
        # number of spatial orbitals is number of spin orbitals divided by 2
        I = slater.spatial_index(i, ns)
        J = slater.spatial_index(j, ns)
        K = slater.spatial_index(k, ns)
        L = slater.spatial_index(l, ns)
        if slater.is_alpha(i, ns) and slater.is_alpha(j, ns) and slater.is_alpha(k, ns) and slater.is_alpha(l, ns):
            return G_matrices[0][I, J, K, L]
        elif slater.is_alpha(i, ns) and not slater.is_alpha(j, ns) and slater.is_alpha(k, ns) and not slater.is_alpha(l, ns):
            return G_matrices[1][I, J, K, L]
        elif not slater.is_alpha(i, ns) and slater.is_alpha(j, ns) and not slater.is_alpha(k, ns) and slater.is_alpha(l, ns):
            # FIXME: check the transposition
            # take the appropraite transpose to get the beta alpha beta alpha form
            return np.einsum('IJKL->JILK', G_matrices[1])[I, J, K, L]
        elif not slater.is_alpha(i, ns) and not slater.is_alpha(j, ns) and not slater.is_alpha(k, ns) and not slater.is_alpha(l, ns):
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

            # moving all the shared orbitals toward one another (in the middle)
            num_transpositions_0 = sum(len([j for j in shared_indices if j<i]) for i in diff_sd0)
            num_transpositions_1 = sum(len([j for j in shared_indices if j<i]) for i in diff_sd1)
            num_transpositions = num_transpositions_0 + num_transpositions_1
            sign = (-1)**num_transpositions

            assert diff_order % 2 == 0, 'One (or both) of the Slater determinants, {0} or {1},'
            'are not DOCI Slater determinants'.format(bin(sd0), bin(sd1))

            # two sd's are different by double excitation
            if diff_order == 2:
                i, j = diff_sd0
                k, l = diff_sd1
                doci_matrix[nsd0, nsd1] += get_G_value(G, i, j, k, l, orb_type=orb_type) * sign
                doci_matrix[nsd0, nsd1] -= get_G_value(G, i, j, l, k, orb_type=orb_type) * sign

            # two sd's are the same
            elif diff_order == 0:
                for ic, i in enumerate(shared_indices):
                    doci_matrix[nsd0, nsd1] += get_H_value(H, i, i, orb_type=orb_type)
                    for j in shared_indices[ic + 1:]:
                        doci_matrix[nsd0, nsd1] += get_G_value(G, i, j, i, j, orb_type=orb_type) * sign
                        doci_matrix[nsd0, nsd1] -= get_G_value(G, i, j, j, i, orb_type=orb_type) * sign

    # Make it Hermitian
    doci_matrix[:, :] = np.triu(doci_matrix) + np.triu(doci_matrix, 1).T

    return doci_matrix
