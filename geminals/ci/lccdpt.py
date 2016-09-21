import numpy as np

from .ci_matrix import phi_H_phi
from ..sd_list import apig_doubles_sd_list, apsetg_doubles_sd_list, apg_doubles_sd_list, doci_sd_list


def solve_lccdpt(self, H_matrices, G_matrices, geminal_type):
    ''' Compute CI matrix elements needed to solve L-CCD perturbatively
    TODO: Write Latex version of the L-CCD system here
    
    Parameters
    ----------
    self: A geminal instance
    H_matrices: A numpy.array or a tuple of numpy arrays
        One electron integrals, rectricted or unrestricted
    G_matrices: A numpy.array or a tuple of numpy arrays
        Two electron integrals, rectricted or unrestricted
    geminal_type: str
        The type of geminal for which to make a guess

    Returns
    -------
    energy: float
    ci_coeffs: np.ndarray
        Array of CI coefficients
    sds: list
        Corresponding list of CI Slater Determinants
    '''

    # Generate appropriate projection space of sds
    if geminal_type == 'apig':
        sds = apig_doubles_sd_list(self)
    elif geminal_type == 'apsetg':
        sds = apsetg_doubles_sd_list(self)
    elif geminal_type == 'apg':
        sds = apg_doubles_sd_list(self)
    else:
        raise ValueError("The geminal_type is incorrect, It must be one of: ('apig'|'apsetg'|'apg')")

    # Define reference determinant and excited states
    ref = sds[0]
    sds = sds[1:]
    k = len(sds)

    # Compute <phi_0|H|phi_0>
    ref_h_ref = phi_H_phi(ref, ref, H_matrices, G_matrices, 'restricted')

    # Define matrix A for Ax=b
    A = np.empty((k+1, k+1))
    A[0,0] = -1
    A[1:,0] = 0
    A_submatrix = A[1:,1:]

    # b is the solution vector
    b = np.empty(k+1)
    b[0] = -ref_h_ref

    # Bottom right sub-matrix is symmetric; k and l take on value of every slater determinant
    for i,k in enumerate(sds):
        b[i+1] = phi_H_phi(k, ref, H_matrices, G_matrices, 'restricted')
        for j,l in enumerate(sds[(i+1):]):
            temp = phi_H_phi(k, l, H_matrices, G_matrices, 'restricted')
            A_submatrix[i,(i+j)] = temp
            A_submatrix[(i+j),i] = temp
        A_submatrix[i,i] = phi_H_phi(k, k, H_matrices, G_matrices, 'restricted') - ref_h_ref

    # The first row of A is the same as b
    A[0,1:] = b[1:]

    # Solve and return coefficients
    c = np.linalg.solve(A,b)
    energy = c[0]
    ci_coeffs = c[1:]

    return energy, ci_coeffs, sds
