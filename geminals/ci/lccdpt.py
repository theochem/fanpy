import numpy as np

from .ci_matrix import get_H_value, get_G_value
from ..sd_list import doci_sd_list, ci_sd_list


#def doci_sd_list(self, num_limit, exc_orders=[]):
def make_system(sds, H_matrices, G_matrices):
    ''' Compute CI matrix elements needed to solve L-CCD perturbatively
    TODO: Write Latex version of the L-CCD system here

    '''

    ref = sds[0]
    sds = sds[1:]
    k = len(sds)

    ref_h_ref = get_H_value(H_matrices, ref, ref, 'restricted') + get_G_value(G_matrices, ref, ref, 'restricted')

    A = np.empty((k+1, k+1))
    A[0,0] = -1
    A[1:,0] = 0
    A[0,1:] = 0
    block = A[1:,1:]

    b = np.empty(k+1)
    b[0] = -ref_H_ref

    # Bottom right sub-matrix is symmetric; k and l take on value of every slater determinant
    for i,k in enumerate(sds):
        b[i+1] = get_H_value(H_matrices, k, ref, 'restricted') + get_G_value(G_matrices, k, ref, 'restricted')

        for j,l in enumerate(sds[(i+1):]):
            temp = get_H_value(H_matrices, k, l, 'restricted') + get_G_value(G_matrices, k, l, 'restricted')
            block[i,(i+j)] = temp
            block[(i+j),i] = temp

        block[i,i] = get_H_value(H_matrices, k, k, 'restricted') + get_G_value(G_matrices, k, k, 'restricted') - ref_h_ref

    c = np.linalg.solve(A,b)
    energy = c[0]
    ci_coeffs = c[1:]

    return energy, ci_coeffs
