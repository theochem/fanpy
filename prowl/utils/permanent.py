from __future__ import absolute_import, division, print_function

from itertools import permutations
import numpy as np
from scipy.optimize import least_squares


def dense(matrix):
    """
    Compute the permanent of a matrix using regular, dense linear algebra.

    Parameters
    ----------
    matrix : 2-index np.ndarray
        A square matrix.

    Returns
    -------
    permanent : number

    """

    permanent = 0
    rows = range(matrix.shape[0])
    for cols in permutations(rows):
        permanent += np.product(matrix[rows, cols])

    return permanent


def dense_deriv(matrix, x, y):
    """
    Compute the partial derivative of the permanent of a matrix using regular,
    dense linear algebra.

    Parameters
    ----------
    matrix : 2-index np.ndarray
        A square matrix.
    x : int
        The row of the coefficient with respect to which to differentiate.
    y : int
        The column of the coefficient with respect to which to differentiate.

    Returns
    -------
    deriv : number

    """

    deriv = 0
    rows = list(range(matrix.shape[0]))
    for cols in (i for i in permutations(rows) if i[x] == y):
        deriv += np.product([matrix[rows[i], cols[i]] for i in rows if i != x])

    return deriv


def adjugate(matrix):

    # If matrix only has one row
    if matrix.size == 1:
        return np.array([1])

    # Get row and column slice lists
    rows = list(range(matrix.shape[0]))
    cols = list(range(matrix.shape[1]))

    # Compute the adjugate matrix
    adj = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rows.remove(i)
            cols.remove(j)
            sgn = -1 if ((i + j) % 2) else 1
            t_rows = np.array(rows)[:, np.newaxis]
            adj[j, i] = sgn * np.linalg.det(matrix[t_rows, cols])
            rows.insert(i, i)
            cols.insert(j, j)

    return adj


def apr2g(x, p, cols):

    # |C|+ == |F|*|diag(zetas)|
    k = (x.size - p) // 2
    F = faribault_factor(x, p, cols)
    zeta_indices = [p + k + c for c in cols]

    return np.linalg.det(F) * np.prod(x[zeta_indices])


def apr2g_deriv(x, p, cols, dx):

    # Initialize structures and check for d|C|+/dx == 0
    k = (x.size - p) // 2
    indices = list(range(p))
    indices.extend([p + c for c in cols])
    indices.extend([p + k + c for c in cols])
    if dx not in indices:
        return 0
    dx = indices.index(dx)
    xnew = x[indices]

    # Initialize Faribault matrices
    F = faribault_factor(x, p, cols)
    dF = np.zeros_like(F)
    D = xnew[(2 * p):].copy()
    dD = np.zeros_like(D)

    # dx == lambda_x
    if dx < p:
        for i in range(p):
            for j in range(p):
                if dx == i == j:
                    dF[i, j] = -1 / ((xnew[i] - xnew[p + i]) ** 2)
                elif dx == j != i:
                    dF[i, j] = -1 / ((xnew[j] - xnew[p + i]) ** 2)
                elif i == j:
                    dF[i, j] = -1 / ((xnew[dx] - xnew[p + i]) ** 2)
                else:
                    dF[i, j] = 0

    # dx == epsilon_x
    elif dx < 2 * p:
        for i in range(p):
            for j in range(p):
                if dx == (i + p):
                    if i == j:
                        for l in range(p):
                            if l != i:
                                dF[i, j] += -1 / ((xnew[p + i] - xnew[p + l]) ** 2)
                            dF[i, j] += 1 / ((xnew[l] - xnew[p + i]) ** 2)
                    else:
                        dF[i, j] = -1 / ((xnew[p + i] - xnew[p + j]) ** 2)
                elif dx == (j + p) != (i + p):
                    dF[i, j] = 1 / ((xnew[p + i] - xnew[p + j]) ** 2)
                elif i == j:
                    dF[i, j] = 1 / ((xnew[p + i] - xnew[dx]) ** 2)
                else:
                    dF[i, j] = 0
    # dx == zeta_x
    else:
        dD[dx - 2 * p] = 1

    # Fix `dF`, `D`, and `dD`'s dimensions
    D = np.diag(D)
    dD = np.diag(dD)

    # Use Jacobi's formula with the product rule
    deriv = np.linalg.det(F) * np.trace(adjugate(D).dot(dD)) \
        + np.linalg.det(D) * np.trace(adjugate(F).dot(dF))

    return deriv

#    # `dx` is a denominator parameter
#    dF = np.zeros_like(F)
#    if dx < (p + k):
#
#        # `dx` is lambda
#        if dx < p:
#            # Update lambda-containing terms
#            for i in range(p):
#                dF[i, i] = -1 / ((x[dx] - x[p + i]) ** 2)
#
#        # `dx` is epsilon
#        else:
#            for i in range(p):
#                for j in range(p):
#                    # `dx` == `i`
#                    if i == (dx - p):
#                        # Update diagonal terms
#                        if i == j:
#                            for l in range(p):
#                                dF[i, i] += 1 / ((x[l] - x[dx]) ** 2)
#                                if l != i:
#                                    dF[i, i] += -1 / ((x[dx] - x[p + l]) ** 2)
#                        # Update off-diagonal terms
#                        else:
#                            dF[i, j] = -1 / ((x[dx] - x[p + j]) ** 2)
#                    # `dx` == `j`
#                    elif j == (dx - p):
#                        # Update off-diagonal terms
#                        if i != j:
#                            dF[i, j] = 1 / ((x[p + i] - x[dx]) ** 2)
#                    # `dx` != `i`, `dx` != `j`
#                    else:
#                        # Update diagonal terms (off-diagonals are zero)
#                        if i == j:
#                            dF[i, i] = 1 / ((x[p + i] - x[dx]) ** 2)
#
#        # Use Jacobi's formula for denominator `dx`s, then multiply by the zetas
#        return np.trace(adjugate(F).dot(dF)) * np.prod(x[zeta_indices])
#
#    # `dx` is a numerator parameter; just remove the zeta `dx`
#    else:
#        dx_index = dx - p - k
#        zetas = np.diag(x[(p + k):].copy())
#        dzetas = zetas.copy()
#        dzetas[dx_index, dx_index] = 1
#        return np.linalg.det(F) * np.trace(adjugate(zetas).dot(dzetas))


def cauchy_factor(x, p):

    # Initialize the Cauchy matrix
    k = (x.size - p) // 2
    C = np.empty((p, k), dtype=x.dtype)

    # Compute the terms
    for i in range(p):
        for j in range(k):
            C[i, j] = x[p + k + j] / (x[i] - x[p + j])

    # Done
    return C


def faribault_factor(x0, p, cols):

    # Get the appropriate slice of `x0`
    k = (x0.size - p) // 2
    indices = list(range(p))
    indices.extend([p + c for c in cols])
    x = x0[indices]

    # Initialize the Faribault matrix
    F = np.empty((p, p), dtype=x.dtype)

    # Compute the off-diagonal terms
    for i in range(p):
        for j in range(p):
            if i != j:
                F[i, j] = 1 / (x[p + i] - x[p + j])

    # Compute the diagonal terms
    for i in range(p):
        tmp = 0
        for l in range(p):
            if l != i:
                tmp += 1 / (x[p + i] - x[p + l])
            tmp += 1 / (x[l] - x[p + i])
        F[i, i] = tmp

    # Done
    return F
