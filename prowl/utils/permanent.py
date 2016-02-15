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
    return np.linalg.inv(matrix) * np.linalg.det(matrix)


def apr2g(matrix):
    return np.linalg.det(matrix ** 2) / np.linalg.det(matrix)


def apr2g_deriv(matrix, x, p, cols, dx):
    """
    Compute the partial derivative of the permanent of a Cauchy matrix.

    """

    # Initialize structures and check for d|C|+/dx == 0
    k = matrix.shape[1]
    indices = list(range(p))
    indices.extend([p + c for c in cols])
    indices.extend([p + k + c for c in cols])
    if dx not in indices:
        return 0
    dx = indices.index(dx)
    xnew = x[indices]

    # Compute square matrix slice and its elementwise square
    matrix = matrix[:, cols]
    ewsquare = matrix.copy()
    ewsquare **= 2

    # Compute determinants
    det_matrix = np.linalg.det(matrix)
    det_ewsquare = np.linalg.det(ewsquare)

    # Compute adjoints
    adj_matrix = adjugate(matrix)
    adj_ewsquare = adjugate(ewsquare)

    # Compute derivative of `matrix` and `ewsquare`
    d_matrix = np.zeros_like(matrix)
    d_ewsquare = np.zeros_like(ewsquare)
    for i in range(p):
        for j in range(p):
            # If deriving wrt {l_i}
            if dx == i:
                d_matrix[i, j] = -xnew[2 * p + j] / ((xnew[p] - xnew[p + j]) ** 2)
                d_ewsquare[i, j] = -2 * (xnew[2 * p + j] ** 2) / ((xnew[p] - xnew[p + j]) ** 3)
            # If deriving wrt {e_j}
            elif dx == (p + j):
                d_matrix[i, j] = xnew[2 * p + j] / ((xnew[p] - xnew[p + j]) ** 2)
                d_ewsquare[i, j] = 2 * (xnew[2 * p + j] ** 2) / ((xnew[p] - xnew[p + j]) ** 3)
            # If deriving wrt {z_j}
            elif dx == (2 * p + j):
                d_matrix[i, j] = 1 / (xnew[p] - xnew[p + j])
                d_ewsquare[i, j] = 2 * xnew[2 * p + j] / ((xnew[p] - xnew[p + j]) ** 2)
            # Else
            else:
                d_matrix[i, j] = 0
                d_ewsquare[i, j] = 0

    # Compute total derivative
    deriv = np.trace(adj_ewsquare.dot(d_ewsquare)) / det_matrix \
        - det_ewsquare * (1 / (det_matrix ** 2)) * np.trace(adj_matrix.dot(d_matrix))

    return deriv


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
