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


def borchardt(matrix):
    """
    Compute the permanent of a Borchardt/Cauchy matrix.

    Parameters
    ----------
    matrix : 2-index np.ndarray
        A square matrix.

    Returns
    -------
    permanent : number

    """

    return np.linalg.det(matrix ** 2) / np.linalg.det(matrix)


def borchardt_deriv(matrix, x, c):
    """
    Compute the partial derivative of the permanent of a Cauchy matrix.

    Parameters
    ----------
    matrix : 2-index np.ndarray
        A square matrix.
    x : 1-d np.ndarray
        The coefficient vector corresponding to the Cauchy matrix.
    c : int
        The coefficient with respect to which to differentiate.

    Returns
    -------
    deriv : number

    """

    p = matrix.shape[0]
    k = matrix.shape[1]

    # Compute elementwise square
    ewsquare = matrix.copy()
    ewsquare **= 2

    # Compute determinants
    det_matrix = np.linalg.det(matrix)
    det_ewsquare = np.linalg.det(ewsquare)
    inv_det_matrix = 1 / det_matrix

    # Compute adjoints
    adj_matrix = np.empty(matrix.shape, dtype=matrix.dtype)
    adj_ewsquare = np.empty(ewsquare.shape, dtype=matrix.dtype)
    for i in range(p):
        for j in range(k):
            sgn = -1 if ((i + j) % 2) else 1
            xlist = list(range(p))
            xlist.remove(i)
            xlist = np.array(xlist)[:, np.newaxis]
            ylist = list(range(k))
            ylist.remove(j)
            adj_matrix[j, i] = sgn * np.linalg.det(matrix[xlist, ylist])
            adj_ewsquare[j, i] = sgn * np.linalg.det(ewsquare[xlist, ylist])

    # Compute derivative of `matrix` and `ewsquare`
    d_matrix = np.zeros(matrix.shape, dtype=matrix.dtype)
    d_ewsquare = np.zeros(ewsquare.shape, dtype=ewsquare.dtype)
    # If deriving wrt {z_j}
    if c < k:
        for i in range(p):
            d_matrix[i, c] = 1 / (x[2 * k + i] + x[k + c])
            d_ewsquare[i, c] = 2 * x[c] / ((x[2 * k + i] + x[k + c]) ** 2)
    # If deriving wrt {e_j}
    elif c < 2 * k:
        for i in range(p):
            d_matrix[i, c - k] = -x[c - k] / ((x[2 * k + i] + x[c]) ** 2)
            d_ewsquare[i, c - k] = -2 * (x[c - k] ** 2) / ((x[2 * k + i] + x[c]) ** 3)
    # If deriving wrt {l_p}
    else:
        for j in range(k):
            d_matrix[c - 2 * k, j] = -x[j] / ((x[c] + x[k + j]) ** 2)
            d_ewsquare[c - 2 * k, j] = -2 * (x[j] ** 2) / ((x[c] + x[k + j]) ** 3)

    # Compute total derivative
    deriv = np.trace(adj_ewsquare.dot(d_ewsquare)) \
        - det_ewsquare * inv_det_matrix * np.trace(adj_matrix.dot(d_matrix))
    deriv *= inv_det_matrix

    return deriv


def borchardt_solve(matrix, x):
    """
    Obtain the Cauchy matrix parameters for a matrix.

    Parameters
    ----------
    matrix : 2-index np.ndarray
        The matrix to be fitted.
    x : 1-index np.ndarray
        An initial guess at the Cauchy matrix parameters.

    Returns
    -------
    x : 1-index np.ndarray

    """

    def solve(x):
        p = matrix.shape[0]
        k = matrix.shape[1]
        obj = []
        for i in range(p):
            for j in range(k):
                obj.append(matrix[i, j] - x[j] / (x[k + j] + x[2 * k + i]))
        return np.array(obj, dtype=x.dtype)

    ubound = np.ones(x.shape)
    ubound[matrix.shape[1]:] *= np.inf
    lbound = ubound.copy()
    lbound *= -1

    solution = least_squares(solve, x, bounds=(lbound, ubound))

    return solution["x"]


def borchardt_factor(x, p):
    """
    Factor a vector of Cauchy matrix parameters into the Cauchy matrix.

    Parameters
    ----------
    x : 1-index np.ndarray
        A vector of Cauchy matrix parameters.
    p : int
        The number of rows in the Cauchy matrix.

    Returns
    -------
    matrix : 2-index np.ndarray

    """

    k = (x.size - p) // 2
    matrix = np.empty((p, k), dtype=x.dtype)
    for i in range(p):
        for j in range(k):
            matrix[i, j] = x[j] / (x[2 * k + i] + x[k + j])
  
    return matrix
