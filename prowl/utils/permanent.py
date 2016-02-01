from __future__ import absolute_import, division, print_function

from itertools import permutations
import numpy as np


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
