from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.misc import comb
from itertools import permutations


def binomial(n, k):
    """
    Return the binomial coefficient of integers ``n`` and ``k``, or "``n``
    choose ``k``".

    Parameters
    ----------
    n : int
        n in (n choose k).
    k : int
        k in (n choose k).

    Returns
    -------
    result : int
        n choose k.

    """

    return comb(n, k, exact=True)


def permanent_combinatoric(matrix):

    permanent = 0.0
    m, n = matrix.shape
    if m > n:
        m, n = n, m
        A = matrix.transpose()
    else:
        A = matrix
    rows = np.arange(m)
    cols = range(n)
    for perm in permutations(cols, m):
        permanent += np.prod(A[rows, perm])

    return permanent


def permanent_ryser(matrix):

    # Ryser formula (Bjorklund et al. 2009, On evaluation of permanents) works
    # on rectangular matrices A(m, n) where m <= n.
    m, n = matrix.shape
    if m != n:
        if m > n:
            matrix = matrix.transpose()
            m, n = n, m
        A = np.pad(matrix, ((0, n - m), (0, 0)), mode="constant", constant_values=((0, 1.0), (0, 0)))
        factor = 1.0 / np.math.factorial(n - m)
    else:
        A = matrix
        factor = 1.0

    # Initialize rowsum array.
    rowsums = np.zeros(n, dtype=A.dtype)
    sign = bool(n % 2)
    permanent = 0.0

    # Initialize the Gray code.
    graycode = np.zeros(n, dtype=bool)

    # Compute all permuted rowsums.
    while not all(graycode):

        # Update the Gray code
        flag = False
        for i in range(n):
            # Determine which bit will change
            if not graycode[i]:
                graycode[i] = True
                cur_position = i
                flag = True
            else:
                graycode[i] = False
            # Update the current value
            if cur_position == n - 1:
                cur_value = graycode[n - 1]
            else:
                cur_value = not \
                (graycode[cur_position] and graycode[cur_position + 1])
            if flag:
                break

        # Update the rowsum array.
        if cur_value:
            rowsums[:] += A[:, cur_position]
        else:
            rowsums[:] -= A[:, cur_position]

        # Compute the next rowsum permutation.
        if sign:
            permanent += np.prod(rowsums)
            sign = False
        else:
            permanent -= np.prod(rowsums)
            sign = True

    return permanent * factor
