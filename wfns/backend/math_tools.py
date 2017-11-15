r"""Functions for doing some math.

Functions
---------
binomial(n, k)
    Returns `n choose k`
permanent_combinatoric(matrix)
    Computes the permanent of a matrix using brute force combinatorics
permanent_ryser(matrix)
    Computes the permanent of a matrix using Ryser algorithm
adjugate(matrix)
    Returns adjugate of a matrix
permanent_borchardt(matrix)
    Computes the permanent of rank-2 Cauchy matrix

"""
from itertools import permutations, combinations
import numpy as np
from scipy.misc import comb


def binomial(n, k):
    r"""Return the binomial coefficient :math:`\binom{n}{k}`.

    .. math::

        \binom{n}{k} = \frac{n!}{k! (n-k)!}

    Parameters
    ----------
    n : int
        n in (n choose k).
    k : int
        k in (n choose k).

    Returns
    -------
    result : int
        Number of ways to select :math:`k` objects out of :math:`n` objects.

    """
    return comb(n, k, exact=True)


def adjugate(matrix):
    r"""Return the adjugate of a matrix.

    Adjugate of a matrix is the transpose of its cofactor matrix

    .. math::

        adj(A) = det(A) A^{-1}

    Returns
    -------
    adjugate : float
        Transpose of the cofactor matrix.

    Raises
    ------
    ValueError
        If matrix has a size of zero.
    LinAlgError
        If matrix is singular (determinant of zero).
        If matrix is not two dimensional.

    """
    if matrix.size == 0:
        raise ValueError('Given matrix has nothing inside.')

    det = np.linalg.det(matrix)
    if abs(det) <= 1e-12:
        raise np.linalg.LinAlgError('Matrix is singular')
    return det * np.linalg.inv(matrix)


def permanent_combinatoric(matrix):
    r"""Calculate the permanent of a matrix naively using combinatorics (brute force).

    If :math:`A` is an :math:`m` by :math:`n` matrix

    .. math::

        perm(A) = \sum_{\sigma \in P_{n,m}} \prod_{i=1}^n a_{i,\sigma(i)}

    The cost of evaluation is :math:`\mathcal{O}(n!)`.

    Parameters
    ----------
    matrix : {np.ndarray(nrow, ncol)}
        Matrix whose permanent will be evaluated.

    Returns
    -------
    permanent : float
        Permanent of the matrix.

    Raises
    ------
    ValueError
        If matrix is not two dimensional.
        If matrix has no numbers.

    """
    nrow, ncol = matrix.shape
    if nrow == 0 or ncol == 0:
        raise ValueError('Given matrix has no numbers')

    # Ensure that the number of rows is less than or equal to the number of columns
    if nrow > ncol:
        nrow, ncol = ncol, nrow
        matrix = matrix.transpose()

    # Sum over all permutations
    rows = np.arange(nrow)
    cols = range(ncol)
    permanent = 0.0
    for perm in permutations(cols, nrow):
        # multiply together all the entries that correspond to
        # matrix[rows[0], perm[0]], matrix[rows[1], perm[1]], ...
        permanent += np.prod(matrix[rows, perm])

    return permanent


def permanent_ryser(matrix):
    r"""Calculate the permanent of a square or rectangular matrix using the Borchardt theorem.

    Borchardt theorem is as follows iIf a matrix is rank two (Cauchy) matrix of the form

    .. math::

        A_{ij} = \frac{1}{\epsilon_j - \lambda_i}

    Then

    .. math::

        perm(A) = det(A \circ A) det(A^{-1})

    Parameters
    ----------
    lambdas : {np.ndarray(M, )}
        Flattened row matrix of the form :math:`\lambda_i`.
    epsilons : {np.ndarray(N, )}
        Flattened column matrix of the form :math:`\epsilon_j`.
    zetas : {np.ndarray(N, )}
        Flattened column matrix of the form :math:`\zeta_j`.
    etas : {None, np.ndarray(M, )}
        Flattened row matrix of the form :math:`\eta_i`.
        By default, all of the etas are set to 1.

    Returns
    -------
    result : float
        permanent of the rank-2 matrix built from the given parameter list.

    Raises
    ------
    ValueError
        If the number of zetas and epsilons (number of columns) are not equal.
        If the number of etas and lambdas (number of rows) are not equal.

    """

    # on rectangular matrices A(m, n) where m <= n.
    nrow, ncol = matrix.shape
    if nrow == 0 or ncol == 0:
        raise ValueError('Given matrix has no numbers')

    factor = 1.0
    # if rectangular
    if nrow != ncol:
        if nrow > ncol:
            matrix = matrix.transpose()
            nrow, ncol = ncol, nrow
        matrix = np.pad(matrix, ((0, ncol - nrow), (0, 0)), mode="constant",
                        constant_values=((0, 1.0), (0, 0)))
        factor /= np.math.factorial(ncol - nrow)

    # Initialize rowsum array.
    rowsums = np.zeros(ncol, dtype=matrix.dtype)
    sign = bool(ncol % 2)
    permanent = 0.0

    # Initialize the Gray code.
    graycode = np.zeros(ncol, dtype=bool)

    # Compute all permuted rowsums.
    while not all(graycode):

        # Update the Gray code
        flag = False
        for i in range(ncol):
            # Determine which bit will change
            if not graycode[i]:
                graycode[i] = True
                cur_position = i
                flag = True
            else:
                graycode[i] = False
            # Update the current value
            if cur_position == ncol - 1:
                cur_value = graycode[ncol - 1]
            else:
                cur_value = not (graycode[cur_position] and graycode[cur_position + 1])
            if flag:
                break

        # Update the rowsum array.
        if cur_value:
            rowsums[:] += matrix[:, cur_position]
        else:
            rowsums[:] -= matrix[:, cur_position]

        # Compute the next rowsum permutation.
        if sign:
            permanent += np.prod(rowsums)
            sign = False
        else:
            permanent -= np.prod(rowsums)
            sign = True

    return permanent * factor


def permanent_borchardt(lambdas, epsilons, zetas, etas=None):
    r"""Calculate the permanent of a square or rectangular matrix using the Borchardt theorem.

    Borchardt theorem is as follows iIf a matrix is rank two (Cauchy) matrix of the form

    .. math::

        A_{ij} = \frac{1}{\epsilon_j - \lambda_i}

    Then

    .. math::

        perm(A) = det(A \circ A) det(A^{-1})

    Parameters
    ----------
    lambdas : {np.ndarray(M, )}
        Flattened row matrix of the form :math:`\lambda_i`.
    epsilons : {np.ndarray(N, )}
        Flattened column matrix of the form :math:`\epsilon_j`.
    zetas : {np.ndarray(N, )}
        Flattened column matrix of the form :math:`\zeta_j`.
    etas : {None, np.ndarray(M, )}
        Flattened row matrix of the form :math:`\eta_i`.
        By default, all of the etas are set to 1.

    Returns
    -------
    result : float
        permanent of the rank-2 matrix built from the given parameter list.

    Raises
    ------
    ValueError
        If the number of zetas and epsilons (number of columns) are not equal.
        If the number of etas and lambdas (number of rows) are not equal.

    """
    if zetas.size != epsilons.size:
        raise ValueError('The the number of zetas and epsilons must be equal')
    if etas is not None and etas.size != lambdas.size:
        raise ValueError('The number of etas and lambdas must be equal')

    num_row = lambdas.size
    num_col = epsilons.size
    if etas is None:
        etas = np.ones(num_row)

    cauchy_matrix = 1 / (lambdas[:, np.newaxis] - epsilons)
    if num_row > num_col:
        num_row, num_col = num_col, num_row
        zetas, etas = etas, zetas
        cauchy_matrix = cauchy_matrix.T

    perm_cauchy = 0
    for indices in combinations(range(num_col), num_row):
        indices = np.array(indices)
        submatrix = cauchy_matrix[:, indices]
        perm_zetas = np.product(zetas[indices])
        perm_cauchy += np.linalg.det(submatrix**2) / np.linalg.det(submatrix) * perm_zetas

    perm_etas = np.prod(etas)

    return perm_cauchy * perm_etas
