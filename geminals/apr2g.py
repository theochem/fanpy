"""
APr2G geminal wavefunction class.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from geminals.apig import APIG
from geminals.slater_det import excite_pairs, is_pair_occupied


class APr2G(APIG):
    """
    A restricted antisymmetrized product of rank-two geminals (R(APr2G)) implementation.

    See APIG class documentation.

    """

    #
    # Class-wide (behaviour-changing) attributes and properties
    #

    _exclude_ground = False
    _normalize = True

    @property
    def _row_indices(self):
        return range(0, self.npairs)

    @property
    def _col_indices(self):
        return range(0, self.norbs)

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        if value is None:
            return
        elif len(value.shape) == 1:
            assert value.size == (2 * self.norbs + self.npairs), \
                "The guess `x0` is not the correct size for the geminal coefficient matrix."
            self._coeffs = self._construct_coeffs(value)
        else:
            assert value.shape == (self.npairs, self.norbs), \
                "The specified geminal coefficient matrix must have shape (npairs, norbs)."
            self._coeffs = value
        self._coeffs_optimized = True

    #
    # Methods
    #

    def _generate_x0(self, model_coeffs=None):
        """ Generates an initial guess

        If model_coeffs is given, we try to find a set of coefficients such that
        is as closely satisfied as possible, using least squares method
        ..math::
            C_{ij} &= \frac{\zeta_i}{\epsilon_i + \lambda_j}\\
            0 &= \zeta_i - C_{ij} \epsilon_i - C_{ij} \lambda_j\\
        This is a set of linear equations. These equations will be solved by least
        squares method.

        The least square has the form of :math:`Ax=b`. Given that the :math:`b=0`
        and the unknowns are
        ..math::
            x = \begin{bmatrix}
            \zeta_1 \\ \vdots\\ \zeta_P\\
            \epsilon_1 \\ \vdots\\ \epsilon_P\\
            \lambda_1 \\ \vdots\\ \lambda_K\\
            \end{bmatrix},
        then A must be
        ..math::
            A = \begin{bmatrix}
            1 & 0 & \dots & 0 & -C_{11} & 0 & \dots & 0 & -C_{11} & 0 \dots & 0\\
            1 & 0 & \dots & 0 & -C_{12} & 0 & \dots & 0 & 0 & -C_{12} & \dots & 0\\
            \vdots & & & & & & & & & & &\\
            1 & 0 & \dots & 0 & -C_{1K} & 0 & \dots & 0 & 0 & 0 & \dots & -C_{1K}\\
            0 & 1 & \dots & 0 & 0 & -C_{2K} & \dots & 0 & -C_{21} & 0 \dots & 0\\
            \vdots & & & & & & & & & & &\\
            0 & 1 & \dots & 0 & 0 & -C_{2K} & \dots & 0 & 0 & 0 & \dots & -C_{2K}\\
            \vdots & & & & & & & & & & &\\
            0 & 0 & \dots & 1 & 0 & 0 & \dots & -C_{P1} & -C_{P1} & 0 \dots & 0\\
            \vdots & & & & & & & & & & &\\
            0 & 0 & \dots & 1 & 0 & 0 & \dots & -C_{PK} & 0 & 0 & \dots & -C_{PK}\\
            \end{bmatrix}

        Parameter
        ---------
        model_coeffs : {None, np.ndarray(P,K)}
            Coefficients after which the initial guess is modeled

        Example
        -------
        Assuming we have a system with 2 electron pairs and 4 spatial orbitals,
        we have
        ..math::
            C = \begin{bmatrix}
            C_{11} & \dots & C_{1K}\\
            \vdots & \vdots & \vdots\\
            C_{P1} & \dots & C_{PK}
            \end{bmatrix}

        ..math::
            A = \begin{bmatrix}
            1 & 0 & -C_{11} & 0 & -C_{11} & 0 & 0 & 0\\
            1 & 0 & -C_{12} & 0 & 0 & -C_{12} & 0 & 0\\
            1 & 0 & -C_{13} & 0 & 0 & 0 & -C_{13} & 0\\
            1 & 0 & -C_{14} & 0 & 0 & 0 & 0 & -C_{14}\\
            0 & 1 & 0 & -C_{21} & -C_{21} & 0 & 0 & 0\\
            0 & 1 & 0 & -C_{22} & 0 & -C_{22} & 0 & 0\\
            0 & 1 & 0 & -C_{23} & 0 & 0 & -C_{23} & 0\\
            0 & 1 & 0 & -C_{24} & 0 & 0 & 0 & -C_{24}\\
            \end{bmatrix}

        ..math::
            x = \begin{bmatrix}
            \zeta_1 \\ \zeta_2\\
            \epsilon_1 \\ \epsilon_2\\
            \lambda_1 \\ \lambda_2 \\ \lambda_3 \\ \lambda_4\\
            \end{bmatrix}
        """

        # This is NOT a good guess, it's a placeholder!
        if model_coeffs is None:
            x0 = 2.0*(np.random.rand(2 * self.norbs + self.npairs) - 0.5)
        else:
            # Here, we try to solve Ax = 0 where
            # A
            A = np.array([])
            for row_ind, row in enumerate(model_coeffs):
                temp = np.array([])
                # temporary column of ones
                temp_col_ones = np.zeros(model_coeffs.T.shape)
                temp_col_ones[:, row_ind] = 1
                # coefficients for zetas
                temp = temp.hstack((temp, temp_col_ones))
                # coefficients for epsilons
                temp = temp.hstack((temp, -temp_col_ones*row.reshape(row.size, 1)))
                # coefficients for lambdas
                temp = temp.hstack((temp, np.identity(-row)))
                A = np.vstack((A, temp))
            # b
            b = np.zeros((A.shape[0], 1))
            # Ax = b, solve for x by least squares
            x0 = np.linalg.lstsq(A, b)
        return x0

    def _construct_coeffs(self, x0):
        """
        See APIG._construct_coeffs().

        """

        coeffs = np.zeros((self.npairs, self.norbs))
        for i in range(self.npairs):
            for j in range(self.norbs):
                coeffs[i, j] = x0[j] / (x0[self.norbs + j] - x0[2 * self.norbs + i])
        return coeffs

    @staticmethod
    def permanent(matrix):
        """
        Compute the permanent of a rank-two matrix, using Borchardt's theorem.

        Parameters
        ----------
        matrix : 2-index np.ndarray
            The rank-two matrix whose permanent is to be computed.

        Returns
        -------
        permanent : float

        """

        return np.linalg.det(matrix * matrix)/np.linalg.det(matrix)

    @staticmethod
    def permanent_derivative(matrix, i, j):
        """
        Compute the partial derivative of a permanent of a rank-two matrix with respect to
        one of its coefficients, using Borchardt's theorem for permanent evaluation with
        the Sherman-Morrison update formula for inverses of rank-one matrices.

        Parameters
        ----------
        matrix : 2-index np.ndarray
            The rank-two matrix whose permanent is to be computed.
        i : int
            `i` in the indices (i, j) of the coefficient with respect to which the partial
            derivative is computed.
        j : int
            See `i`.  This is `j`.

        Returns
        -------
        derivative : float

        """

        pass

# vim: set textwidth=90 :
