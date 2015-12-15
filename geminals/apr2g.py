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

    def _generate_x0(self):
        """
        See APIG._generate_x0().

        """

        # This is NOT a good guess, it's a placeholder!
        x0 = 2.0*(np.random.rand(2 * self.norbs + self.npairs) - 0.5)
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
