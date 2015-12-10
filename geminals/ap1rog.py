"""
AP1roG geminal wavefunction class.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from geminals.apig import APIG
from geminals.slater_det import excite_pairs, is_pair_occupied


class AP1roG(APIG):
    """
    A restricted antisymmetrized product of one reference orbital geminals ((R)AP1roG)
    implementation.

    See APIG class documentation.

    """

    #
    # Class-wide (behaviour-changing) attributes and properties
    #

    _exclude_ground = True
    _normalize = False

    @property
    def _row_indices(self):
        return range(0, self.npairs)

    @property
    def _col_indices(self):
        return range(self.npairs, self.norbs)

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        if value is None:
            return
        elif len(value.shape) == 1:
            assert value.size == self.npairs * (self.norbs - self.npairs), \
                "The guess `x0` is not the correct size for the geminal coefficient matrix."
            coeffs = np.eye(self.npairs, M=self.norbs)
            coeffs[:, self.npairs:] += value.reshape(self.npairs, self.norbs - self.npairs)
            self._coeffs = coeffs
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

        params = self.npairs * (self.norbs - self.npairs)
        scale = 2.0 / np.around(10 * params, decimals=-1)
        return scale * (np.random.rand(params) - 0.5)

    def generate_pspace(self):
        """
        See APIG.generate_pspace().

        """

        ground = self.ground
        pspace = [ground]

        # Return a tuple of all unique pair excitations
        for unoccup in range(self.npairs, self.norbs):
            for i in range(self.npairs):
                pspace.append(excite_pairs(ground, i, unoccup))
        return tuple(set(pspace))

    def _construct_coeffs(self, x0):
        """
        See APIG._construct_coeffs().

        """

        coeffs = np.eye(self.npairs, M=self.norbs)
        coeffs[:, self.npairs:] += x0.reshape(self.npairs, self.norbs - self.npairs)
        return coeffs

    def overlap(self, phi, coeffs=None):
        """
        See APIG.overlap().

        """

        if coeffs is None:
            assert self._coeffs_optimized, \
                "The geminal coefficient matrix has not yet been optimized."
            coeffs = self.coeffs

        # If bad Slater determinant
        if phi == 0:
            return 0
        elif phi not in self.pspace:
            return 0

        # If good Slater determinant, get the pair-excitation indices
        from_index = []
        to_index = []
        excite_count = 0
        for i in range(self.npairs):
            if not is_pair_occupied(phi, i):
                excite_count += 1
                from_index.append(i)
        for i in range(self.npairs, self.norbs):
            if is_pair_occupied(phi, i):
                to_index.append(i)

        # If it's not excited
        if excite_count == 0:
            # If deriving wrt one of the non-diagonals of the identity block
            if self._overlap_derivative and self._overlap_indices[0] != self._overlap_indices[1]:
                return 0
            # If not deriving, or if deriving wrt a diagonal
            return 1

        # If it's singly pair-excited
        if excite_count == 1:
            # If deriving
            if self._overlap_derivative:
                # If the coefficient wrt which we are deriving is substituted in
                if self._overlap_indices == (from_index[0], to_index[0]):
                    return 1
                # If deriving wrt an element of the identity matrix
                if self._overlap_indices[0] == self._overlap_indices[1]:
                    return coeffs[from_index[0], to_index[0]]
                # If deriving wrt another element
                return 0
            # If we're not deriving
            return coeffs[from_index[0], to_index[0]]

        # If something went wrong
        raise ValueError("The AP1roG implementation cannot handle multiple pair excitations.")

# vim: set textwidth=90 :
