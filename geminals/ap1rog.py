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
    def num_params(self):
        """ Number of parameters needed

        """
        return self.npairs*(self.norbs-self.npairs)*self.offset_complex


    #
    # Methods
    #

    def generate_pspace(self):
        """
        See APIG.generate_pspace().

        """

        ground = self.ground_sd
        pspace = [ground]

        # Return a tuple of all unique pair excitations
        for unoccup in range(self.npairs, self.norbs):
            for i in range(self.npairs):
                pspace.append(excite_pairs(ground, i, unoccup))
        return tuple(set(pspace))

    def the_function(self, params, elec_config):
        """ The function that assigns a coefficient to a given configuration given some
        parameters

        Parameters
        ----------
        params : np.ndarray(M,)
            Parameters that describe the APIG geminal
        elec_config : int
            Slater determinant that corresponds to certain electron configuration

        Returns
        -------
        coefficient : float
            Coefficient of thet specified Slater determinant

        Raises
        ------
        NotImplementedError
        """
        if self._is_complex:
            # Instead of dividing params into a real and imaginary part and adding
            # them, we add the real part to the imaginary part
            # Imaginary part is assigned first because this forces the numpy array
            # to be complex
            temp_params = 1j*params[params.size//2:]
            # Add the real part
            temp_params += params[:params.size//2]
            params = temp_params
        # TODO: make some sort of determinant "difference"
        excite_from = []
        excite_to = []
        for i in range(self.npairs):
            if not is_pair_occupied(elec_config, i):
                excite_from.append(i)
        for i in range(self.npairs, self.norbs):
            if is_pair_occupied(elec_config, i):
                excite_to.append(i)
        excite_count = len(excite_to)

        assert excite_count in [0, 1]
        if excite_count == 0:
            return 1
        elif excite_count == 1:
            return params[excite_from[0]*(self.norbs-self.npairs)+(excite_to[0]-self.npairs)]

    def differentiate_the_function(self, params, elec_config, index):
        """ Differentiates the function wrt to a specific parameter

        Parameters
        ----------
        params : np.ndarray(M,)
            Set of numbers
        elec_config : int
            Slater determinant that corresponds to certain electron configuration
        index : int
            Index of the parameter with which we are differentiating

        Returns
        -------
        coefficient : float
            Coefficient of thet specified Slater determinant

        Raises
        ------
        NotImplementedError
        """
        assert 0 <= index < params.size
        if self._is_complex:
            # Instead of dividing params into a real and imaginary part and adding
            # them, we add the real part to the imaginary part
            # Imaginary part is assigned first because this forces the numpy array
            # to be complex
            temp_params = 1j*params[params.size//2:]
            # Add the real part
            temp_params += params[:params.size//2]
            params = temp_params
        # TODO: make some sort of determinant "difference"
        excite_from = []
        excite_to = []
        for i in range(self.npairs):
            if not is_pair_occupied(elec_config, i):
                excite_from.append(i)
        for i in range(self.npairs, self.norbs):
            if is_pair_occupied(elec_config, i):
                excite_to.append(i)
        excite_count = len(excite_to)

        assert excite_count in [0, 1]
        i, j = np.unravel_index(index, (self.npairs, self.norbs-self.npairs))
        if excite_count == 0:
            return 0
        elif excite_count == 1:
            # If the coefficient wrt which we are deriving is substituted in
            if i == excite_from[0] and j == excite_to[0]-self.npairs:
                return 1
            # If deriving wrt another element
            else:
                return 0

# vim: set textwidth=90 :
