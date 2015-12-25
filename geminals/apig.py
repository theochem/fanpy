"""
APIG geminal wavefunction class.

"""

from __future__ import absolute_import, division, print_function
from itertools import combinations, permutations
import numpy as np
from scipy.misc import comb
from scipy.optimize import root
from geminals.fancy_ci import FancyCI
from geminals.slater_det import excite_orbs, excite_pairs, is_occupied, is_pair_occupied


class APIG(FancyCI):
    """
    A generalized antisymmetrized product of interacting geminals (APIG) implementation.

    Attributes
    ----------
    npairs : int
        The number of electron pairs in the geminal wavefunction.
    norbs : int
        The number of spatial orbitals in the geminal wavefunction.
    nelec : int
        The number of electrons in the geminal wavefunction (2 * npairs).
    ham : tuple of a 2-index np.ndarray, a 4-index np.ndarray
        The two- and four- electron integral Hamiltonian matrices.
    core_energy : float
        The core energy of the system.
    coeffs : 2-index np.ndarray
        The optimized geminal coefficient matrix.
    _coeffs_optimized : bool
        Whether the geminal coefficient matrix has already been optimized.
    _is_complex : bool
        Whether the geminal coefficient matrix is complex
        Set to True if coeffs is complex
        Forces coefficients to be complex if True
    pspace : iterable of ints.
        An iterable of integers that, in binary representation, describe the Slater
        determinants in the geminal wavefunction's projection space.
    energy : float
        The ground-state energy of the geminal wavefunction.
    energies : dict
        The keys correspond to a type of energy, i.e., "coulomb", or "exchange", and the
        values are floats corresponding to the geminal wavefunction's energy of that type.

    Other attributes (developer stuff)
    ----------------------------------
    _exclude_ground : bool
        Whether to disallow the nonlinear system to be solved during coefficient
        optimization from containing an equation corresponding to the ground state.
        This may be needed if the equation can be reduced to "energy == energy",
        making the Jacobian matrix singular.
    _normalize : bool
        Whether to include an equation for intermediate normalization into the nonlinear
        system to be solved during coefficient optimization.  This may not be necessary
        for some geminal types (e.g. AP1roG), where normalization is always satisfied at
            the optimal coefficients.
    _row_indices : iterable
        An iterable returning the range of rows in the geminal coefficient matrix that
        contain the optimizable parameters.  It *should* always be equivalent to
        range(number_of_rows).
    _col_indices : iterable
        An iterable returning the range of columns in the geminal coefficient matrix that
        contain the optimizable parameters.  This often varies between geminal types.

    Methods
    -------
    __init__
        Initialize the APIG instance.
    solve_coeffs
        Optimize the geminal coefficients.
    _generate_x0
        Generate a guess at the optimal geminal coefficients.
    _construct_coeffs
        Construct a coefficient matrix from a guess `x0`.
    generate_pspace
        Generate an appropriate projection space for the optimization of the geminal
        coefficients.
    permanent
        Compute the permanent of a matrix.
    permanent_derivative
        Compute the partial derivative of the permanent of a matrix wrt one of its
        coefficients.
    overlap
        Compute the overlap of a Slater determinant with the geminal wavefunction.
    compute_energy
        Compute the energy of the geminal wavefunction.
    _double_compute_energy
        Efficient backend for compute_energy() for doubly-occupied Slater determinants.
    _brute_compute_energy
        Most general backend for compute_energy().
    nonlin
        Construct a nonlinear system of equations for the optimization of the geminal
        coefficients.
    nonlin_jac
        Compute the Jacobian of nonlin().

    Notes
    -----
    Slater determinants are expressed by an integer that, in binary form, shows which spin
    orbitals are used to construct it. The position of each "1" from the right is the
    index of the orbital that is included in the Slater determinant. The even positions
    (i.e. 0, 2, 4, ...) are the alpha orbitals, and the odd positions (i.e. 1, 3, 5, ...)
    are the beta orbitals.  E.g., 5=0b00110011 is a Slater determinant constructed with
    the 0th and 2nd spatial orbitals, or the 0th, 1st, 5th, 6th spin orbitals (where the
    spin orbitals are ordered by the alpha beta pairs).  The number of electrons is
    assumed to be even.

    Behaviour for complex inputs
    ----------------------------
    If self._is_complex is True, then all self._coeffs assigned will be turned complex
    If self._coeffs is assigned to a complex array, then self_is_complex will be
    assigned True


    """

    #
    # Class-wide (behaviour-changing) attributes and properties
    #

    _exclude_ground = False
    _normalize = True
    _is_spatial = True

    #
    # Properties
    #

    @property
    def num_params(self):
        """ Number of parameters needed

        """
        return self.npairs*self.norbs

    @property
    def npairs(self):
        """ Number of electron pairs

        """
        return self._npairs

    @npairs.setter
    def npairs(self, value):
        """ Setter for number of electron pairs

        Raises
        ------
        AssertionError
            If value is not a positive integer
            If value is greater than the number of orbitals
        """
        assert isinstance(value, int), \
            "The number of electron pairs must be an integer."
        assert value > 0, \
            "The number of electron pairs must be greater than zero."
        assert value <= self.norbs, \
            "The number of electron pairs must be less or equal to the number of spatial orbitals."
        self._npairs = value

    @property
    def nelec(self):
        """ Number of electrons

        """
        return 2 * self.npairs

    @property
    def params(self):
        """ Parameters for the "function" of the wavefunction
        """
        return self._params

    @params.setter
    def params(self, value):
        """ Sets the parameters for the "function" of the wavefunction

        Raises
        ------
        AssertionError
            If value is not a numpy array
            If value is not a one or two dimensional numpy array
            If value is two dimensional and does not have the right shape
            If value does not have the right number of elements

        """
        assert isinstance(value, np.ndarray),\
            'Parameters must be given as a numpy array'
        assert len(value.shape) in [1, 2],\
            'Unsupported numpy array shape'
        if len(value.shape) == 2:
            assert value.shape == (self.npairs, self.norbs)
            value = value.flatten()
        if value.dtype == 'complex' and value.size*self.offset_complex == self.num_params:
            value = np.hstack((np.real(value), np.imag(value)))
            self._is_complex = True
        else:
            assert value.size == self.num_params
        self._params = value

    @property
    def pspace(self):
        """ Projection space
        """
        return self._pspace

    @pspace.setter
    def pspace(self, value):
        """ Setter for the projection space

        Raises
        ------
        AssertionError
            If given Slater determinant does not contain the same number of electrons
            as the number of electrons
            If given Slater determinant is expressed wrt an orbital that doesn't exist
        """
        assert hasattr(value, '__iter__')
        for phi in value:
            bin_phi = bin(phi)[2:]
            # Pad `bin_phi` with zeroes on the left so that it is of even length
            bin_phi = ("0" * (len(bin_phi) % 2)) + bin_phi
            assert bin_phi.count("1") == self.nelec, \
                ("Slater det. {0} does not contain the same number of occupied"
                 " spin-orbitals as number of electrons.".format(bin_phi))
            # if self.npairs > 2:
            assert bin_phi[0::2] == bin_phi[1::2], "Slater det. {} is unrestricted.".format(bin_phi)
            index_last_spin = len(bin_phi) - 1 - bin_phi.index("1")
            assert index_last_spin < self.norbs*self.offset_spatial, \
                ("Slater det. {0} contains orbitals whose indices exceed the"
                 " specified number of orbitals.".format(bin_phi))
        self._pspace = tuple(value)

    #
    # Special methods
    #

    def __init__(self, npairs, norbs, ham, init_params=None, pspace=None, is_complex=False):
        """
        Initialize the APIG instance.

        Parameters
        ----------
        npairs : int
            Number of electron pairs in system
        norbs : int
            Number of spatial orbitals in system
        ham : tuple of a 2-index np.ndarray, a 4-index np.ndarray
            Two- and four- electron integral Hamiltonian matrices in
            molecular orbital basis.
        init_params : np.ndarray(M,)
            Initial guess for the parameters of the function
        pspace : iterable of ints.
            An iterable of integers that, in binary representation, describe the Slater
            determinants in the wavefunction's projection space.
        is_complex : bool
            Flag for complex coefficient matrix
        """
        # Flags
        self._is_complex = is_complex
        # Initialize private variables
        self._npairs = npairs
        self._norbs = norbs
        self._ham = None
        self._params = None
        self._pspace = None
        # Assign attributes their values using their setters
        self.npairs = npairs
        self.norbs = norbs
        self.ham = ham
        self.params = init_params if init_params is not None else self._generate_init()
        self.pspace = pspace if pspace is not None else self.generate_pspace()

    def generate_pspace(self):
        """
        Generate an appropriate projection space.

        Notes
        -----
        We need to have an `npairs * norbs`-dimensional projection space.  First is the
        ground state HF slater determinant (using the first npair spatial orbitals) [1].
        Then, all single pair excitations from any occupieds to any virtuals (ordered HOMO
        to lowest energy for occupieds and LUMO to highest energy for virtuals).  Then,
        all double excitations of the appropriate number of HOMOs to appropriate number of
        LUMOs.  The appropriate number is the smallest number that would generate the
        remaining slater determinants.  This number will maximally be the number of
        occupied spatial orbitals.  Then all triple excitations of the appropriate number
        of HOMOs to appropriate number of LUMOs.

        It seems that certain combinations of npairs and norbs is not possible because we
        are assuming that we only use pair excitations. For example, a 2 electron system
        cannot ever have appropriate number of Slater determinants, a 4 electron system
        must have 6 spatial orbitals, a 6 electron system must have 6 spatial orbitals.  a
        8 electron system must have 7 spatial orbitals.

        Returns
        -------
        pspace : tuple of ints
            Iterable of integers that, in binary, describes which spin orbitals are used
            to build the Slater determinant.

        Raises
        ------
        AssertionError
            If the method is unable to generate an appropriate projection space.

        """

        # Will there be complex coefficients?
        complex_mod = 2 if self._is_complex else 1

        # Find the minimum rank of a non-underdetermined system
        min_rank = complex_mod * (self.npairs * self.norbs + int(self._exclude_ground))

        # Find the ground state and occupied/virtual indices
        ground = self.ground_sd
        pspace = [ground]
        ind_occ = [i for i in range(self.norbs) if is_pair_occupied(ground, i)]
        ind_occ.reverse()  # reverse ordering to put HOMOs first
        ind_virt = [i for i in range(self.norbs) if not is_pair_occupied(ground, i)]

        # Add pair excitations
        ind_excited = 1
        while len(pspace) < min_rank and ind_excited <= len(ind_occ):
            # Determine the smallest usable set of frontier (HOMO/LUMO) orbitals
            for i in range(2, len(ind_occ) + 1):
                if comb(i, 2, exact=True) ** 2 >= min_rank - len(pspace):
                    nfrontier = i
                    break
            else:
                nfrontier = max(len(ind_occ), len(ind_virt)) + 1
            # Add excitations from all possible combinations of nfrontier HOMOs...
            for occs in combinations(ind_occ[:nfrontier], ind_excited):
                # ...to all possible combinations of nfrontier LUMOs
                for virts in combinations(ind_virt[:nfrontier], ind_excited):
                    orbs = list(occs) + list(virts)
                    pspace.append(excite_pairs(ground, *orbs))

            ind_excited += 1

        # Add single excitations (necessary for systems with less than two electron pairs
        # and probably for some minimal basis set cases) by exciting betas to alphas
        if self.npairs <= complex_mod * 2:
            for i in ind_occ:
                for j in ind_virt:
                    pspace.append(excite_orbs(ground, i * 2 + 1, j * 2))

        # Sanity checks
        assert len(pspace) == len(set(pspace)), \
            "generate_pspace() is making duplicate Slater determinants.  This shouldn't happen!"
        assert len(pspace) >= min_rank, \
            "generate_pspace() was unable to generate enough Slater determinants."

        # Return projection space truncated to `min_rank`
        return tuple(pspace[:min_rank])

    def overlap(self, phi, params, differentiate_wrt=None):
        """
        Calculate the overlap between a Slater determinant and the geminal wavefunction.

        Parameters
        ----------
        phi : int
            An integer that, in binary representation, describes the Slater determinant.
        params : np.ndarray(K,), optional.
            Coefficient matrix for the wavefunction.
        differentiate_wrt : {None, int}
            Index wrt the overlap will be differentiated
            If unspecified, overlap is not differentiated

        Returns
        -------
        overlap : float

        Raises
        ------
        AssertionError
            If differentiate_wrt is not an integer

        """

        # If the Slater determinant is bad
        if phi is None:
            return 0
        # If the Slater det. and wavefuntion have a different number of electrons
        elif bin(phi).count("1") != self.nelec:
            return 0
        # If the Slater determinant has orbitals that are not doubly occupied
        elif any(is_occupied(phi, i*2) != is_occupied(phi, i*2+1) for i in range(self.norbs)):
            return 0
        elif differentiate_wrt is None:
            return self.the_function(params, phi)
        else:
            assert isinstance(differentiate_wrt, int)
            return self.differentiate_the_function(params, phi, differentiate_wrt)

    def compute_energy(self, phi, params, differentiate_wrt=None):
        """
        Calculate the energy of the geminal wavefunction projected onto a Slater
        determinant.

        Parameters
        ----------
        phi : int
            An integer that, in binary representation, describes the Slater determinant.
            Defaults to the ground state.
        params : np.ndarray(M,)
            Set of parameters that describe the function
        differentiate_wrt : {None, int}
            Index wrt the overlap will be differentiated
            If unspecified, overlap is not differentiated

        Returns
        -------
        energy : tuple of float
            Tuple of the one electron, coulomb, and exchange energies

        Raises
        ------
        AssertionError
            If `coeffs` is not specified and self.coeffs is None

        Notes
        -----
        Makes no assumption about the Slater determinant structure.
        Assumes that one- and two-electron integrals are Hermitian.

        """
        # Do the fast pairwise calculation if possible
        bin_phi = bin(phi)[2:]
        # Pad `bin_phi` with zeroes on the left so that it is of even length
        bin_phi = ("0" * (len(bin_phi) % 2)) + bin_phi
        alpha_occ = bin_phi[0::2]
        beta_occ = bin_phi[1::2]
        if alpha_occ == beta_occ:
            one, two = self.ham
            one_electron = 0.0
            coulomb0 = 0.0
            coulomb1 = 0.0
            exchange = 0.0
            diff_index = differentiate_wrt
            olp = self.overlap(phi, params, diff_index)

            for i in range(self.norbs):
                if is_pair_occupied(phi, i):
                    one_electron += 2 * one[i, i]
                    coulomb0 += two[i, i, i, i]

                    for j in range(i + 1, self.norbs):
                        if is_pair_occupied(phi, j):
                            coulomb0 += 4 * two[i, j, i, j]
                            exchange -= 2 * two[i, j, j, i]
                    for a in range(self.norbs):
                        if not is_pair_occupied(phi, a):
                            excitation = excite_pairs(phi, i, a)
                            coulomb1 += two[i, i, a, a] * self.overlap(excitation, params, diff_index)
            return one_electron * olp, coulomb0 * olp + coulomb1, exchange * olp
        else:
            return super(self.__class__, self).compute_energy(phi, params, i)

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
        indices = [i for i in range(self.norbs) if is_pair_occupied(elec_config, i)]
        matrix = params.reshape(self.npairs, self.norbs)[:, indices]
        return self.__class__.permanent(matrix)

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
        i, j = np.unravel_index(index, (self.npairs, self.norbs))
        if is_pair_occupied(elec_config, j):
            matrix = params.reshape(self.npairs, self.norbs)
            matrix = np.delete(matrix, i, axis=0)
            indices = [k for k in range(self.norbs) if is_pair_occupied(elec_config, k) and k != j]
            matrix = matrix[:, indices]
            return self.__class__.permanent(matrix)*params[index]
        else:
            return 0

    @staticmethod
    def permanent(matrix):
        """
        Combinatorically compute the permanent of a matrix.

        Parameters
        ----------
        matrix : 2-index np.ndarray
            A 2-dimensional, square NumPy array.

        Returns
        -------
        permanent : float

        Raises
        ------
        AssertionError
            If the matrix is not square.

        """
        assert matrix.shape[0] is matrix.shape[1], \
            "The permanent of a non-square matrix cannot be computed."
        permanent = 0
        row_indices = range(matrix.shape[0])
        for col_indices in permutations(row_indices):
            permanent += np.product(matrix[row_indices, col_indices])
        return permanent


# vim: set textwidth=90 :
