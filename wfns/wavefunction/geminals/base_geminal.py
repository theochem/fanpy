"""Base class for Geminal wavefunction."""
from __future__ import absolute_import, division, print_function
import abc
import numpy as np
import functools
from wfns.backend import slater
from wfns.backend import math_tools
from wfns.wavefunction.base_wavefunction import BaseWavefunction

__all__ = []


class BaseGeminal(BaseWavefunction):
    r"""Base Geminal Wavefunctions.

    A Geminal is a two-electron wavefunction.

    .. math::
        G^\dagger_p = \sum_{pq} C_{pq} a^\dagger_p a^\dagger_q

    All Geminal wavefunctions can be expresed as an antisymmeterized product of geminals:

    .. math::
        \ket{\Psi} = \prod_{p=1}^P G^\dagger_p \ket{\theta}

    These wavefunctions can be re-expressed in terms of orbital pairs (i.e.
    :math:`a^\dagger_p a^\dagger_q`), where the overlap of the wavefunction with a Slater
    determinant is the sum over all possible combinations of orbital pairs that construct the
    Slater determinant (each set of orbital pairs must be disjoint to follow Slater-Condon rule).
    These combinations are equivalent to the perfect matchings (disjoint and exhaustive pairing) of
    the orbitals of a Slater determinant.
    Different flavors of geminals can allow a different set of perfect matchings for a given Slater
    determinant.
    The method `generate_possible_orbpairs` yields a perfect matching for a given Slater
    determinant.
    The symmetery of electron pair interchange is captured through the evaluation of a permanent.
    Different approximations of the permanent can be implemented using the method
    `compute_permanent`.

    Alternatively, the sum over the different perfect matchings and the permanent evaluation can be
    merged to construct a different combinatorial sum (such as Pffafian).
    To implement these wavefunctions, the method `get_overlap` should be changed to use this sum and
    to ignore `generate_possible_orbpairs` and `compute_permanent`.
    These methods are only called in `get_overlap` so there should be no issue.
    If you'd like, you can always raise NotImplementedError.

    Attributes
    ----------
    nelec : int
        Number of electrons
    nspin : int
        Number of spin orbitals (alpha and beta)
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction
    params : np.ndarray
        Parameters of the wavefunction
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j
        to the column index of the geminal coefficient matrix
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j

    Properties
    ----------
    npairs : int
        Number of electorn pairs
    nspatial : int
        Number of spatial orbitals
    ngem : int
        Number of geminals
    spin : float, None
        Spin of the wavefunction
        :math:`\frac{1}{2}(N_\alpha - N_\beta)` (Note that spin can be negative)
        None means that all spins are allowed
    seniority : int, None
        Seniority (number of unpaired electrons) of the wavefunction
        None means that all seniority is allowed
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters
    template_params : np.ndarray
        Template for the initial guess of geminal coefficient matrix
        Depends on the attributes given

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_nspin(self, nspin)
        Assigns the number of spin orbitals
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_params(self, params)
        Assigns the parameters of the wavefunction
    assign_ref_sds(self, ref_sds=None)
        Assigns the reference Slater determinants from which the initial guess, energy, and norm are
        calculated
        Default is the first Slater determinant of projection space
    assign_orbpairs(self, orbpairs=None)
        Assigns the orbital pairs that will be used to construct geminals
    compute_permanent(self, orbpairs, deriv_row_col=None)
        Compute the permanent that corresponds to the given orbital pairs
    get_overlap(self, sd, deriv_ind=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization

    Abstract Method
    ---------------
    generate_possible_orbpairs(self, occ_indices)
        Yields the possible orbital pairs that can construct the given Slater determinant.
    """
    def __init__(self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None,
                 params=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons
        nspin : int
            Number of spin orbitals
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type
            Default is `np.float64`
        ngem : int, None
            Number of geminals
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal
        params : np.ndarray
            Geminal coefficient matrix
        """
        super().__init__(nelec, nspin, dtype=dtype)
        self.assign_ngem(ngem=ngem)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_params(params=params)

    @property
    def spin(self):
        """Spin of geminal wavefunction."""
        return None

    @property
    def seniority(self):
        """Seniority of geminal wavefunction."""
        return None

    @property
    def npair(self):
        """Return number of electron pairs"""
        return self.nelec//2

    @property
    def norbpair(self):
        """Return number of orbital pairs used to construct the geminals"""
        return len(self.dict_ind_orbpair)

    @property
    def template_params(self):
        """Return the template of the parameters in a Geminal wavefunction.

        Uses the spatial orbitals (alpha beta pair) of HF ground state as reference.

        Returns
        -------
        np.ndarray

        Note
        ----
        Need `nelec`, `norbpair` (i.e. `dict_ind_orbpair`), and `dtype`
        """
        params = np.zeros((self.ngem, self.norbpair), dtype=self.dtype)
        for i in range(self.ngem):
            try:
                col_ind = self.dict_orbpair_ind[(i, i+self.nspatial)]
            except KeyError as error:
                raise ValueError('Given orbital pairs do not contain the default orbitals pairs '
                                 'used in the template. Please give a different set of orbital '
                                 'pairs or provide the paramaters') from error
            params[i, col_ind] += 1
        return params

    def assign_nelec(self, nelec):
        """Set the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons

        Raises
        ------
        TypeError
            If number of electrons is not an integer or long
        ValueError
            If number of electrons is not a positive number
        NotImplementedError
            If number of electrons is odd
        """
        super().assign_nelec(nelec)
        if self.nelec % 2 != 0:
            raise ValueError('Odd number of electrons is not supported')

    def assign_ngem(self, ngem=None):
        """Set the number of geminals.

        Parameters
        ----------
        ngem : int, None
            Number of geminals

        Raises
        ------
        TypeError
            If number of geminals is not an integer
        ValueError
            If number of geminals is less than the number of electron pairs

        Note
        ----
        Needs to have `npair` defined (i.e. `nelec` must be defined)
        """
        if ngem is None:
            ngem = self.npair
        if not isinstance(ngem, int):
            raise TypeError('`ngem` must be an integer.')
        elif ngem < self.npair:
            raise ValueError('`ngem` must be greater than the number of electron pairs.')
        self.ngem = ngem

    def assign_orbpairs(self, orbpairs=None):
        """Set the orbital pairs that will be used to construct the geminals.

        Parameters
        ----------
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal
            Default is all possible orbital pairs

        Raises
        ------
        TypeError
            If `orbpairs` is not an iterable
            If an orbital pair is not given as a list or a tuple
            If an orbital pair does not contain exactly two elements
            If an orbital index is not an integer
        ValueError
            If an orbital pair has the same integer
            If an orbital pair occurs more than once

        Note
        ----
        Must have `nspin` defined for the default option
        """
        # FIXME: terrible memory usage
        if orbpairs is None:
            orbpairs = tuple((i, j) for i in range(self.nspin) for j in range(i+1, self.nspin))

        if not hasattr(orbpairs, '__iter__'):
            raise TypeError('`orbpairs` must iterable.')

        dict_orbpair_ind = {}
        for i, orbpair in enumerate(orbpairs):
            if not isinstance(orbpair, (list, tuple)):
                raise TypeError('Each orbital pair must be a list or a tuple')
            elif len(orbpair) != 2:
                raise TypeError('Each orbital pair must contain two elements')
            elif not (isinstance(orbpair[0], int) and isinstance(orbpair[1], int)):
                raise TypeError('Each orbital index must be given as an integer')
            elif orbpair[0] == orbpair[1]:
                raise ValueError('Orbital pair of the same orbital is invalid')

            orbpair = tuple(orbpair)
            if orbpair[0] > orbpair[1]:
                orbpair = orbpair[::-1]
            if orbpair in dict_orbpair_ind:
                raise ValueError('The given orbital pairs have multiple entries of {0}'
                                 ''.format(orbpair))
            else:
                dict_orbpair_ind[orbpair] = i

        self.dict_orbpair_ind = dict_orbpair_ind
        self.dict_ind_orbpair = {i: orbpair for orbpair, i in dict_orbpair_ind.items()}

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the geminal wavefunction.

        Parameters
        ----------
        params : np.ndarray, BaseGeminal, None
            Parameters of the wavefunction
            If BaseGeminal instance is given, then the parameter of this instance are used.
        add_noise : bool
            Flag to add noise to the given parameters

        Raises
        ------
        TypeError
            If `params` is not a numpy array
            If `params` does not have data type of `float`, `complex`, `np.float64` and
            `np.complex128`
            If `params` has complex data type and wavefunction has float data type
        ValueError
            If `params` does not have the same shape as the template_params
            If given BaseGeminal instance does not have the same number of electrons as self
            If given BaseGeminal instance does not have the same number of spin orbitals as self
            If given BaseGeminal instance does not have the same number of geminals as self

        Note
        ----
        Depends on dtype, template_params, and nparams
        """
        # FIXME: self.__class__ doesn't work if the method is inherited
        if isinstance(params, BaseGeminal):
            other = params
            if self.nelec != other.nelec:
                raise ValueError('The number of electrons in the two wavefunctions must be the '
                                 'same.')
            if self.nspin != other.nspin:
                raise ValueError('The number of spin orbitals in the two wavefunctions must be the '
                                 'same.')
            if self.ngem != other.ngem:
                raise ValueError('The number of geminals in the two wavefunctions must be the '
                                 'same.')
            params = np.zeros(self.params_shape, dtype=self.dtype)
            for ind, orbpair in other.dict_ind_orbpair.items():
                if orbpair in self.dict_orbpair_ind:
                    params[:, self.dict_orbpair_ind[orbpair]] = other.params[:, ind]
                else:
                    print('The orbital pair of the given wavefunction, {0}, is not possible in the '
                          'current wavefunction. Parameters corresponding to this orbital pair will'
                          ' be ignored.')
        super().assign_params(params=params, add_noise=add_noise)

    def compute_permanent(self, col_inds, row_inds=None, deriv_row_col=None):
        """Compute the permanent that corresponds to the given orbital pairs

        Parameters
        ----------
        col_inds : np.ndarray
            Indices of the columns of geminal coefficient matrices that will be used.
        row_inds : np.ndarray
            Indices of the rows of geminal coefficient matrices that will be used.
        deriv : 2-tuple of int, None
            Row and column indices of the element with respect to which the permanent is derivatized
            Default is no derivatization

        Returns
        -------
        permanent :float
        """
        if row_inds is None:
            row_inds = np.arange(self.ngem)
        else:
            row_inds = np.array(row_inds)
        col_inds = np.array(col_inds)
        # select function that evaluates the permanent
        if col_inds.size <= 3 >= row_inds.size:
            permanent = math_tools.permanent_ryser
        else:
            permanent = math_tools.permanent_combinatoric

        if deriv_row_col is None:
            return permanent(self.params[row_inds, :][:, col_inds])
        else:
            # cut out rows and columns that corresponds to the element with which the permanent is
            # derivatized
            row_inds_trunc = row_inds[row_inds != deriv_row_col[0]]
            col_inds_trunc = col_inds[col_inds != deriv_row_col[1]]
            if row_inds_trunc.size == row_inds.size or col_inds_trunc.size == col_inds.size:
                return 0.0
            elif row_inds_trunc.size == col_inds_trunc.size == 0:
                return 1.0
            else:
                return permanent(self.params[row_inds_trunc, :][:, col_inds_trunc])

    def get_overlap(self, sd, deriv=None):
        r"""Compute the overlap between the geminal wavefunction and a Slater determinant.

        The results are cached in self._cache_fns.

        .. math::
            \big| \Psi \big>
            &= \prod_{p=1}^{N_{gem}} \sum_{pq} C_{pq} a^\dagger_p a^\dagger_q \big| \theta \big>\\
            &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} |C(\mathbf{m})|^+
            \big| \mathbf{m} \big>

        where :math:`N_{gem}` is the number of geminals, :math:`\mathbf{m}` is a Slater determinant.

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant as a bitstring
        deriv : None, int
            Index of the paramater with respect to which the overlap is derivatized
            Default is no derivatization

        Returns
        -------
        overlap : float

        Raises
        ------
        TypeError
            If Slater determinant is not gmpy2.mpz object.

        Note
        ----
        Bit of performance is lost in exchange for generalizability. Hopefully it is still readable.
        """
        if not slater.is_internal_sd(sd):
            sd = slater.internal_sd(sd)

        # if no derivatization
        if deriv is None:
            # if cached function has not been created yet
            if 'overlap' not in self._cache_fns:
                # assign memory allocated to cache
                if self.memory == np.inf:
                    memory = None
                else:
                    memory = int((self.memory - 5*8*self.params.size) / (self.params.size + 1))

                # create function that will be cached
                @functools.lru_cache(maxsize=memory, typed=False)
                def _olp(sd):
                    # FIXME: ugly, repeats code
                    # NOTE: sd is used as the key because it uses less memory
                    # NOTE: Need to recreate occ_indices
                    occ_indices = slater.occ_indices(sd)

                    val = 0.0
                    for orbpairs, sign in self.generate_possible_orbpairs(occ_indices):
                        if len(orbpairs) == 0:
                            continue

                        col_inds = np.array([self.dict_orbpair_ind[orbp] for orbp in orbpairs])
                        val += sign * self.compute_permanent(col_inds)
                    return val

                # store the cached function
                self._cache_fns['overlap'] = _olp
            # if cached function already exists
            else:
                # reload cached function
                _olp = self._cache_fns['overlap']

            return _olp(sd)
        # if derivatization
        elif isinstance(deriv, int):
            if deriv >= self.nparams:
                return 0.0
            # convert parameter index to row and col index
            col_removed = deriv % self.norbpair
            # find orbital pair that corresponds to removed column
            orb_1, orb_2 = self.dict_ind_orbpair[col_removed]

            # if either of these orbitals are not present in the Slater determinant, skip
            if not (slater.occ(sd, orb_1) and slater.occ(sd, orb_2)):
                return 0.0

            # if cached function has not been created yet
            if 'overlap derivative' not in self._cache_fns:
                # assign memory allocated to cache
                if self.memory == np.inf:
                    memory = None
                else:
                    memory = int((self.memory - 5*8*self.params.size)
                                 / (self.params.size + 1) * self.params.size)

                # create function that will be cached
                @functools.lru_cache(maxsize=memory, typed=False)
                def _olp_deriv(sd, deriv):
                    # FIXME: ugly, repeats code
                    # NOTE: sd and deriv is used as the key because it uses less memory
                    # NOTE: Need to recreate occ_indices, row_removed, col_removed
                    occ_indices = slater.occ_indices(sd)
                    row_removed = deriv // self.norbpair
                    col_removed = deriv % self.norbpair

                    val = 0.0
                    for orbpairs, sign in self.generate_possible_orbpairs(occ_indices):
                        # ASSUMES: permanent evaluation is much more expensive than the lookup
                        # NOTE: derivatization with respect to parameters that are not present in
                        #       the sd is already skipped by line 463
                        if len(orbpairs) == 0:
                            continue
                        col_inds = np.array([self.dict_orbpair_ind[orbp] for orbp in orbpairs])
                        val += sign*self.compute_permanent(col_inds,
                                                           deriv_row_col=(row_removed, col_removed))
                    return val

                # store the cached function
                self._cache_fns['overlap derivative'] = _olp_deriv
            # if cached function already exists
            else:
                # reload cached function
                _olp_deriv = self._cache_fns['overlap derivative']

            return _olp_deriv(sd, deriv)

    @abc.abstractmethod
    def generate_possible_orbpairs(self, occ_indices):
        """Yields the possible orbital pairs that can construct the given Slater determinant.

        Parameters
        ----------
        occ_indices : N-tuple of int
            Indices of the orbitals from which the Slater determinant is constructed
            Must be strictly increasing.

        Yields
        ------
        orbpairs : P-tuple of 2-tuple of ints
            Indices of the creation operators (grouped by orbital pairs) that construct the Slater
            determinant.
        sign : int
            Signature of the transpositions required to shuffle the `orbitalpairs` back into the
            original order in `occ_indices`.
        """
        pass
