"""Base class for Geminal wavefunctions."""
import abc
import numpy as np
import functools
from wfns.backend import slater
from wfns.backend import math_tools
from wfns.wfn.base import BaseWavefunction


# FIXME: define some function to get the column indices of the parameters from the orbital pairs
#        so that it can be overwritten with something faster than dictionary lookup (the default can
#        be dictionary lookup)
class BaseGeminal(BaseWavefunction):
    r"""Base Geminal Wavefunction.

    A Geminal is a two-electron wavefunction.

    .. math::

        G^\dagger_p = \sum_{ij} C_{pij} a^\dagger_i a^\dagger_j

    All Geminal wavefunctions can be expresed as an antisymmeterized product of geminals:

    .. math::

        \left| \Psi \right> = \prod_{p=1}^P G^\dagger_p \left| \theta \right>

    These wavefunctions can be re-expressed in terms of orbital pairs (i.e. :math:`a^\dagger_p
    a^\dagger_q`), where the overlap of the wavefunction with a Slater determinant is the sum over
    all possible combinations of orbital pairs that construct the Slater determinant (each set of
    orbital pairs must be disjoint to follow Slater-Condon rule). These combinations are equivalent
    to the perfect matchings (disjoint and exhaustive pairing) of the orbitals of a Slater
    determinant. Different flavors of geminals can allow a different set of perfect matchings for a
    given Slater determinant. The method `generate_possible_orbpairs` yields a perfect matching for
    a given Slater determinant. The symmetery of electron pair interchange is captured through the
    evaluation of a permanent. Different approximations of the permanent can be implemented using
    the method `compute_permanent`.

    Alternatively, the sum over the different perfect matchings and the permanent evaluation can be
    merged to construct a different combinatorial sum (such as Pffafian). To implement these
    wavefunctions, the method `get_overlap` should be changed to use this sum and to ignore
    `generate_possible_orbpairs` and `compute_permanent`. These methods are only called in
    `get_overlap` so there should be no issue. If you'd like, you can always raise
    NotImplementedError.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction.
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j to the
        column index of the geminal coefficient matrix.
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    param_shape : tuple of int
        Shape of the parameters.
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    npair : int
        Number of electron pairs.
    norbpair : int
        Number of orbital pairs used to construct the geminals.
    template_params : np.ndarray
        Default parameters of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_dtype(self, dtype)
        Assign the data type of the parameters.
    assign_memory(self, memory=None)
        Assign memory available for the wavefunction.
    assign_ngem(self, ngem=None)
        Assign the number of geminals.
    assign_orbpairs(self, orbpairs=None)
        Assign the orbital pairs that will be used to construct the geminals.
    assign_params(self, params=None, add_noise=False)
        Assign the parameters of the geminal wavefunction.
    get_col_ind(self, orbpair)
        Get the column index that corresponds to the given orbital pair.
    get_orbpair(self, col_ind)
        Get the orbital pair that corresponds to the given column index.
    compute_permanent(self, col_inds, row_inds=None, deriv=None)
        Compute the permanent of the matrix that corresponds to the given orbital pairs.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.

    Abstract Methods
    ----------------
    generate_possible_orbpairs(self, occ_indices)
        Yield the possible orbital pairs that can construct the given Slater determinant.

    """
    def __init__(self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None,
                 params=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type.
            Default is `np.float64`.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).
        ngem : {int, None}
            Number of geminals.
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal.
        params : np.ndarray
            Geminal coefficient matrix.

        """
        super().__init__(nelec, nspin, dtype=dtype)
        self.assign_ngem(ngem=ngem)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_params(params=params)
        self.load_cache()

    @property
    def npair(self):
        """Return number of electron pairs.

        Returns
        -------
        npair : int
            Number of electron pairs.

        """
        return self.nelec//2

    @property
    def norbpair(self):
        """Return the number of orbital pairs used to construct the geminals.

        Returns
        -------
        norbpair : int
            Number of orbital pairs used to construct the geminals.

        """
        return len(self.dict_ind_orbpair)

    @property
    def template_params(self):
        """Return the template of the parameters of the given wavefunction.

        Uses the spatial orbitals (alpha-beta spin orbital pairs) of HF ground state as reference.

        Returns
        -------
        template_params : np.ndarray(ngem, norbpair)
            Default parameters of the geminal wavefunction.

        Notes
        -----
        Need `nelec`, `norbpair` (i.e. `dict_ind_orbpair`), and `dtype`

        """
        params = np.zeros((self.ngem, self.norbpair), dtype=self.dtype)
        for i in range(self.ngem):
            try:
                col_ind = self.get_col_ind((i, i+self.nspatial))
            except KeyError as error:
                raise ValueError('Given orbital pairs do not contain the default orbitals pairs '
                                 'used in the template. Please give a different set of orbital '
                                 'pairs or provide the paramaters') from error
            params[i, col_ind] += 1
        return params

    def assign_nelec(self, nelec):
        """Assign the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons.

        Raises
        ------
        TypeError
            If number of electrons is not an integer.
        ValueError
            If number of electrons is not a positive number.
            If number of electrons is odd.

        """
        super().assign_nelec(nelec)
        if self.nelec % 2 != 0:
            raise ValueError('Odd number of electrons is not supported')

    def assign_ngem(self, ngem=None):
        """Assign the number of geminals.

        Parameters
        ----------
        ngem : {int, None}
            Number of geminals.
            Default is the number of electron pairs.

        Raises
        ------
        TypeError
            If number of geminals is not an integer.
        ValueError
            If number of geminals is less than the number of electron pairs.

        Notes
        -----
        Needs to have `npair` defined (i.e. `nelec` must be defined).

        """
        if ngem is None:
            ngem = self.npair
        if not isinstance(ngem, int):
            raise TypeError('`ngem` must be an integer.')
        elif ngem < self.npair:
            raise ValueError('`ngem` must be greater than the number of electron pairs.')
        self.ngem = ngem

    def assign_orbpairs(self, orbpairs=None):
        """Assign the orbital pairs that will be used to construct the geminals.

        Parameters
        ----------
        orbpairs : iterable of 2-tuple/list of ints
            Indices of the orbital pairs that will be used to construct each geminal.
            Default is all possible orbital pairs.

        Raises
        ------
        TypeError
            If `orbpairs` is not an iterable.
            If an orbital pair is not given as a list or a tuple.
            If an orbital pair does not contain exactly two elements.
            If an orbital index is not an integer.
        ValueError
            If an orbital pair has the same integer.
            If an orbital pair occurs more than once.

        Notes
        -----
        Must have `nspin` defined for the default option.

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
            # sort orbitals within the pair
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
        params : {np.ndarray, BaseGeminal, None}
            Parameters of the geminal wavefunction.
            If BaseGeminal instance is given, then the parameters of this instance are used.
            Default uses the template parameters.

        Raises
        ------
        ValueError
            If `params` does not have the same shape as the template_params.
            If given BaseGeminal instance does not have the same number of electrons.
            If given BaseGeminal instance does not have the same number of spin orbitals.
            If given BaseGeminal instance does not have the same number of geminals.

        Notes
        -----
        Depends on dtype, template_params, and nparams.

        """
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
                try:
                    params[:, self.get_col_ind(orbpair)] = other.params[:, ind]
                except ValueError:
                    print('The orbital pair of the given wavefunction, {0}, is not possible in the '
                          'current wavefunction. Parameters corresponding to this orbital pair will'
                          ' be ignored.')
        super().assign_params(params=params, add_noise=add_noise)

    def get_col_ind(self, orbpair):
        """Get the column index that corresponds to the given orbital pair.

        Parameters
        ----------
        orbpair : 2-tuple of int
            Indices of the orbital pairs that will be used to construct each geminal.
            Default is all possible orbital pairs.

        Returns
        -------
        col_ind : int
            Column index that corresponds to the given orbital pair.

        Raises
        ------
        ValueError
            If given orbital pair is not valid.

        """
        try:
            return self.dict_orbpair_ind[orbpair]
        except (KeyError, TypeError):
            raise ValueError('Given orbital pair, {0}, is not included in the '
                             'wavefunction.'.format(orbpair))

    def get_orbpair(self, col_ind):
        """Get the orbital pair that corresponds to the given column index.

        Parameters
        ----------
        col_ind : int
            Column index that corresponds to the given orbital pair.

        Returns
        -------
        orbpair : 2-tuple of int
            Indices of the orbital pairs that will be used to construct each geminal.
            Default is all possible orbital pairs.

        Raises
        ------
        ValueError
            If given orbital pair is not valid.

        """
        try:
            return self.dict_ind_orbpair[col_ind]
        except (KeyError, TypeError):
            raise ValueError('Given column index, {0}, is not used in the '
                             'wavefunction'.format(col_ind))

    def compute_permanent(self, col_inds, row_inds=None, deriv=None):
        """Compute the permanent of the matrix that corresponds to the given orbital pairs.

        Parameters
        ----------
        col_inds : np.ndarray
            Indices of the columns of geminal coefficient matrices that will be used.
        row_inds : {np.ndarray, None}
            Indices of the rows of geminal coefficient matrices that will be used.
            Default is all rows.
        deriv : {int, None}
            C-order index (in the flattened array) of the element with with respect to which the
            permanent is derivatized.
            Default is no derivatization.

        Returns
        -------
        permanent : float
            Permanent of the selected submatrix.

        """
        if row_inds is None:
            row_inds = np.arange(self.ngem)
        else:
            row_inds = np.array(row_inds)
        col_inds = np.array(col_inds)
        # select function that evaluates the permanent
        # Ryser algorithm is faster if the number of rows and columns are greater than 3
        if col_inds.size <= 3 >= row_inds.size:
            permanent = math_tools.permanent_ryser
        else:
            permanent = math_tools.permanent_combinatoric

        if deriv is None:
            return permanent(self.params[row_inds, :][:, col_inds])
        else:
            row_removed = deriv // self.norbpair
            col_removed = deriv % self.norbpair
            # cut out rows and columns that corresponds to the element with which the permanent is
            # derivatized
            row_inds_trunc = row_inds[row_inds != row_removed]
            col_inds_trunc = col_inds[col_inds != col_removed]
            if row_inds_trunc.size == row_inds.size or col_inds_trunc.size == col_inds.size:
                return 0.0
            elif row_inds_trunc.size == col_inds_trunc.size == 0:
                return 1.0
            else:
                return permanent(self.params[row_inds_trunc, :][:, col_inds_trunc])

    def load_cache(self):
        """Load the functions whose values will be cached.

        To minimize the cache size, the input is made as small as possible. Therefore, the cached
        function is not a method of an instance (because the instance is an input) and the smallest
        representation of the Slater determinant (an integer) is used as the only input. However,
        the functions must access other properties/methods of the instance, so they are defined
        within this method so that the instance is available within the namespace w/o use of
        `global` or `local`.

        Since the bitstring is used to represent the Slater determinant, they need to be processed,
        which may result in repeated processing depending on when the cached function is accessed.

        It is assumed that the cached functions will not be used to calculate redundant results. All
        simplifications that can be made is assumed to have already been made. For example, it is
        assumed that the overlap derivatized with respect to a parameter that is not associated with
        the given Slater determinant will never need to be evaluated because these conditions are
        caught before calling the cached functions.

        Notes
        -----
        Needs to access `memory` and `params`.

        """
        # assign memory allocated to cache
        if self.memory == np.inf:
            memory = None
        else:
            memory = int((self.memory - 5*8*self.params.size) / (self.params.size + 1))

        # create function that will be cached
        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp(sd):
            """Calculate the overlap with the Slater determinant.

            Parameters
            ----------
            sd : gmpy2.mpz
                Occupation vector of a Slater determinant given as a bitstring.

            Returns
            -------
            olp : {float, complex}
                Overlap of the current instance with the given Slater determinant.

            """
            # NOTE: Need to recreate occ_indices
            occ_indices = slater.occ_indices(sd)

            val = 0.0
            for orbpairs, sign in self.generate_possible_orbpairs(occ_indices):
                if len(orbpairs) == 0:
                    continue

                col_inds = np.array([self.get_col_ind(orbp) for orbp in orbpairs])
                val += sign * self.compute_permanent(col_inds)
            return val

        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp_deriv(sd, deriv):
            """Calculate the derivative of the overlap with the Slater determinant.

            Parameters
            ----------
            sd : gmpy2.mpz
                Occupation vector of a Slater determinant given as a bitstring.
            deriv : int
                Index of the parameter with respect to which the overlap is derivatized.

            Returns
            -------
            olp : {float, complex}
                Derivative of the overlap with respect to the given parameter.

            """
            # NOTE: Need to recreate occ_indices, row_removed, col_removed
            occ_indices = slater.occ_indices(sd)

            val = 0.0
            for orbpairs, sign in self.generate_possible_orbpairs(occ_indices):
                # ASSUMES: permanent evaluation is much more expensive than the lookup
                if len(orbpairs) == 0:
                    continue
                col_inds = np.array([self.get_col_ind(orbp) for orbp in orbpairs])
                val += sign * self.compute_permanent(col_inds, deriv=deriv)
            return val

        # create cache
        if not hasattr(self, '_cache_fns'):
            self._cache_fns = {}

        # store the cached function
        self._cache_fns['overlap'] = _olp
        self._cache_fns['overlap derivative'] = _olp_deriv

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::
            \left| \Psi \right>
            &= \prod_{p=1}^{N_{gem}} \sum_{ij}
               C_{pij} a^\dagger_i a^\dagger_j \left| \theta \right>\\
            &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} |C(\mathbf{m})|^+
            \left| \mathbf{m} \right>

        where :math:`N_{gem}` is the number of geminals, :math:`\mathbf{m}` is a Slater determinant.

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : int
            Index of the parameter to derivatize.
            Default does not derivatize.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        Notes
        -----
        Bit of performance is lost in exchange for generalizability. Hopefully it is still readable.

        """
        sd = slater.internal_sd(sd)

        # if no derivatization
        if deriv is None:
            return self._cache_fns['overlap'](sd)
        # if derivatization
        elif isinstance(deriv, (int, np.int64)):
            if deriv >= self.nparams:
                return 0.0
            # convert parameter index to row and col index
            col_removed = deriv % self.norbpair
            # find orbital pair that corresponds to removed column
            orb_1, orb_2 = self.get_orbpair(col_removed)

            # if either of these orbitals are not present in the Slater determinant, skip
            if not (slater.occ(sd, orb_1) and slater.occ(sd, orb_2)):
                return 0.0

            return self._cache_fns['overlap derivative'](sd, deriv)

    @abc.abstractmethod
    def generate_possible_orbpairs(self, occ_indices):
        """Yield the possible orbital pairs that can construct the given Slater determinant.

        Parameters
        ----------
        occ_indices : N-tuple of int
            Indices of the orbitals from which the Slater determinant is constructed.
            Must be strictly increasing.

        Yields
        ------
        orbpairs : P-tuple of 2-tuple of ints
            Indices of the creation operators (grouped by orbital pairs) that construct the Slater
            determinant.
        sign : int
            Signature of the transpositions required to shuffle the `orbpairs` back into the
            original order in `occ_indices`.

        """
        pass
