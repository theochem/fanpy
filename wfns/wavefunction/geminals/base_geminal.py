"""Base class for Geminal wavefunction."""
from __future__ import absolute_import, division, print_function
import numpy as np
from ..base_wavefunction import BaseWavefunction

__all__ = []


class BaseGeminal(BaseWavefunction):
    """Generic Geminal Wavefunctions.

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
    cache : dict of sd to float
        Cache of the overlaps that are calculated for each Slater determinant encountered
    d_cache : dict of gmpy2.mpz to float
        Cache of the derivative of overlaps that are calculated for each Slater determinant and
        derivative index encountered

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
    __init__(self, nelec, one_int, two_int, dtype=None)
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

    Abstract Method
    ---------------
    get_overlap(self, sd, deriv=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    """
    def __init__(self, nelec, nspin, dtype=None, ngem=None, orbpairs=None, params=None):
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
        """ Assigns the parameters of the wavefunction

        Parameters
        ----------
        params : np.ndarray, None
            Parameters of the wavefunction
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

        Note
        ----
        Depends on dtype, template_params, and nparams
        """
        super().assign_params(params=params, add_noise=add_noise)
        self.cache = {}
        self.d_cache = {}
