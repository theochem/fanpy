""" Base class for geminals
"""
from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractproperty
from ..proj_wavefunction import ProjectedWavefunction

class Geminal(ProjectedWavefunction):
    """ Projected Wavefunction class

    Contains the necessary information to solve the wavefunction

    Class Variables
    ---------------
    _nconstraints : int
        Number of constraints
    _seniority : int, None
        Seniority of the wavefunction
        None means that all seniority is allowed
    _spin : float, None
        Spin of the wavefunction
        :math:`\frac{1}{2}(N_\alpha - N_\beta)` (Note that spin can be negative)
        None means that all spins are allowed

    Attributes
    ----------
    nelec : int
        Number of electrons
    one_int : 1- or 2-tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        1-tuple for spatial (restricted) and generalized orbitals
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
    two_int : 1- or 3-tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        In physicist's notation
        1-tuple for spatial (restricted) and generalized orbitals
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components)
    dtype : {np.float64, np.complex128}
        Numpy data type
    nuc_nuc : float
        Nuclear-nuclear repulsion energy
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    pspace : tuple of gmpy2.mpz
        Slater determinants onto which the wavefunction is projected
    ref_sds : tuple of gmpy2.mpz
        Slater determinants that will be used as a reference for the wavefunction (e.g. for
        initial guess, energy calculation, normalization, etc)
    params : np.ndarray
        Parameters of the wavefunction (including energy)
    cache : dict of sd to float
        Cache of the overlaps that are calculated for each Slater determinant encountered
    d_cache : dict of gmpy2.mpz to float
        Cache of the derivative of overlaps that are calculated for each Slater determinant and
        derivative index encountered
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j
        to the column index of the geminal coefficient matrix
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j


    Properties
    ----------
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals
    nparams : int
        Number of parameters
    nproj : int
        Number of Slater determinants
    npair : int
        Number of electron pairs
    ngem : int
        Number of geminals

    Method
    ------
    __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_integrals(self, one_int, two_int, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
    assign_pspace(self, pspace=None)
        Assigns the tuple of Slater determinants onto which the wavefunction is projected
        Default uses `generate_pspace`
    generate_pspace(self)
        Generates the default tuple of Slater determinants with the appropriate spin and seniority
        in increasing excitation order.
        The number of Slater determinants is truncated by the number of parameters plus a magic
        number (42)
    assign_ref_sds(self, ref_sds=None)
        Assigns the reference Slater determinants from which the initial guess, energy, and norm are
        calculated
        Default is the first Slater determinant of projection space
    assign_params(self, params=None)
        Assigns the parameters of the wavefunction (including energy)
        Default contains coefficients from abstract property, `template_coeffs`, and the energy of
        the reference Slater determinants with the coefficients from `template_coeffs`
    assign_orbpairs(self, orbpairs=None)
        Assigns the orbital pairs that will be used to construct geminals
    get_overlap(self, sd, deriv=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    compute_norm(self, ref_sds=None, deriv=None)
        Calculates the norm from given Slater determinants
        Default `ref_sds` is the `ref_sds` given by the intialization
        Default is no derivatization
    compute_hamitonian(self, slater_d, deriv=None)
        Calculates the expectation value of the Hamiltonian projected onto the given Slater
        determinant, `slater_d`
        By default no derivatization
    compute_energy(self, include_nuc=False, ref_sds=None, deriv=None)
        Calculates the energy projected onto given Slater determinants
        Default `ref_sds` is the `ref_sds` given by the intialization
        By default, electronic energy, no derivatization
    objective(self, x, weigh_constraints=True)
        Objective of the equations that will need to be solved (to solve the Projected Schrodinger
        equation)
    jacobian(self, x, weigh_constraints=True)
        Jacobian of the objective

    Abstract Property
    -----------------
    template_coeffs : np.ndarray
        Initial guess coefficient matrix for the given reference Slater determinants
    template_orbpairs : tuple
        List of orbital pairs that will be used to construct the geminals

    Abstract Method
    ---------------
    compute_overlap(self, sd, deriv=None)
        Calculates the overlap between the wavefunction and a Slater determinant
        Function in FancyCI
    """
    __metaclass__ = ABCMeta
    def __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None, pspace=None,
                 ref_sds=None, params=None, ngem=None, orbpairs=None):
        """ Initializes a wavefunction

        Parameters
        ----------
        nelec : int
            Number of electrons

        one_int : np.ndarray(K,K), 1- or 2-tuple np.ndarray(K,K)
            One electron integrals
            For spatial and generalized orbitals, np.ndarray or 1-tuple of np.ndarray
            For unretricted spin orbitals, 2-tuple of np.ndarray

        two_int : np.ndarray(K,K,K,K), 1- or 3-tuple np.ndarray(K,K,K,K)
            For spatial and generalized orbitals, np.ndarray or 1-tuple of np.ndarray
            For unrestricted orbitals, 3-tuple of np.ndarray

        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type
            Default is `np.float64`

        nuc_nuc : {float, None}
            Nuclear nuclear repulsion value
            Default is `0.0`

        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbital used in obtaining the one-electron and two-electron integrals
            Default is `'restricted'`

        pspace : list/tuple of int/long/gmpy2.mpz, None
            Slater determinants onto which the wavefunction is projected
            Default uses `generate_pspace`

        ref_sds : int/long/gmpy2.mpz, list/tuple of int/long/gmpy2.mpz, None
            Slater determinants that will be used as a reference for the wavefunction (e.g. for
            initial guess, energy calculation, normalization, etc)
            Default uses first Slater determinant of `pspace`

        params : np.ndarray, None
            Parameters of the wavefunction (including energy)
            Default uses `template_coeffs` and energy of the reference Slater determinants

        ngem : int, None
            Number of geminals
        """
        # NOTE: need to use Wavefunction.__init__ because ProjectedWavefunction.__init__ has some
        #       problems with the dependency ordering
        #       need nelec -> ngem -> template_coeffs -> assign_pspace
        #       but ProjectedWavefunction.__init__ does
        #       nelec -> assign_pspace
        super(ProjectedWavefunction, self).__init__(nelec, one_int, two_int, dtype=dtype,
                                                    nuc_nuc=nuc_nuc, orbtype=orbtype)
        self.cache = {}
        self.d_cache = {}
        self.assign_ngem(ngem=ngem)
        self.assign_pspace(pspace=pspace)
        self.assign_ref_sds(ref_sds=ref_sds)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_params(params=params)


    @property
    def npair(self):
        """ Number of electron pairs
        """
        return self.nelec//2


    def assign_nelec(self, nelec):
        """ Sets the number of electrons

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
        """
        # NOTE: use super?
        if not isinstance(nelec, (int, long)):
            raise TypeError('`nelec` must be of type {0} or {1}'.format(int, long))
        elif nelec <= 0:
            raise ValueError('`nelec` must be a positive integer')
        elif nelec % 2 != 0:
            raise ValueError('`nelec` must be an even number')
        self.nelec = nelec


    def assign_ngem(self, ngem=None):
        """ Assigns the number of geminals

        Parameters
        ----------
        ngem : int, None
            Number of geminals

        Raises
        ------
        TypeError
            If number of geminals is not an integer or long
        ValueError
            If number of geminals is less than the number of electron pairs
        """
        if ngem is None:
            ngem = int(self.npair)
        if not isinstance(ngem, (int, long)):
            raise TypeError('`ngem` must be of type {0} or {1}'.format(int, long))
        elif ngem < self.npair:
            raise ValueError('`ngem` must be greater than the number of electron pairs')
        self.ngem = ngem


    def assign_orbpairs(self, orbpairs=None):
        """ Assigns the orbital pairs that will be used to construct geminals

        Parameters
        ----------
        orbpairs : tuple/list of 2-tuple of ints
            Each 2-tuple is an orbital pair that is allowed to contribute to geminals

        Raises
        ------
        TypeError
            If `orbpairs` is not a tuple/list of 2-tuples
        ValueError
            If an orbital pair has the same integer
        """
        if orbpairs is None:
            orbpairs = self.template_orbpairs

        if (not isinstance(orbpairs, (tuple, list)) or
                not all(isinstance(orbpair, tuple) for orbpair in orbpairs) or
                not all(len(orbpair) == 2 for orbpair in orbpairs)):
            raise TypeError('`orbpairs` must be given as a tuple/list of 2-tuple')
        if not all(isinstance(i[0], int) and isinstance(i[1], int) for i in orbpairs):
            raise ValueError('Each orbital pair must be given as a 2-tuple of integers')
        if any(orbpair[0] == orbpair[1] for orbpair in orbpairs):
            raise ValueError('Found an orbital pair that consists of only one orbital')

        # sort
        orbpairs = sorted((i, j) if i < j else (j, i) for (i, j) in orbpairs)

        self.dict_orbpair_ind = {orbpair:i for i, orbpair in enumerate(orbpairs)}
        self.dict_ind_orbpair = {i:orbpair for i, orbpair in enumerate(orbpairs)}


    #####################
    # Abstract Property #
    #####################
    @abstractproperty
    def template_orbpairs(self):
        """ List of orbital pairs that will be used to construct the geminals
        """
        pass
