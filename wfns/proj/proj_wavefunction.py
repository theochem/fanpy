""" Parent class of projected wavefunctions

This module describes wavefunction that are solved projectively
"""
from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from ..wavefunction import Wavefunction
from ..sd_list import sd_list
from .. import slater
from .proj_hamiltonian import hamiltonian, sen0_hamiltonian


# TODO: move out hamiltonian into a separate module? (include orbital rotation)
# TODO: move out constraints to somewhere else
# TODO: move out pspace, constraints, objective, jacobian to solver? rename ProjectedWavefunction to
#       FancyCIWavefunction?
class ProjectedWavefunction(Wavefunction):
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

    Abstract Method
    ---------------
    compute_overlap(self, sd, deriv=None)
        Calculates the overlap between the wavefunction and a Slater determinant
        Function in FancyCI

    """
    __metaclass__ = ABCMeta
    # Default wavefunction properties
    _nconstraints = 1
    _seniority = None
    _spin = None

    def __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None, pspace=None,
                 ref_sds=None, params=None):
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
        """
        super(ProjectedWavefunction, self).__init__(nelec, one_int, two_int, dtype=dtype,
                                                    nuc_nuc=nuc_nuc, orbtype=orbtype)

        self.cache = {}
        self.d_cache = {}
        self.assign_pspace(pspace=pspace)
        self.assign_ref_sds(ref_sds=ref_sds)
        self.assign_params(params=params)


    ##############
    # Properties #
    ##############
    @property
    def nparams(self):
        """ Number of parameters
        """
        return self.template_coeffs.size + 1


    @property
    def nproj(self):
        """ Number of Slater determinants to project against
        """
        return len(self.pspace)


    ######################
    # Assignment methods #
    ######################
    def generate_pspace(self):
        """ Generates Slater determinants onto which the wavefunction is projected

        Returns
        -------
        pspace : list of gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant as a bitstring
        """
        #FIXME
        magic = 42
        return sd_list(self.nelec, self.nspatial, num_limit=self.nparams+magic, spin=self._spin,
                       seniority=self._seniority)


    def assign_pspace(self, pspace=None):
        """ Sets the Slater determinants on which to project against

        Parameters
        ----------
        pspace : list/tuple of int/long/gmpy2.mpz, None
            List/tuple of Slater determinants
            Default uses `generate_pspace`

        Raises
        ------
        TypeError
            If projection space is not a list or a tuple
            If Slater determinant is in a form that is not supported
        """
        if pspace is None:
            pspace = self.generate_pspace()
        if not isinstance(pspace, (list, tuple)):
            raise TypeError('`pspace` must be given as a list or tuple of Slater determinants')
        pspace = tuple(slater.internal_sd(sd) for sd in pspace)
        # filter
        self.pspace = tuple(sd for sd in pspace if
                            (self._spin in [None, slater.get_spin(sd, self.nspatial)]) and
                            (self._seniority in [None, slater.get_seniority(sd, self.nspatial)]))


    def assign_ref_sds(self, ref_sds=None):
        """ Assigns the reference Slater determinants

        Reference Slater determinants are used to calculate the `energy`, `norm`, and
        `template_coeffs`.

        Parameters
        ----------
        ref_sds : int/long/gmpy2.mpz, list/tuple of ints, None
            Slater determinants that will be used as a reference for the wavefunction (e.g. for
            initial guess, energy calculation, normalization, etc)
            If `int` or `gmpy2.mpz`, then the equivalent Slater determinant (see `wfns.slater`) is
            used as a reference
            If `list` or `tuple` of Slater determinants, then multiple Slater determinants will be
            used as a reference. Note that multiple references require an initial guess
            Default is the first element of the `self.pspace`

        Raises
        ------
        TypeError
            If Slater determinants in a list or tuple are not compatible with the format used
            internally
            If Slater determinants are given in a form that is not int/long/gmpy2.mpz, list/tuple of
            ints or None
        """
        # FIXME: repeated code (from assign_ref_sds)
        if ref_sds is None:
            self.ref_sds = (self.pspace[0], )
        elif isinstance(ref_sds, (int, long)) or slater.is_internal_sd(ref_sds):
            self.ref_sds = (slater.internal_sd(ref_sds), )
        elif isinstance(ref_sds, (list, tuple)):
            self.ref_sds = tuple(slater.internal_sd(i) for i in ref_sds)
        else:
            raise TypeError('Unsupported reference Slater determinants, {0}'.format(type(ref_sds)))

        for sd in self.ref_sds:
            if slater.total_occ(sd) != self.nelec:
                raise ValueError('Reference Slater determinant, {0}, does not have the same number'
                                 ' of electrons as the wavefunction'.format(bin(sd)))
            elif self._spin not in [None, slater.get_spin(sd, self.nspatial)]:
                raise ValueError('Reference Slater determinant, {0}, does not have the same spin'
                                 ' as the selected spin, {1}'.format(bin(sd), self._spin))
            elif self._seniority not in [None, slater.get_seniority(sd, self.nspatial)]:
                raise ValueError('Reference Slater determinant, {0}, does not have the same'
                                 ' seniority as the selected seniority, {1}'
                                 ''.format(bin(sd), self._seniority))


    def assign_params(self, params=None, add_noise=False):
        """ Assigns the parameters of the wavefunction

        Parameters
        ----------
        params : np.ndarray, None
            Parameters of the wavefunction
            Last parameter is the energy
            Default is the `template_coeffs` for the coefficient and energy of the reference
            Slater determinants
            If energy is given as zero, the energy of the reference Slater determinants are used
        add_noise : bool
            Flag to add noise to the given parameters

        Raises
        ------
        TypeError
            If `params` is not a numpy array
            If `params` does not have data type of `float`, `complex`, `np.float64` and
            `np.complex128`
            If `params` has data type of `float or `np.float64` and wavefunction does not have data
            type of `np.float64`
        ValueError
            If `params` is None (default) and `ref_sds` has more than one Slater determinants
            If `params` is not a one dimensional numpy array with appropriate dimensions
        """
        if params is None:
            if len(self.ref_sds) > 1:
                raise ValueError('Cannot use default initial parameters if there is more than one'
                                 ' reference Slater determinants.')
            params = self.template_coeffs.astype(self.dtype).flatten()
            params = np.hstack((params, 0))

        ncoeffs = self.template_coeffs.size
        # check input
        if not isinstance(params, np.ndarray):
            raise TypeError('Parameters must be given as a np.ndarray')
        elif params.shape != (self.nparams, ):
            raise ValueError('Parameters must be given as a one dimension array of size, {0}'
                             ''.format(self.nparams))
        elif params.dtype not in (float, complex, np.float64, np.complex128):
            raise TypeError('Data type of the parameters must be one of `float`, `complex`,'
                            ' `np.float64` and `np.complex128`')
        if params.dtype in (complex, np.complex128) and self.dtype != np.complex128:
            raise TypeError('If the parameters are `complex`, then the `dtype` of the wavefunction'
                            ' must be `np.complex128`')

        # add random noise
        if add_noise:
            # set scale
            scale = 0.2 / ncoeffs
            params[:ncoeffs] += scale * (np.random.random(ncoeffs) - 0.5)
            if params.dtype == np.complex128:
                params[:ncoeffs] += 0.001j * scale * (np.random.random(ncoeffs) - 0.5)

        self.params = params.astype(self.dtype)
        # add energy
        if self.params[-1] == 0:
            self.params[-1] = self.compute_energy(ref_sds=self.ref_sds)
        # clear cache
        self.cache = {}
        self.d_cache = {}



    def get_overlap(self, sd, deriv=None):
        """ Returns the overlap of the wavefunction with a Slater determinant

        ..math::
            \big< \Phi_i \big| \Psi \big>

        Parameters
        ----------
        sd : int, mpz
            Slater Determinant against which to project.
        deriv : int
            Index of the parameter to derivatize
            Default does not derivatize

        Returns
        -------
        overlap : float

        Raises
        ------
        TypeError
            If given Slater determinant is not compatible with the format used internally
        """
        sd = slater.internal_sd(sd)
        try:
            if deriv is None:
                return self.cache[sd]
            else:
                return self.d_cache[(sd, deriv)]
        except KeyError:
            return self.compute_overlap(sd, deriv=deriv)


    def compute_norm(self, ref_sds=None, deriv=None):
        """ Returns the norm of the wavefunction

        ..math::
            \sum_i c_i \braket{\Phi_i | \Psi}
            &= \sum_i c_i \sum_j c_j \braket{\Phi_i | \Phi_j}
            &= \sum_i c_i^2

        Parameters
        ----------
        ref_sds : int/long/gmpy2.mpz, list/tuple of ints, None
            Slater determinants that will be used as a reference for the wavefunction (e.g. for
            initial guess, energy calculation, normalization, etc)
            If `int` or `gmpy2.mpz`, then the equivalent Slater determinant (see `wfns.slater`) is
            used as a reference
            If `list` or `tuple` of Slater determinants, then multiple Slater determinants will be
            used as a reference. Note that multiple references require an initial guess
            Default uses `self.ref_sds`
        deriv : int
            Index of the parameter to derivatize
            Default is no derivatization

        Returns
        -------
        norm : float

        Raises
        ------
        TypeError
            If Slater determinants in a list or tuple are not compatible with the format used
            internally
            If Slater determinants are given in a form that is not int/long/gmpy2.mpz, list/tuple of
            ints or None
        """
        # FIXME: repeated code (from assign_ref_sds)
        if ref_sds is None:
            ref_sds = self.ref_sds
        elif isinstance(ref_sds, (int, long)) or slater.is_internal_sd(ref_sds):
            ref_sds = (slater.internal_sd(ref_sds), )
        elif isinstance(ref_sds, (list, tuple)):
            ref_sds = tuple(slater.internal_sd(i) for i in ref_sds)
        else:
            raise TypeError('Unsupported reference Slater determinants, {0}'.format(type(ref_sds)))

        # if not derivatized
        if deriv is None:
            return sum(self.get_overlap(i)**2 for i in ref_sds)
        # if derivatized
        else:
            return sum(2 * self.get_overlap(i) * self.get_overlap(i, deriv=deriv) for i in ref_sds)


    def get_energy(self, include_nuc=False, deriv=None):
        """ Returns the energy of the system (from the projected Schrodinger equation)

        Parameters
        ----------
        include_nuc : bool
            Flag to include nuclear nuclear repulsion
            Default is False
        deriv : int
            Index of the parameter to derivatize
            Default is no derivatization

        Returns
        -------
        energy : float
            If include_nuc is True, then total energy
            If include_nuc is False, then electronic energy
            Default is electronic energy
        """
        nuc_nuc = 0.0
        if include_nuc:
            nuc_nuc = self.nuc_nuc

        # if not derivatized
        if deriv is None:
            return self.params[-1] + nuc_nuc
        # if derivatized
        elif deriv == self.params.size - 1:
            return 1.0
        else:
            return 0.0


    def compute_hamiltonian(self, slater_d, deriv=None):
        """ Computes the hamiltonian of the wavefunction with respect to a Slater
        determinant

        ..math::
            \big< \Phi_i \big| H \big| \Psi \big>

        Parameters
        ----------
        slater_d : int, gmpy2.mpz
            Slater Determinant against which to project.
        deriv : None, int
            Index of the parameter with which to derivatize against
            Default is no derivatization

        Returns
        -------
        one_electron : float
            Electron nuclear attraction energy
        coulomb : float
            Coulomb electron electron repulsion energy
        exchange : float
            Exchange electron electron repulsion energy
        """
        if self._seniority == 0:
            return sen0_hamiltonian(self, slater_d, self.orbtype, deriv=deriv)
        else:
            return hamiltonian(self, slater_d, self.orbtype, deriv=deriv)


    def compute_energy(self, include_nuc=None, ref_sds=None, deriv=None):
        """ Computes the energy projected against some set of Slater determinants

        ..math::
            \frac{\sum_i \braket{\Phi_i | \Psi} \braket{\Phi_i | H | \Psi}}
            {\sum_j \braket{\Phi_j | \Psi}^2}

        Parameters
        ----------
        include_nuc : bool
            Flag to include nuclear nuclear repulsion
            Default is False
        ref_sds : int/long/gmpy2.mpz, list/tuple of ints, None
            Slater determinants that will be used as a reference for the wavefunction (e.g. for
            initial guess, energy calculation, normalization, etc)
            If `int` or `gmpy2.mpz`, then the equivalent Slater determinant (see `wfns.slater`) is
            used as a reference
            If `list` or `tuple` of Slater determinants, then multiple Slater determinants will be
            used as a reference. Note that multiple references require an initial guess
            Default uses `self.ref_sds`
        deriv : int
            Index of the parameter to derivatize
            Default is no derivatization

        Returns
        -------
        energy : float
            If include_nuc is True, then total energy
            If include_nuc is False, then electronic energy
            Default is electronic energy

        Raises
        ------
        TypeError
            If the given reference Slater determinants are not supported
        ValueError
            If the norm of the wavefunction is zero
            If the norm of the wavefunction is negative
        """
        nuc_nuc = 0.0
        if include_nuc:
            nuc_nuc = self.nuc_nuc

        # FIXME: repeated code (from assign_ref_sds)
        if ref_sds is None:
            ref_sds = self.ref_sds
        elif isinstance(ref_sds, (int, long)) or slater.is_internal_sd(ref_sds):
            ref_sds = (slater.internal_sd(ref_sds), )
        elif isinstance(ref_sds, (list, tuple)):
            ref_sds = tuple(slater.internal_sd(i) for i in ref_sds)
        else:
            raise TypeError('Unsupported reference Slater determinants, {0}'.format(type(ref_sds)))

        norm = self.compute_norm(ref_sds=ref_sds)
        if np.abs(norm) < 1e-9:
            raise ValueError('Norm of the waefunction is zero')
        elif isinstance(norm, (complex, np.complex128)) and (norm.real < 1e-9 or
                                                             abs(norm.imag) > 1e-9):
            raise ValueError('Norm of the wavefunction is complex')
        # if not derivatized
        if deriv is None:
            elec_energy = sum(self.get_overlap(i)*sum(self.compute_hamiltonian(i)) for i in ref_sds)
            elec_energy /= norm
        # if derivatized
        else:
            olp = np.array([self.get_overlap(i) for i in ref_sds])
            d_olp = np.array([self.get_overlap(i, deriv=deriv) for i in ref_sds])

            ham = np.array([sum(self.compute_hamiltonian(i)) for i in ref_sds])
            d_ham = np.array([sum(self.compute_hamiltonian(i, deriv=deriv)) for i in ref_sds])

            d_norm = self.compute_norm(ref_sds=ref_sds, deriv=deriv)

            elec_energy = np.sum(d_olp * ham + olp * d_ham) / norm
            elec_energy += np.sum(olp * ham) / (-norm**2) * d_norm

        return elec_energy + nuc_nuc


    #############
    # Objective #
    #############
    def objective(self, x, weigh_constraints=True, save_file=None):
        """ System of (usually) nonlinear functions that corresponds to the projected Schrodinger
        equation

        ..math::
            f_1(x) &= \braket{\Phi_1 | H | \Psi} - E \braket{\Phi_1 | \Psi}
            &\vdots
            f_K(x) &= \braket{\Phi_K | H | \Psi} - E \braket{\Phi_K | \Psi}
            f_{K+1}(x) &= constraint_1
            &\vdots

        where :math:`K` is the number of Slater determinant onto which the wavefunction is projected
        Equations after the :math:`K`th index are the constraints on the system of equations.
        The constraints, hopefully, will move out into their own module some time in the future.
        By default, the normalization constraint
        ..math::
            f_{K+1} = norm - 1
        is present where :math:`norm` is defined by ProjectedWavefunction.compute_norm.

        Parameters
        ----------
        x : 1-index np.ndarray
            Coefficient vector
        weigh_constraints : bool
            Flag for weighing the norm heavier by some arbitrary value
            By default, the norm equation is weighted heavier by a factor of 1000*(number of terms
            in the system of nonlinear equations)
        save_file : str
            Name of the `.npy` file that will be used to store the parameters in the course of the
            optimization
            Default is no save file

        Returns
        -------
        obj : np.ndarray(self.nproj+self._nconstraints,)
        """
        # Update the coefficient vector
        self.params[:] = x
        # Save params
        if save_file is not None:
            np.save('{0}_temp.npy'.format(save_file), self.params)
        # Clear cache
        self.cache = {}

        obj = np.empty(self.nproj + self._nconstraints, dtype=self.dtype)
        # <SD|H|Psi> - E<SD|H|Psi> == 0
        for i, sd in enumerate(self.pspace):
            obj[i] = sum(self.compute_hamiltonian(sd)) - self.get_energy() * self.get_overlap(sd)
        # Add constraints
        # FIXME: constraint weight needs to be repeated in jacobian
        if weigh_constraints:
            obj[self.nproj] = (self.compute_norm() - 1)*(self.nproj + self._nconstraints)
        else:
            obj[self.nproj] = (self.compute_norm() - 1)

        return obj


    def jacobian(self, x, weigh_constraints=True):
        """ Jacobian of the objective function

        A matrix is returned
        ..math::
            J_{ij}(\vec{x}) = \frac{\partial f_i(\vec{x})}{\partial x_j}
        where different value is returned depending on the :math:`\vec{x}`

        Parameters
        ----------
        x : 1-index np.ndarray
            Coefficient vector
        weigh_constraints : bool
            Flag for weighing the norm heavier by some arbitrary value
            By default, the norm equation is weighted heavier by a factor of 1000*(number of terms
            in the system of nonlinear equations)

        Returns
        -------
        jac : np.ndarray(self.nproj+self._nconstraints, self.nparams)
            Value of the Jacobian :math:`J_{ij}`
        """
        # Clear cache
        self.cache = {}
        self.d_cache = {}

        # set energy
        energy = self.get_energy()
        jac = np.empty((self.nproj + self._nconstraints, self.nparams), dtype=self.dtype)

        for j in range(self.nparams):
            d_energy = self.get_energy(deriv=j)
            for i, sd in enumerate(self.pspace):
                # <SD|H|Psi> - E<SD|H|Psi> = 0
                jac[i, j] = (sum(self.compute_hamiltonian(sd, deriv=j))
                             - energy*self.get_overlap(sd, deriv=j) - d_energy*self.get_overlap(sd))
            # Add normalization constraint
            # FIXME: constrain weight needs to be repeated here
            if weigh_constraints:
                jac[self.nproj, j] = (self.compute_norm(deriv=j)-1)*(self.nproj+self._nconstraints)
            else:
                jac[self.nproj, j] = (self.compute_norm(deriv=j)-1)

        return jac


    #####################
    # Abstract Property #
    #####################
    @abstractproperty
    def template_coeffs(self):
        """ Default parameters for the given reference Slater determinantes

        Note
        ----
        Returned value must be a numpy array in the desired shape
        """
        pass


    ####################
    # Abstract Methods #
    ####################
    @abstractmethod
    def compute_overlap(self, sd, deriv=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        overlap : float
        """
        pass

    @abstractmethod
    def normalize(self):
        """ Normalizes the wavefunction such that the norm with respect to `ref_sds` is 1
        """
        pass
