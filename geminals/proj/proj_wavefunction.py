from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from gmpy2 import mpz
from scipy.optimize import root, least_squares

from ..wavefunction import Wavefunction


class ProjectionWavefunction(Wavefunction):
    """ Wavefunction obtained through projection

    Contains the necessary information to projectively solve the wavefunction

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : np.ndarray(K,K) or tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        If tuple of np.ndarray (length 2), one electron integrals for the (alpha, alpha)
        and the (beta, beta) unrestricted orbitals
    G : np.ndarray(K,K,K,K) or tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        If tuple of np.ndarray (length 3), two electron integrals for the
        (alpha, alpha, alpha, alpha), (alpha, beta, alpha, beta), and
        (beta, beta, beta, beta) unrestricted orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nelec : int
        Number of electrons
    orb_type : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    params : np.ndarray(K)
        Guess for the parameters
        Iteratively updated during convergence
        Initial guess before convergence
        Coefficients after convergence
    cache : dict of mpz to float
        Cache of the Slater determinant to the overlap of the wavefunction with this
        Slater determinant
    d_cache : dict of (mpz, int) to float
        Cache of the Slater determinant to the derivative(with respect to some index)
        of the overlap of the wavefunction with this Slater determinan

    Properties
    ----------
    _methods : dict of func
        Dictionary of methods that are used to solve the wavefunction
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals
    npair : int
        Number of electron pairs (rounded down)
    nparam : int
        Number of parameters used to define the wavefunction
    nproj : int
        Number of Slater determinants to project against
    energy_index : int
        Index of the energy in the list of parameters
    ref_sd : int or list of int
        Reference Slater determinants with respect to which the norm and the energy
        are calculated
        Integer that describes the occupation of a Slater determinant as a bitstring
        Or list of integers

    Method
    ------
    __init__(nelec=None, H=None, G=None, dtype=None, nuc_nuc=None, orb_type=None)
        Initializes wavefunction
    __call__(method="default", **kwargs)
        Solves the wavefunction
    assign_dtype(dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_integrals(H, G, orb_type=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
        (and the wavefunction)
    assign_nuc_nuc(nuc_nuc=None)
        _Assigns the nuclear nuclear repulsion
    assign_nelec(nelec)
        Assigns the number of electrons
    _solve_least_squares(**kwargs)
        Solves the system of nonliear equations (and the wavefunction) using
        least squares method
    assign_params(params=None)
        Assigns the parameters used to describe the wavefunction.
        Adds random noise from the template if necessary
    assign_pspace(pspace=None)
        Assigns projection space
    overlap(sd, deriv=None)
        Retrieves overlap from the cache if available, otherwise compute overlap
    compute_norm(sd=None, deriv=None)
        Computes the norm of the wavefunction
    compute_energy(include_nuc=False, sd=None, deriv=None)
        Computes the energy of the wavefunction
    objective(x)
        The objective (system of nonlinear equations) associated with the projected
        Schrodinger equation
    jacobian(x)
        The Jacobian of the objective

    Abstract Property
    -----------------
    template_params : np.ndarray(K)
        Default numpy array of parameters.
        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

    Abstract Method
    ---------------
    compute_pspace
        Generates a tuple of Slater determinants onto which the wavefunction is projected
    compute_overlap
        Computes the overlap of the wavefunction with one or more Slater determinants
    compute_hamiltonian
        Computes the hamiltonian of the wavefunction with respect to one or more Slater
        determinants
        By default, the energy is determined with respect to ref_sd
    normalize
        Normalizes the wavefunction (different definitions available)
        By default, the norm should the projection against the ref_sd squared
    """
    # FIXME: turn C into property and have a better attribute name
    __metaclass__ = ABCMeta

    #
    # Abstract Property
    #
    @abstractproperty
    def template_params(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_params : np.ndarray(K, )

        """
        pass

    #
    # Properties
    #
    @property
    def _methods(self):
        """ Dictionary of methods for solving the wavefunction

        Returns
        -------
        methods : dict
            "default" -> least squares nonlinear solver
        """
        return {
                "default": self._solve_root, 
                "leastsq": self._solve_least_squares,
                }

    @property
    def nparam(self):
        """ Number of parameters

        Returns
        -------
        nparam : int
        """
        return self.params.size

    @property
    def nproj(self):
        """ Number of Slater determinants to project against

        Returns
        -------
        nproj : int
        """
        return len(self.pspace)

    @property
    def energy_index(self):
        """ Index of the energy in the list of parameters

        Returns
        -------
        energy_index : int
        """
        if self.energy_is_param:
            return self.nparam - 1
        else:
            return self.nparam

    @property
    def ref_sd(self):
        """ Reference Slater determinant

        You can overwrite this attribute if you want to change the reference Slater determinant

        Returns
        -------
        ref_sd : int, list of int
            Integer that describes the occupation of a Slater determinant as a bitstring
            Or list of integers
        """
        return self.pspace[0]

    #
    # Special methods
    #

    def __init__(
        self,
        # Mandatory arguments
        nelec=None,
        H=None,
        G=None,
        # Arguments handled by base Wavefunction class
        dtype=None,
        nuc_nuc=None,
        # Arguments handled by FullCI class
        params=None,
        pspace=None,
        energy_is_param=True
    ):
        super(ProjectionWavefunction, self).__init__(
            nelec=nelec,
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
        )
        self.energy_is_param = energy_is_param
        self.assign_params(params=params)
        self.assign_pspace(pspace=pspace)
        del self._energy
        self.cache = {}
        self.d_cache = {}

    #
    # Solver methods
    #
    def _solve_root(self, **kwargs):
        """
        Optimize `self.objective(params)` to solve the coefficient vector.

        Parameters
        ----------
        jacobian : bool, optional
        If False, the Jacobian is not used in the optimization.
        kwargs : dict, optional
        Keywords to pass to the internal solver, `scipy.optimize.root`.

        Returns
        -------
        result : tuple
        See `scipy.optimize.root` documentation.
        """

        if 'jacobian' not in kwargs or kwargs['jacobian']:  
            # Update solver options
            options = {
                    # Powell's hybrid method (MINPACK)
                    "method": 'hybr',
                    # "bounds": self.bounds,
                    "jac": self.jacobian,
                    "options": {
                        "xtol":1.0e-9,
                        },
                    }
        else:
            # Update solver options
            options = {
                # Newton-Krylov Quasinewton method
                "method": 'krylov',
                # "bounds": self.bounds,
                # "jac": None,
                "options": {
                    "fatol":1.0e-9,
                    "xatol":1.0e-7,
                    },
                }

        options.update(kwargs)

        # Solve
        result = root(self.objective, self.params, **options)
        return result

    def _solve_least_squares(self, **kwargs):
        """
        Optimize `self.objective(params)` to solve the coefficient vector.

        Parameters
        ----------
        jacobian : bool, optional
        If False, the Jacobian is not used in the optimization.
        kwargs : dict, optional
        Keywords to pass to the internal solver, `scipy.optimize.leastsq`.

        Returns
        -------
        result : tuple
        See `scipy.optimize.leastsq` documentation.
        """

        # Update solver options
        options = {
            "jac": self.jacobian,
            # "bounds": self.bounds,
            # "jac": None,
            "xtol": 1.0e-15,
            "ftol": 1.0e-15,
            "gtol": 1.0e-15,
        }
        options.update(kwargs)
        # Use appropriate Jacobian approximation if necessary
        if not options["jac"]:
            if self.dtype == np.complex128:
                options["jac"] = "cs"
            else:
                options["jac"] = "3-point"
        # Solve
        result = least_squares(self.objective, self.params, **options)
        return result

    #
    # Assignment methods
    #
    def assign_params(self, params=None):
        """ Assigns the parameters to the wavefunction

        Parameters
        ----------
        params : np.ndarray(K,)
            Parameters of the wavefunction
        """
        # parameter shape
        params_shape = (self.template_params.size + self.energy_is_param,)
        # number of coefficients (non energy parameters)
        ncoeffs = self.template_params.size
        if params is None:
            params = self.template_params
            # set scale
            scale = 1.0 / ncoeffs
            # set energy
            if self.energy_is_param:
                params = np.hstack((params, 0.0))
            # add random noise to template
            params[:ncoeffs] += scale * (np.random.random(ncoeffs) - 0.5)
            if params.dtype == np.complex128:
                params[:ncoeffs] += 1j * scale * (np.random.random(ncoeffs) - 0.5)
        if not isinstance(params, np.ndarray):
            raise TypeError("params must be of type {0}".format(np.ndarray))
        elif params.shape != params_shape:
            raise ValueError("params must be of right shape({0})".format(params_shape))
        elif params.dtype not in (float, complex, np.float64, np.complex128):
            raise TypeError("params's dtype must be one of {0}".format((float, complex, np.float64, np.complex128)))

        self.params = params
        self.cache = {}
        self.d_cache = {}

    def assign_pspace(self, pspace=None):
        """ Sets the Slater determinants on which to project against

        Parameters
        ----------
        pspace : int, iterable of int,
            If iterable, then it is a list of Slater determinants (in the form of integers that describe
            the occupation as a bitstring)
            If integer, then it is the number of Slater determinants to be generated
        """
        if pspace is None:
            pspace = self.compute_pspace(self.nparam - 1)
        if isinstance(pspace, int):
            pspace = self.compute_pspace(pspace)
        elif isinstance(pspace, (list, tuple)):
            if not all(type(i) in [int, type(mpz())] for i in pspace):
                raise ValueError('Each Slater determinant must be an integer or mpz object')
            pspace = [mpz(sd) for sd in pspace]
        else:
            raise TypeError("pspace must be an int, list or tuple")
        self.pspace = tuple(pspace)

    #
    # View method
    #

    def overlap(self, sd, deriv=None):
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
        """
        sd = mpz(sd)
        try:
            if deriv is None:
                return self.cache[sd]
            else:
                # construct new mpz to describe the slater determinant and
                # derivation index
                return self.d_cache[(sd, deriv)]
        except KeyError:
            return self.compute_overlap(sd, deriv=deriv)

    def compute_norm(self, sd=None, deriv=None):
        """ Returns the norm of the wavefunction

        ..math::
            \big< \Phi_i \big| \Psi \big> &= \sum_j c_j \big< \Phi_i \big| \Phi_j \big>\\
                                          &= c_i
        or
        ..math::
            c_i \big< \Phi_i \big| \Psi \big> &= c_i \sum_j c_j \big< \Phi_i \big| \Phi_j \big>\\
                                              &= c_i^2
        or
        ..math::
            \sum_i c_i \big< \Phi_i \big| \Psi \big> &= \sum_i c_i \sum_j c_j \big< \Phi_i \big| \Phi_j \big>\\
                                                     &= \sum_i c_i^2

        Parameters
        ----------
        sd : int, mpz, iterable of (int, gmpy2.mpz)
            If int or mpz, then an integer that describes the occupation of
            a Slater determinant as a bitstring
            If iterable, then list of integers that describes the occupation of
            a Slater determinant as a bitstring
        deriv : int
            Index of the parameter to derivatize
            Default is no derivatization

        Returns
        -------
        norm : float
        """
        if sd is None:
            sd = self.ref_sd
        if type(sd) in [int, type(mpz())]:
            sd = (sd,)
        if not isinstance(sd, (list, tuple)):
            raise TypeError('Slater determinants must be given as an int or mpz or list/tuple of these')
        if not all(type(i) in [int, type(mpz())] for i in sd):
            raise TypeError('List of Slater determinants must all be of type int or mpz')
        # convert to mpz
        sd = [mpz(i) for i in sd]
        # if not derivatized
        if deriv is None:
            return sum(self.overlap(i)**2 for i in sd)
        # if derivatized
        else:
            return sum(2 * self.overlap(i) * self.overlap(i, deriv=deriv) for i in sd)

    def compute_energy(self, include_nuc=False, sd=None, deriv=None):
        """ Returns the energy of the system

        ..math::
            \big< \Phi_i \big| H \big| \Psi \big>

        Parameters
        ----------
        sd : int, mpz, list of (int, gmpy2.mpz)
            If an int or mpz is given,
            ..math::
                \frac{c_i \big< \Phi_i \big| H \big| \Psi \big>}{\big< \Phi_i \big| \Psi \big>}
            is calculated
            If a list of int or mpz is given,
            ..math::
                \frac{\sum_i \big< \Phi_i \big| \Psi \big> \big< \Phi_i \big| H \big| \Psi \big>}{\sum_j \big< \Phi_j \big| \Psi \big>^2}
            is calculated
            This is only useful if the energy is not a parameter
            Default is the self.ref_sd
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
        # if energy is a parameter
        if self.energy_is_param:
            if sd is not None:
                print('Warning: Cannot specify Slater determinant to compute energy if energy is a parameter')
            # if not derivatized
            if deriv is None:
                return self.params[-1] + nuc_nuc
            # if derivatized
            elif deriv == self.energy_index:
                return 1.0
            else:
                return 0.0
        # if energy is not a parameter
        else:
            # if sd is None
            if sd is None:
                sd = self.ref_sd
            # if sd is not None
            if type(sd) in [int, type(mpz())]:
                sd = (sd,)
            if not isinstance(sd, (list, tuple)):
                raise TypeError('Unsupported Slater determinant type {0}'.format(type(sd)))
            if not all(type(i) in [int, type(mpz())] for i in sd):
                raise TypeError('List of Slater determinants must all be of type int or mpz')
            # convert to mpz
            sd = [mpz(i) for i in sd]

            # if not derivatized
            if deriv is None:
                elec_energy = sum(self.overlap(i) * self.compute_hamiltonian(i) for i in sd)
                elec_energy /= self.compute_norm(sd=sd)
            # if derivatized
            else:
                olp = np.array([self.overlap(i) for i in sd])
                d_olp = np.array([self.overlap(i, deriv=deriv) for i in sd])
                ham = np.array([self.compute_hamiltonian(i) for i in sd])
                d_ham = np.array([self.compute_hamiltonian(i, deriv=deriv) for i in sd])
                norm = self.compute_norm(sd=sd)
                d_norm = self.compute_norm(sd=sd, deriv=deriv)
                elec_energy = np.sum(d_olp * ham + olp * d_ham) / norm
                elec_energy += np.sum(olp * ham) / (-norm**2) * d_norm
            return elec_energy + nuc_nuc

    #
    # Objective
    #
    def objective(self, x):
        """ System of nonlinear functions that corresponds to the projected Schrodinger equation

        The set of equations is
        ..math::
            f_i(x) = \big< \Phi_i \big| H \big| \Psi \big> - E \big< \Phi_i \big| \Psi \big>
        where :math:`E` is defined by ProjectionWavefunction.compute_energy.
        An extra equation is added at the end so that the waavefunction is normalized
        ..math::
            f_{last} = norm - 1
        where :math:`norm` is defined by ProjectionWavefunction.compute_norm.
        This function will be zero if the function is normalized.
        The solver solves for `x` such that
        ..math::
            f_i(x) = 0

        Parameters
        ----------
        x : 1-index np.ndarray
            The coefficient vector

        Returns
        -------
        value : np.ndarray(self.nproj+1,)
            Value of the function :math:`f`

        """
        # Update the coefficient vector
        self.params[:] = x
        # Clear cache
        self.cache = {}
        self.d_cache = {}

        # set energy
        energy = self.compute_energy()
        obj = np.empty(self.nproj + 1, dtype=self.dtype)

        # <SD|H|Psi> - E<SD|H|Psi> == 0
        for i, sd in enumerate(self.pspace):
            obj[i] = self.compute_hamiltonian(sd) - energy * self.overlap(sd)
        # Add normalization constraint
        obj[-1] = self.compute_norm() - 1.0
        return obj

    def jacobian(self, x):
        """ Jacobian of the objective function

        A matrix is returned
        ..math::
            J_{ij}(\vec{x}) = \frac{\partial f_i(\vec{x})}{\partial x_j}
        where different value is returned depending on the :math:`\vec{x}`

        Parameters
        ----------
        x : 1-index np.ndarray
            The coefficient vector

        Returns
        -------
        value : np.ndarray(self.nproj+1, self.nparam)
            Value of the Jacobian :math:`J_{ij}`

        """
        # Update the coefficient vector
        self.params[:] = x
        # Clear cache
        self.cache = {}
        self.d_cache = {}

        # set energy
        energy = self.compute_energy()
        jac = np.empty((self.nproj + 1, self.nparam), dtype=self.dtype)

        for j in range(self.nparam):
            d_energy = self.compute_energy(deriv=j)
            for i, sd in enumerate(self.pspace):
                # <SD|H|Psi> - E<SD|H|Psi> = 0
                jac[i, j] = (self.compute_hamiltonian(sd, deriv=j) -
                             energy * self.overlap(sd, deriv=j) - d_energy * self.overlap(sd))
            # Add normalization constraint
            jac[-1, j] = self.compute_norm(deriv=j)
        return jac

    #
    # Abstract Methods
    #
    @abstractmethod
    def compute_pspace(self, num_sd):
        """ Generates Slater determinants to project onto

        Parameters
        ----------
        num_sd : int
            Number of Slater determinants to generate

        Returns
        -------
        pspace : list of gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        """
        pass

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
        # caching is done wrt mpz objects, so you should convert sd to mpz first
        sd = gmpy2.mpz(sd)
        pass

    @abstractmethod
    def compute_hamiltonian(self, sd, deriv=None):
        """ Computes the hamiltonian of the wavefunction with respect to a Slater
        determinant

        ..math::
            \big< \Phi_i \big| H \big| \Psi \big>

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
        float
        """
        pass

    @abstractmethod
    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in
        ProjectionWavefunction.compute_norm

        Some of the cache are emptied because the parameters are rewritten
        """
        pass
