from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from gmpy2 import mpz

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
    orbtype : {'restricted', 'unrestricted', 'generalized'}
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
    ref_sd : int or list of int
        Reference Slater determinants with respect to which the norm and the energy
        are calculated
        Integer that describes the occupation of a Slater determinant as a bitstring
        Or list of integers

    Method
    ------
    __init__(nelec=None, H=None, G=None, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    __call__(method="default", **kwargs)
        Solves the wavefunction
    assign_dtype(dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_integrals(H, G, orbtype=None)
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
    template_coeffs : np.ndarray(K)
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
    # FIXME: does this need to be changed into an abstract property
    @abstractproperty
    def template_coeffs(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_coeffs : np.ndarray(K, )

        """
        pass

    #
    # Properties
    #
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
    def default_ref_sds(self):
        """ Reference Slater determinant

        You can overwrite this attribute if you want to change the reference Slater determinant

        Returns
        -------
        default_ref_sd : int, list of int
            Integer that describes the occupation of a Slater determinant as a bitstring
            Or list of integers
        """
        return (self.pspace[0], )

    @property
    def bounds(self):
        """ Boundaries for the parameters

        Used to set bounds on the optimizer

        Returns
        -------
        bounds : iterable of 2-tuples
            Each 2-tuple correspond to the min and the max value for the parameter
            with the same index.
        """
        low_bounds = [-1.2 for i in range(self.nparam)]
        upp_bounds = [1.2 for i in range(self.nparam)]
        # remove boundary on energy
        low_bounds[-1] = -np.inf
        upp_bounds[-1] = np.inf
        return (tuple(low_bounds), tuple(upp_bounds))


    @property
    def nconstraints(self):
        """Number of constraints on the sollution of the projected wavefunction.

        By default this is - 1 because we need one equation for normalization.

        Returns
        -------
        nconstraints : int
        """
        self._nconstraints = 1
        return self._nconstraints

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
        params=None,
        pspace=None,
        # Arguments for saving parameters
        params_save_name=''
    ):
        super(ProjectionWavefunction, self).__init__(
            nelec=nelec,
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
        )
        self.assign_params_save(params_save_name=params_save_name)
        self.assign_params(params=params)
        self.assign_pspace(pspace=pspace)
        if params is None:
            self.params[-1] = self.compute_energy(ref_sds=self.default_ref_sds)
        del self._energy
        self.cache = {}
        self.d_cache = {}

    #
    # Assignment methods
    #
    def assign_params_save(self, params_save_name=''):
        """ Assigns the npy file name that stores the parameters

        Parameters
        ----------
        npy_name : str
            Name of the npy file that will contain the parameters

        """
        if not isinstance(params_save_name, str):
            raise TypeError('The numpy file name must be a string')
        self.params_save_name = params_save_name

    def assign_params(self, params=None):
        """ Assigns the parameters to the wavefunction

        Parameters
        ----------
        params : np.ndarray(K,)
            Parameters of the wavefunction
        """
        # number of coefficients (non energy parameters)
        ncoeffs = self.template_coeffs.size
        if params is None:
            params = self.template_coeffs.flatten()
            # set scale
            scale = 0.2 / ncoeffs
            # add random noise to template
            params[:ncoeffs] += scale * (np.random.random(ncoeffs) - 0.5)
            if params.dtype == np.complex128:
                params[:ncoeffs] += 0.001j * scale * (np.random.random(ncoeffs) - 0.5)
            # set energy
            # NOTE: the energy cannot be set with compute_energy right now because
            # certain terms must be defined for compute_hamiltonian to work
            energy = 0.0
            params = np.hstack((params, energy))
        if not isinstance(params, np.ndarray):
            raise TypeError("params must be of type {0}".format(np.ndarray))
        elif params.shape != (ncoeffs+1, ):
            raise ValueError("params must be of right shape({0})".format(ncoeffs + 1))
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
            pspace = self.compute_pspace(self.nparam - self.nconstraints)
            # - the number of constraints already impored on the wfn
            # in general this is - 1 because we need one equation for normalization
        # FIXME: this is quite terrible
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

    def compute_norm(self, ref_sds=None, deriv=None):
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
        ref_sds : iterable of (int, gmpy2.mpz)
            If iterable, then list of integers that describes the occupation of
            a Slater determinant as a bitstring
        deriv : int
            Index of the parameter to derivatize
            Default is no derivatization

        Returns
        -------
        norm : float
        """
        if ref_sds is None:
            ref_sds = self.default_ref_sds
        if not isinstance(ref_sds, (list, tuple)):
            raise TypeError('The reference Slater determinants must be given as a list or tuple')
        if not all(type(i) in [int, type(mpz())] for i in ref_sds):
            raise TypeError('Each reference Slater determinant must be of type int or mpz')
        # convert to mpz
        ref_sds = [mpz(i) for i in ref_sds]
        # if not derivatized
        if deriv is None:
            return sum(self.overlap(i)**2 for i in ref_sds)
        # if derivatized
        else:
            return sum(2 * self.overlap(i) * self.overlap(i, deriv=deriv) for i in ref_sds)

    def compute_energy(self, include_nuc=False, ref_sds=None, deriv=None):
        """ Returns the energy of the system

        ..math::
            \big< \Phi_i \big| H \big| \Psi \big>

        Parameters
        ----------
        ref_sds : int, mpz, list of (int, gmpy2.mpz)
            If an int or mpz is given,
            ..math::
                \frac{c_i \big< \Phi_i \big| H \big| \Psi \big>}{\big< \Phi_i \big| \Psi \big>}
            is calculated
            If a list of int or mpz is given,
            ..math::
                \frac{\sum_i \big< \Phi_i \big| \Psi \big> \big< \Phi_i \big| H \big| \Psi \big>}{\sum_j \big< \Phi_j \big| \Psi \big>^2}
            is calculated
            This is only useful if the energy is not a parameter
            Default is the self.default_ref_sds
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
        # set nuclear nuclear repulsion
        nuc_nuc = 0.0
        if include_nuc:
            nuc_nuc = self.nuc_nuc
        if ref_sds is None:
            # if not derivatized
            if deriv is None:
                return self.params[-1] + nuc_nuc
            # if derivatized
            elif deriv == self.params.size - 1:
                return 1.0
            else:
                return 0.0
        else:
            if not isinstance(ref_sds, (list, tuple)):
                raise TypeError('The reference Slater determinants must given as a list or tuple')
            if not all(type(i) in [int, type(mpz())] for i in ref_sds):
                raise TypeError('Each Slater determinants must be of type int or mpz')
            # convert to mpz
            ref_sds = [mpz(sd) for sd in ref_sds]

            # if not derivatized
            if deriv is None:
                elec_energy = sum(self.overlap(i) * self.compute_hamiltonian(i) for i in ref_sds)
                elec_energy /= self.compute_norm(ref_sds=ref_sds)
            # if derivatized
            else:
                olp = np.array([self.overlap(i) for i in ref_sds])
                d_olp = np.array([self.overlap(i, deriv=deriv) for i in ref_sds])
                ham = np.array([self.compute_hamiltonian(i) for i in ref_sds])
                d_ham = np.array([self.compute_hamiltonian(i, deriv=deriv) for i in ref_sds])
                norm = self.compute_norm(ref_sds=ref_sds)
                d_norm = self.compute_norm(ref_sds=ref_sds, deriv=deriv)
                elec_energy = np.sum(d_olp * ham + olp * d_ham) / norm
                elec_energy += np.sum(olp * ham) / (-norm**2) * d_norm
            return elec_energy + nuc_nuc


    #
    # Objective
    #
    def objective(self, x, weigh_norm=True):
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
        weigh_norm : bool
            Flag for weighing the norm heavier by some arbitrary value
            By default, the norm equation is weighted heavier by a factor of
            1000*(number of terms in the system of nonlinear equations)

        Returns
        -------
        value : np.ndarray(self.nproj+1,)
            Value of the function :math:`f`

        """
        # Update the coefficient vector
        self.params[:] = x
        # Normalize
        # self.normalize()
        # Save params
        if self.params_save_name:
            np.save('{0}_temp.npy'.format(self.params_save_name), self.params)
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
        if weigh_norm:
            obj[-1] = (self.compute_norm() - 1.0)*self.params.size*(len(self.pspace)+1)*1000
        else:
            obj[-1] = (self.compute_norm() - 1.0)

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
            jac[-1, j] = self.compute_norm(deriv=j)*self.params.size*(len(self.pspace)+1)*1000
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
