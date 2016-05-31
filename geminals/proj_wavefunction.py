from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from wavefunction import Wavefunction
import slater
from gmpy2 import mpz
from scipy.optimize import least_squares


class ProjectionWavefunction(Wavefunction):
    """ Wavefunction obtained through projection

    Contains the necessary information to projectively solve the wavefunction

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : np.ndarray(K,K)
        One electron integrals for the spatial orbitals
    Ha : np.ndarray(K,K)
        One electron integrals for the alpha spin orbitals
    Hb : np.ndarray(K,K)
        One electron integrals for the beta spin orbitals
    G : np.ndarray(K,K,K,K)
        Two electron integrals for the spatial orbitals
    Ga : np.ndarray(K,K,K,K)
        Two electron integrals for the alpha spin orbitals
    Gb : np.ndarray(K,K,K,K)
        Two electron integrals for the beta spin orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nspatial : int
        Number of spatial orbitals
    nspin : int
        Number of spin orbitals (alpha and beta)
    nelec : int
        Number of electrons
    npair : int
        Number of electron pairs
        Assumes that the number of electrons is even
    nparticle : int
        Number of quasiparticles (electrons)
    ngeminal : int
        Number of geminals
    nproj : int
        Dimension of the projection space
    nparam : int
        Number of parameters
    x : np.ndarray(K)
        Guess for the parameters
        Iteratively updated during convergence
        Initial guess before convergence
        Coefficients after convergence
    jacobian
        Returns the Jacobian of the parameters
        Not an abstract method because the wavefunction can still be solved without it

    Private
    -------
    _methods : dict
        Default dimension of projection space

    Abstract Property
    -----------------
    _nproj_default : int
        Default number of projection states

    Abstract Method
    ---------------
    compute_pspace
        Generates a list of states to project onto
    compute_overlap
        Computes the overlap of a Slater deteriminant with the wavefunction
    compute_hamiltonian
        Computes the hamiltonian of a Slater deteriminant with the wavefunction
    """
    # FIXME: turn C into property and have a better attribute name
    __metaclass__ = ABCMeta

    #
    # Default attribute values
    #

    @abstractproperty
    def _nproj_default(self):
        """
        Default number of Slater determinants in the projection space
        """
        pass

    @property
    def _methods(self):
        """ Dictionary of methods for solving the wavefunction

        Returns
        -------
        methods : dict
            "default" -> least squares nonlinear solver
        """
        return {"default": self._solve_least_squares}

    #
    # Properties
    #
    @property
    def nparam(self):
        """ Number of parameters
        """
        return self.params.size

    @property
    def energy_index(self):
        if self.energy_is_param:
            return self.nparam-1
        else:
            return self.nparam

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
        nproj=None,
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
        self.assign_nproj(nproj=nproj)
        del self._energy
        self.cache = {}
        self.d_cache = {}

    #
    # Solver methods
    #
    def _solve_least_squares(self, **kwargs):
        """
        Optimize `self.objective(params)` to solve the coefficient vector.

        Parameters
        ----------
        jacobian : bool, optional
        If False, the Jacobian is not used in the optimization.
        If False, it is not used.
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
    @abstractproperty
    def template_params(self):
        pass

    def assign_params(self, params=None):
        """ Assigns the parameters to the wavefunction

        Parameters
        ----------
        params : np.ndarray(K,)
        Parameters of the wavefunction
        """
        if self.energy_is_param:
            nparam = self._nproj_default + 1
        else:
            nparam = self._nproj_default
        if params is None:
            params = self.template_params
            # set scale
            scale = 1.0/(self._nproj_default)
            # set energy
            if self.energy_is_param:
                energy_index = nparam - 1
                params[energy_index] = 0.0
            else:
                energy_index = nparam
            # add random noise to template
            params[:energy_index] += scale*(2*np.random.random(self._nproj_default) - 1)
            if params.dtype == np.complex128:
                params[:energy_index] += 1j*scale*(2*np.random.random(self._nproj_default) - 1)
        if not isinstance(params, np.ndarray):
            raise TypeError("params must be of type {0}".format(np.ndarray))
        elif params.shape != (nparam,):
            raise ValueError("params must be of length nparam ({0})".format(nparam))
        elif params.dtype not in (float, complex, np.float64, np.complex128):
            raise TypeError("params's dtype must be one of {0}".format((float, complex, np.float64, np.complex128)))

        self.params = params
        self.cache = {}
        self.d_cache = {}


    def assign_pspace(self, pspace=None):
        """ Sets the Slater determinants on which to project against

        Parameters
        ----------
        civec : iterable of int
            List of Slater determinants (in the form of integers that describe
            the occupation as a bitstring)
        """
        #FIXME: code repeated in ci_wavefunction.assign_civec
        #FIXME: cyclic dependence on pspace
        if pspace is None:
            pspace = self.compute_pspace()
        if not isinstance(pspace, (list, tuple)):
            raise TypeError("pspace must be a list or a tuple")
        if not all(type(i) in [int, type(mpz())] for i in pspace):
            raise ValueError('Each Slater determinant must be an integer or mpz object')
        self.pspace = tuple(pspace)

    def assign_nproj(self, nproj=None):
        """ Sets number of Slater determinants on which to project against

        Parameters
        ----------
        nproj : int
            Number of projection states
        """
        #FIXME: cyclic dependence on nproj
        if nproj is None:
            nproj = self._nproj_default
        if not isinstance(nproj, int):
            raise TypeError("nproj must be of type {0}".format(int))
        self.nproj = nproj

    #
    # Computation methods
    #

    @abstractmethod
    def compute_pspace(self):
        """ Generates Slater determinants on which to project against

        Number of Slater determinants generated is determined strictly by the size of the
        projection space (self.nproj). First row corresponds to the ground state SD, then
        the next few rows are the first excited, then the second excited, etc

        Returns
        -------
        civec : list of ints
            Integer that describes the occupation of a Slater determinant as a bitstring
        """
        pass

    @abstractmethod
    def compute_overlap(self, sd, deriv=None):
        pass

    @abstractmethod
    def compute_hamiltonian(self, sd, deriv=None):
        pass

    #
    # View method
    #

    def overlap(self, sd, deriv=None):
        """ Returns the overlap of the wavefunction with a Slater determinant

        ..math::
            \big< \Phi_i \big| \Psi \big>

        Parameters
        ----------
        sd : int
            Slater Determinant against which to project.
        deriv : int
            Index of the parameter to derivatize
            Default does not derivatize

        Returns
        -------
        overlap : float
            Overlap
        """
        try:
            if deriv is None:
                return self.cache[sd]
            else:
                # construct new gmpy2.mpz to describe the slater determinant and
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
        """
        if sd is None:
            sd = self.pspace[0]
        if type(sd) in [int, type(mpz())]:
            sd = (sd,)
        if not isinstance(sd, (list, tuple)):
            raise TypeError('Slater determinants must be given as an int or gmpy2.mpz or list/tuple of these')
        if not all(type(i) in [int, type(mpz())] for i in sd):
            raise TypeError('List of Slater determinants must all be of type int or gmpy2.mpz')
        # if not derivatized
        if deriv is None:
            return sum(self.overlap(i)**2 for i in sd)
        # if derivatized
        else:
            return sum(2*self.overlap(i)*self.overlap(i, deriv=deriv) for i in sd)

    def compute_energy(self, include_nuc=True, sd=None, deriv=None):
        """ Returns the energy of the system

        ..math::
            \big< \Phi_i \big| H \big| \Psi \big>

        Parameters
        ----------
        sd : int, gmpy2.mpz, list of int or gmpy2.mpz
            Slater Determinant against which to project.
            Default is the energy used to parameterize the system
            If an int or gmpy2.mpz is given,
            ..math::
                \frac{c_i \big< \Phi_i \big| H \big| \Psi \big>}{\big< \Phi_i \big| \Psi \big>}
            is calculated
            If a list of int or gmpy2.mpz is given,
            ..math::
                \frac{\sum_i c_i \big< \Phi_i \big| H \big| \Psi \big>}{\sum_j \big< \Phi_j \big| \Psi \big>^2}
            is calculated
        include_nuc : bool
            Flag to include nuclear nuclear repulsion

        Returns
        -------
        energy : float
            Total energy if include_nuc is True
        """
        nuc_nuc = 0.0
        if include_nuc and deriv is None:
            nuc_nuc = self.nuc_nuc
        # if energy is a parameter
        if self.energy_is_param:
            if not sd is None:
                print('Warning: Cannot specify Slater determinant to compute energy if energy is a parameter')
            # if not derivatized
            if deriv is None:
                return self.params[-1]+nuc_nuc
            # if derivatized
            elif deriv == self.energy_index:
                return 1.0
            else:
                return 0.0
        # if energy is not a parameter
        else:
            # if sd is None
            if sd is None:
                sd = self.pspace[0]
            # if sd is not None
            if type(sd) in [int, type(mpz())]:
                sd = (sd,)
            if not isinstance(sd, (list, tuple)):
                raise TypeError('Unsupported Slater determinant type {0}'.format(type(sd)))
            if not all(type(i) in [int, type(mpz())] for i in sd):
                raise TypeError('List of Slater determinants must all be of type int or gmpy2.mpz')

            # if not derivatized
            if deriv is None:
                elec_energy = sum(self.overlap(i)*self.compute_hamiltonian(i) for i in sd)
                elec_energy /= self.compute_norm(sd=sd)
            # if derivatized
            else:
                olp = np.array([self.overlap(i) for i in sd])
                d_olp = np.array([self.overlap(i, deriv=deriv) for i in sd])
                ham = np.array([self.compute_hamiltonian(i) for i in sd])
                d_ham = np.array([self.compute_hamiltonian(i, deriv=deriv) for i in sd])
                norm = self.compute_norm(sd=sd)
                d_norm = self.compute_norm(sd=sd, deriv=deriv)
                elec_energy = np.sum(d_olp*ham + olp*d_ham)/norm
                elec_energy += np.sum(olp*ham)/(-norm**2)*d_norm
            return elec_energy + nuc_nuc

    @abstractmethod
    def normalize(self):
        pass

    #
    # Objective
    #
    def objective(self, x):
        """ System of nonlinear functions to solve

        The solver likely solves for the root, so the system of equations will be
        rearranged to zero
        ..math::
            f(x_1) - b_1 &= g(x_1)\\
            f(x_2) - b_2 &= g(x_2)\\
            f(x_3) - b_3 &= g(x_3)\\
        and the solver solves for `x` such that the values of the function `g` is minimized

        Parameters
        ----------
        x : 1-index np.ndarray
            The coefficient vector.

        Returns
        -------
        value : np.ndarray(self.nproj,)
            Value of the function `g`

        """
        # Update the coefficient vector
        self.params[:] = x
        # Clear cache
        self.cache = {}
        self.d_cache = {}

        # set reference SD
        ref_sd = self.pspace[0]
        # set energy
        if self.energy_is_param:
            energy = self.params[-1]
            obj = np.empty(self.nproj+2, dtype=self.dtype)
        else:
            energy = self.compute_energy(sd=ref_sd)
            obj = np.empty(self.nproj+1, dtype=self.dtype)

        # <SD|H|Psi> - E<SD|H|Psi> == 0
        for i, sd in enumerate(self.pspace):
            obj[i] = self.compute_hamiltonian(sd) - energy*self.overlap(sd)
        # Add normalization constraint
        obj[-1] = self.compute_norm(sd=ref_sd) - 1.0
        # obj[-1] = self.overlap(ref_sd) - 1.0
        return obj

    def jacobian(self, x):
        # Update the coefficient vector
        self.params[:] = x
        # Clear cache
        self.cache = {}
        self.d_cache = {}

        # set reference SD
        ref_sd = self.pspace[0]

        # set energy
        if self.energy_is_param:
            energy = self.params[-1]
            jac = np.empty((self.nproj+2, self.nparam), dtype=self.dtype)
        else:
            energy = self.compute_energy(sd=ref_sd)
            jac = np.empty((self.nproj+1, self.nparam), dtype=self.dtype)

        for j in range(self.nparam):
            if self.energy_is_param:
                d_energy = 0.0
            else:
                d_energy = self.compute_energy(sd=ref_sd, deriv=j)
            for i, sd in enumerate(self.pspace):
                # <SD|H|Psi> - E<SD|H|Psi> == 0
                if j < self.energy_index:
                    jac[i, j] = (self.compute_hamiltonian(sd, deriv=j)
                                 -energy*self.overlap(sd, deriv=j)-d_energy*self.overlap(sd))
                else:
                    jac[i, j] = self.compute_hamiltonian(sd, deriv=j) - self.overlap(sd)
            # Add normalization constraint
            jac[-1, j] = self.compute_norm(sd=ref_sd, deriv=j)
            # jac[-1, j] = self.overlap(ref_sd, deriv=j)
        return jac
