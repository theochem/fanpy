from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod, abstractproperty

from itertools import combinations, product

import numpy as np

from .wavefunction import Wavefunction
from . import slater


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
    C : np.ndarray(K)
        Coefficient of the Slater determinants

    Private
    -------
    _methods : dict
        Default dimension of projection space
    _energy : float
        Electronic energy

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
    compute_projection
        Generates the appropriate nonlinear equation
    _compute_projection_deriv
        Computes derivative of projection

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
    ):

        super(ProjWavefunction, self).__init__(
            nelec=nelec,
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
        )

        self.assign_params(params=params)
        self.assign_pspace(pspace=pspace)
        self.assign_nproj(nproj=nproj)

    #
    # Solver methods
    #
    def _solve_least_squares(self):
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
            "bounds": self.bounds,
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
                result = least_squares(self.objective, self.x, **options)
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

        nparam = self._nproj_default + 1
        # create a random guess
        if params is None:
            params = np.empty(nparam, dtype=self.dtype)
            scale = 2.0/(nparam ** 2)
            params[:-1] = scale*(np.random.random(nparam - 1) - 0.5)
            if params.dtype == np.complex128:
                params[:-1] = 1j*scale* (np.random.random(nparam - 1) - 0.5)

        if not isinstance(params, np.ndarray):
            raise TypeError("params must be of type {0}".format(np.ndarray))
        elif params.shape != (nparam,):
            raise ValueError("params must be of length nparam ({0})".format(nparam))
        elif params.dtype not in (float, complex, np.float64, np.complex128):
            raise TypeError("params's dtype must be one of {0}".format((float, complex, np.float64, np.complex128)))

        energy = params[-1, ...] # Use the Ellipsis to obtain a view
        self.params = params

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
        self.pspace = tuple(pspace)
        self.cache = {}
        self.d_cache = {}

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
    def compute_energy(self, include_nuc=True, deriv=None):
        """ Returns the energy of the system

        Parameters
        ----------
        sd : int
            Integer that describes the occupation of a Slater determinant as a bitstring
            Slater determinant

        """

        nuc_nuc = self.nuc_nuc if nuc_nuc else 0.0

        if sd is None:
            return self._energy + nuc_nuc

        else:
            # TODO: ADD HAMILTONIANS
            raise NotImplementedError

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
    def compute_overlap(self):
        pass

    @abstractmethod
    def compute_projection(self, sd, deriv=None):
        pass

    @abstractmethod
    def _compute_projection_deriv(self, sd, deriv):
        pass
