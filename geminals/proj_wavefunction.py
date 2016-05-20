from __future__ import absolute_import, division, print_function

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
    compute_civec
        Generates a list of states to project onto
    compute_overlap
        Computes the overlap of a Slater deteriminant with the wavefunction
    compute_projection
        Generates the appropriate nonlinear equation
    _compute_projection_deriv
        Computes derivative of projection

    """
    # FIXME: turn C into property and have a better attribute name

    #
    # Default attribute values
    #

    @abstractproperty
    def _nproj_default(self):
        pass

    @property
    def _methods(self):
        raise NotImplementedError

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
        naproj=None,
        nrproj=None,
        # Arguments handled by FullCI class
        x=None,
        civec=None,
    ):

        super(FullCI, self).__init__(
            nelec=nelec,
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
        )

        self.assign_x(x=x)
        self.assign_civec(civec=civec)
        self.assign_nproj(nproj=nproj, naproj=naproj, nrproj=nrproj)

    #
    # Assignment methods
    #

    def assign_x(self, x=None):

        nparam = self._nproj_default + 1

        if x is not None:
            if not isinstance(x, np.ndarray):
                raise TypeError("x must be of type {0}".format(np.ndarray))
            elif x.shape != (nparam,):
                raise ValueError("x must be of length nparam ({0})".format(nparam))
            elif x.dtype not in (float, complex, np.float64, np.complex128):
                raise TypeError("x's dtype must be one of {0}".format((float, complex, np.float64, np.complex128)))

        else:
            x = np.empty(nparam, dtype=self.dtype)
            x[:-1] = 2.0 * (np.random.random(nparam - 1) - 0.5) / (nparam ** 2)
            if x.dtype == np.complex128:
                x[:-1] = 2.0j * (np.random.random(nparam - 1) - 0.5) / (nparam ** 2)

        C = x[:-1]
        C[0] = 1.0 # Intermediate normalization
        energy = x[-1, ...] # Use the Ellipsis to obtain a view

        self.nparam = nparam
        self.x = x
        self.C = C
        self._energy = energy

    def assign_civec(self, civec=None):
        #FIXME: code repeated in ci_wavefunction.py

        if civec is not None:
            if not isinstance(civec, list):
                raise TypeError("civec must be of type {0}".format(list))
        else:
            civec = self.compute_civec()

        self.civec = civec
        self.cache = {}
        self.d_cache = {}

    def assign_nproj(self, nproj=None, naproj=None, nrproj=None):
        """ Sets number of projection determinants

        Parameters
        ----------
        nproj : int
            Number of projection states
        naproj : FIXME
        nrproj : FIXME
        """

        #NOTE: there is an order to the assignme

        if nproj is None:
            nproj = self._nproj_default

        elif nproj is not None:
            if not isinstance(nproj, int):
                raise TypeError("nproj must be of type {0}".format(int))

            if sum(not i is None for i in (naproj, nrproj)) <= 1:
                raise ValueError("At most one of (naproj, nrproj) should be specified")

            if naproj is not None:
                if not isinstance(naproj, int):
                    raise TypeError("naproj must be of type {0}".format(int))
                else:
                    nproj += naproj

            if nrproj is not None:
                if not isinstance(nrproj, float):
                    raise TypeError("nrproj must be of type {0}".format(float))
                elif nrproj < 1.0:
                    raise ValueError("nrproj must be greater than 1.0")
                else:
                    nproj = np.round(nrproj * nproj)

        self.nproj = nproj

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
    #
    # Computation methods
    #

    @abstractmethod
    def compute_civec(self):
        """ Generates projection space
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
