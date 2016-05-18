from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractproperty, abstractmethod

from itertools import combinations, product

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

from wavefunction import Wavefunction
import slater


class CIWavefunction(Wavefunction):
    """ Wavefunction expressed as a linear combination of Slater determinants

    Contains the necessary information to variationally solve the CI wavefunction

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

    Private
    -------
    _methods : dict
        Default dimension of projection space
    _energy : float
        Electronic energy

    Abstract Properties
    -------------------
    _nci_default : int
        Number of Slater determinants

    Abstract Methods
    ----------------
    compute_civec
        Generates a list of Slater determinants
    compute_ci_matrix
        Generates the Hamiltonian matrix of the Slater determinants
    """
    # FIXME: turn C into property and have a better attribute name
    __metaclass__ = ABCMeta

    #
    # Default attribute values
    #
    @abstractproperty
    def _nci_default(self):
        """ Default number of configurations

        """
        pass

    @property
    def _methods(self):
        """ Dictionary of methods for solving the wavefunction

        Returns
        -------
        methods : dict
            "default" -> eigenvalue decomposition
        """

        return {"default": self._solve_eigh}

    @property
    def C(self):
        """ Coefficients for the Slater determinants
        """
        return self.sd_coeffs

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
            nparticle=None,
            odd_nelec=None,
            # Arguments handled by FullCI class
            nci=None,
            civec=None,
    ):

        super(CIWavefunction, self).__init__(
            nelec=nelec,
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
            nparticle=nparticle,
            odd_nelec=odd_nelec,
        )
        self.assign_nci(nci=nci)
        self.assign_civec(civec=civec)
        self.sd_coeffs = np.zeros(len(self.civec))
        self._energy = 0.0

    #
    # Solver methods
    #

    def _solve_eigh(self, which='SA', **kwargs):
        """ Solves for the ground state using eigenvalue decomposition
        """

        ci_matrix = self.compute_ci_matrix()
        result = eigsh(ci_matrix, 1, which=which, **kwargs)
        # result = eigh(ci_matrix, **kwargs)
        del ci_matrix

        self.C[...] = result[1][:, 0]
        self._energy = result[0][0]

        return result

    #
    # Assignment methods
    #
    def assign_nci(self, nci=None):
        """ Sets number of projection determinants

        Parameters
        ----------
        nci : int
            Number of configuratons
        """

        #NOTE: there is an order to the assignme

        if nci is None:
            nci = self._nci_default
        self.nci = nci

    def assign_civec(self, civec=None):

        if civec is not None:
            if not isinstance(civec, list):
                raise TypeError("civec must be of type {0}".format(list))
        else:
            civec = self.compute_civec()

        self.civec = civec
        self.cache = {}
        self.d_cache = {}

    #
    # Computation methods
    #
    @abstractmethod
    def compute_civec(self):
        """ Generates Slater determinants

        Number of Slater determinants generated is determined strictly by the size of the
        projection space (self.nci). First row corresponds to the ground state SD, then
        the next few rows are the first excited, then the second excited, etc

        Returns
        -------
        civec : np.ndarray(nci, nspin)
            Boolean array that describes the occupations of the Slater determinants
            Each row is a Slater determinant
            Each column is the index of the spin orbital

        """
        pass


    @abstractmethod
    def compute_ci_matrix(self):
        """ Returns Hamiltonian matrix in the Slater determinant basis

        ..math::
            H_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

        Returns
        -------
        matrix : np.ndarray(K, K)
        """
        pass
