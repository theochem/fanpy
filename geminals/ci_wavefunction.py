from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractproperty, abstractmethod

from itertools import combinations, product

import numpy as np
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
    nci : int
        Number of (user-specified) Slater determinants

    Private
    -------
    _methods : dict
        Default dimension of projection space
    _energy : float
        Electronic energy

    Abstract Properties
    -------------------
    _nci : int
        Total number of (default) Slater determinants

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
    def _nci(self):
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

    def dict_sd_coeff(self, exc_lvl=0):
        """ Dictionary of the coefficient

        Parameters
        ----------
        exc_lvl : int
            Excitation level of the wavefunction
            0 is the ground state wavefunction
            1 is the first excited wavefunction

        Returns
        -------
        Dictionary of SD to coefficient
        """
        if not isinstance(exc_lvl, int):
            raise TypeError('Excitation level must be an integer')
        if exc_lvl < 0:
            raise ValueError('Excitation level cannot be negative')
        return {sd:coeff for sd,coeff in zip(self.civec, self.sd_coeffs[:, exc_lvl].flat)}

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
            nci=None,
            civec=None,
    ):

        super(CIWavefunction, self).__init__(
            nelec=nelec,
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
        )
        self.assign_nci(nci=nci)
        self.assign_civec(civec=civec)
        self.sd_coeffs = np.zeros([self.nci, self.nci])

    #
    # Solver methods
    #

    def _solve_eigh(self, which='SA', **kwargs):
        """ Solves for the ground state using eigenvalue decomposition
        """

        ci_matrix = self.compute_ci_matrix()
        result = eigh(ci_matrix, **kwargs)
        del ci_matrix

        # NOTE: overwrites last sd_coeffs
        self.sd_coeffs = result[1]
        self._energy = result[0]

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
        #FIXME: cyclic dependence on civec
        if nci is None:
            nci = self._nci
        if not isinstance(nci, int):
            raise TypeError('Number of determinants must be an integer')
        self.nci = nci

    def assign_civec(self, civec=None):
        """ Sets the Slater determinants used in the wavefunction

        Parameters
        ----------
        civec : iterable of int
            List of Slater determinants (in the form of integers that describe
            the occupation as a bitstring)
        """
        #FIXME: cyclic dependence on nci
        if civec is None:
            civec = self.compute_civec()
        if not isinstance(civec, (list, tuple)):
            raise TypeError("civec must be a list or a tuple")
        self.civec = tuple(civec)

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
        civec : list of ints
            Integer that describes the occupation of a Slater determinant as a bitstring

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

    def compute_energy(self, include_nuc=True, exc_lvl=0):
        """ Returns the energy of the system

        Parameters
        ----------
        include_nuc : bool
            Flag to include nuclear nuclear repulsion
        exc_lvl : int
            Excitation level of the wavefunction
            0 is the ground state wavefunction
            1 is the first excited wavefunction

        Returns
        -------
        energy : float
            Total energy if include_nuc is True
            Electronic energy if include_nuc is False
        """
        if not isinstance(exc_lvl, int):
            raise TypeError('Excitation level must be an integer')
        if exc_lvl < 0:
            raise ValueError('Excitation level cannot be negative')
        nuc_nuc = self.nuc_nuc if include_nuc else 0.0
        return self._energy[exc_lvl] + nuc_nuc

    def to_other(self, Other, exc_lvl=0):
        """ Converts CIWavefunction to ProjWavefunction as best as possible

        Parameters
        ----------
        Other : ProjWavefunction class
            Class of the wavefunction to turn into
        exc_lvl : int
            Excitation level of the wavefunction
            0 is the ground state wavefunction
            1 is the first excited wavefunction

        Returns
        -------
        new_instance : Other instance
            Instance of the specified wavefunction with parameters/coefficients
            that tries to resemble self
        """
        new_instance = Other(nelec=self.nelec,
                             H=self.H,
                             G=self.G,
                             dtype=self.dtype,
                             nuc_nuc=self.nuc_nuc,
                             orbtype=self.orbtype,
                             nproj=None,
                             x=None,)
        sd_coeff = dict_sd_coeff(self, exc_lvl=0)
        def objective(x_vec):
            """ Function to minimize

            We will find the root of this function, which means that we find the
            x_vec such that the objective gives back the smallest vector

            parameters
            ----------
            x_vec : np.ndarray(K,)
                One dimensional numpy array

            Returns
            -------
            val : np.ndarray(K,)
                One dimensional numpy array
            """
            val = np.empty(self.nci, dtype=self.dtype)
            for i, sd in enumerate(self.civec):
                val[i] =  new_instance.compute_overlap(sd) - sd_coeff[sd]
            return val
        result = least_squares(self.objective, new_instance.x)
        new_instance.assign_civec(result.x)
        return new_instance
