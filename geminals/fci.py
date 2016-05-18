from __future__ import absolute_import, division, print_function

from ci_wavefunction import CIWavefunction
from sd_list import ci_sd_list
from ci_matrix import ci_matrix
from math_tools import binomial

class FCI(CIWavefunction):
    """ Full Configuration Interaction Wavefunction

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
    _nci_default : int
        Number of Slater determinants

    Methods
    -------
    compute_civec
        Generates a list of Slater determinants
    compute_ci_matrix
        Generates the Hamiltonian matrix of the Slater determinants
    """
    @property
    def _nci_default(self):
        """ Total number of configurations

        """
        return binomial(self.nspin, self.nelec)

    def compute_civec(self):
        """ Generates Slater determinants

        Number of Slater determinants is limited by num_limit. First Slater determinant is the ground
        state, next are the first excitations from exc_orders, then second excitation from
        exc_orders, etc

        Returns
        -------
        civec : list of ints
            Integer that describes the occupation of a Slater determinant as a bitstring
        """
        return ci_sd_list(self, self._nci_default)

    def compute_ci_matrix(self):
        """ Returns Hamiltonian matrix in the arbitrary Slater (orthogonal) determinant basis

        ..math::
            H_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

        Returns
        -------
        matrix : np.ndarray(K, K)
        """
        return ci_matrix(self, 'restricted')
