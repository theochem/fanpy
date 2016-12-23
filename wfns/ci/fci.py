from __future__ import absolute_import, division, print_function

from ..math_tools import binomial
from .ci_wavefunction import CIWavefunction
from ..sd_list import sd_list
from .ci_matrix import ci_matrix


class FCI(CIWavefunction):
    """ Full Configuration Interaction Wavefunction

    Contains the necessary information to variationally solve the CI wavefunction

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    one_int : np.ndarray(K,K)
        One electron integrals for the spatial orbitals
    two_int : np.ndarray(K,K,K,K)
        Two electron integrals for the spatial orbitals
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
    _nci : int
        Number of Slater determinants

    Methods
    -------
    generate_civec
        Generates a list of Slater determinants
    compute_ci_matrix
        Generates the Hamiltonian matrix of the Slater determinants
    """
    def generate_civec(self):
        """ Generates Slater determinants for FCI

        All Slater determinants

        Returns
        -------
        civec : list of ints
            Integer that describes the occupation of a Slater determinant as a bitstring
        """
        return sd_list(self.nelec, self.nspatial, spin=self.spin)

    def compute_ci_matrix(self):
        """ Returns Hamiltonian matrix in the arbitrary Slater (orthogonal) determinant basis

        ..math::
            H_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

        Returns
        -------
        matrix : np.ndarray(K, K)
        """
        return ci_matrix(self.one_int, self.two_int, self.civec, self.dtype, self.orbtype)
