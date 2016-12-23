from __future__ import absolute_import, division, print_function

from .ci_wavefunction import CIWavefunction
from ..math_tools import binomial
from ..sd_list import sd_list
from .ci_matrix import ci_matrix


class DOCI(CIWavefunction):
    """ Doubly Occupied Configuration Interaction Wavefunction

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
    def assign_spin(self, spin=None):
        """ Sets the spin of the projection determinants

        Parameters
        ----------
        spin : int
            Total spin of the wavefunction
            Default is no spin (all spins possible)
            0 is singlet, 0.5 and -0.5 are doublets, 1 and -1 are triplets, etc
            Positive spin means that there are more alpha orbitals than beta orbitals
            Negative spin means that there are more beta orbitals than alpha orbitals
        """
        if spin is None:
            spin = 0
        if not isinstance(spin, int):
            raise TypeError('Invalid spin of the wavefunction')
        if spin > 0:
            raise ValueError('DOCI wavefunction can only be singlet')
        self.spin = spin

    def generate_civec(self):
        """ Generates Slater determinants of DOCI

        All seniority zero Slater determinants

        Returns
        -------
        civec : list of ints
            Integer that describes the occupation of a Slater determinant as a bitstring
        """
        return sd_list(self.nelec, self.nspatial, seniority=0)

    def compute_ci_matrix(self):
        """ Returns Hamiltonian matrix in the arbitrary Slater (orthogonal) determinant basis

        ..math::
            H_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

        Returns
        -------
        matrix : np.ndarray(K, K)
        """
        return ci_matrix(self.one_int, self.two_int, self.civec, self.dtype, self.orbtype)
