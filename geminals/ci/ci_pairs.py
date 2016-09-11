from __future__ import absolute_import, division, print_function

from ..math_tools import binomial
from .ci_wavefunction import CIWavefunction
from ..sd_list import doci_sd_list
from .ci_matrix import doci_matrix
from .. import slater


class CIPairs(CIWavefunction):
    """ Configuration Interaction Pairs (DOCI with only one pair excitation)

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
    _nci : int
        Number of Slater determinants

    Methods
    -------
    compute_civec
        Generates a list of Slater determinants
    compute_ci_matrix
        Generates the Hamiltonian matrix of the Slater determinants
    """
    @property
    def _nci(self):
        """ Total number of configurations
        """
        num_singles = binomial(self.npair, 1) * binomial(self.nspatial - self.npair, 1)
        return 1 + num_singles

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
        return doci_sd_list(self, self.nci, [1])

    def compute_ci_matrix(self):
        """ Returns Hamiltonian matrix in the arbitrary Slater (orthogonal) determinant basis

        ..math::
            H_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

        Returns
        -------
        matrix : np.ndarray(K, K)
        """
        return doci_matrix(self, self.orb_type)

    def to_ap1rog(self, exc_lvl=0):
        """ Returns geminal matrix given then converged CIPairs wavefunction coefficients

        Parameters
        ----------
        exc_lvl : int
            Excitation level of the wavefunction
            0 is the ground state wavefunction
            1 is the first excited wavefunction

        Returns
        -------
        gem_coeffs : np.ndarray(self.npair, self.nspatial-self.npair)
            AP1roG geminal coefficients
        """
        # dictionary of slater determinant to coefficient
        sd_coeffs = self.dict_sd_coeff(exc_lvl=exc_lvl)
        # ground state SD
        ground = slater.ground(nelec, 2 * nspatial)
        # fill empty geminal coefficient
        gem_coeffs = np.zeros((self.npair, self.nspatial - self.npair))
        for i in range(self.npair):
            for a in range(self.npair, self.nspatial):
                # because first self.npair columns are removed
                a -= self.npair
                # excite slater determinant
                sd_exc = slater.excite(ground, i, a)
                # set geminal coefficient
                gem_coeffs[i, a] = sd_coeffs[sd_exc]
        return gem_coeffs
