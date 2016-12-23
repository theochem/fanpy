from __future__ import absolute_import, division, print_function

from ..math_tools import binomial
from .doci import DOCI
from ..sd_list import sd_list
from .. import slater


class CIPairs(DOCI):
    """ Configuration Interaction Pairs (DOCI with only one pair excitation)

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
        """ Generates Slater determinants for CI-Pairs wavefunction

        All seniority zero Slater determinants with only electron pair excitations from ground state

        Returns
        -------
        civec : list of ints
            Integer that describes the occupation of a Slater determinant as a bitstring
        """
        return sd_list(self.nelec, self.nspatial, exc_orders=[2], seniority=0)

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

        Raises
        ------
        TypeError
            If excitation level is not an integer
        ValueError
            If excitation level is negative
        """
        if not isinstance(exc_lvl, int):
            raise TypeError('Excitation level must be an integer')
        if exc_lvl < 0:
            raise ValueError('Excitation level cannot be negative')

        # dictionary of slater determinant to coefficient
        dict_sd_coeff = {sd: coeff for sd, coeff in zip(self.civec,
                                                        self.sd_coeffs[:, exc_lvl].flat)}
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
                gem_coeffs[i, a] = dict_sd_coeff[sd_exc]
        return gem_coeffs
