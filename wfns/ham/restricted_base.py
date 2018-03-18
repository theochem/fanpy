"""Base Hamiltonian for restricted orbitals."""
from wfns.ham.generalized_base import BaseGeneralizedHamiltonian


class BaseRestrictedHamiltonian(BaseGeneralizedHamiltonian):
    """Base class for Hamiltonian with restricted orbitals.

    Attributes
    ----------
    energy_nuc_nuc : float
        Nuclear-nuclear repulsion energy.
    one_int : np.ndarray(K, K)
        One-electron integrals.
    two_int : np.ndarray(K, K, K, K)
        Two-electron integrals.

    Properties
    ----------
    dtype : {np.float64, np.complex128}
        Data type of the Hamiltonian.
    nspin : int
        Number of spin orbitals.

    Methods
    -------
    __init__(self, one_int, two_int, orbtype=None, energy_nuc_nuc=None)
        Initialize the Hamiltonian.
    assign_energy_nuc_nuc(self, energy_nuc_nuc=None)
        Assigns the nuclear nuclear repulsion.
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.

    Abstract Methods
    ----------------
    integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """
    @property
    def nspin(self):
        """Return the number of spin orbitals.

        Returns
        -------
        nspin : int
            Number of spin orbitals.

        """
        return 2 * self.one_int.shape[0]
