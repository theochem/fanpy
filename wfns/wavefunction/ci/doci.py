""" DOCI wavefunction (seniority zero)

Seniority zero means that only Slater determinants constructed from alpha-beta paired spatial
orbitals
"""
from __future__ import absolute_import, division, print_function
from .ci_wavefunction import CIWavefunction

__all__ = []


class DOCI(CIWavefunction):
    """Doubly Occupied Configuration Interaction (DOCI) Wavefunction.

    CI wavefunction constructed from all of the seniority zero Slater determiannts within the given
    basis.

    Attributes
    ----------
    nelec : int
        Number of electrons
    nspin : int
        Number of spin orbitals (alpha and beta)
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction
    params : np.ndarray
        Parameters of the wavefunction
    _spin : float
        Total spin of each Slater determinant
        :math:`\frac{1}{2}(N_\alpha - N_\beta)`
        Default is no spin (all spins possible)
    _seniority : int
        Number of unpaired electrons in each Slater determinant
    sd_vec : tuple of int
        List of Slater determinants used to construct the CI wavefunction
    dict_sd_index : dictionary of int to int
        Dictionary from Slater determinant to its index in sd_vec

    Properties
    ----------
    nspatial : int
        Number of spatial orbitals
    spin : float, None
        Spin of the wavefunction
        Spin is zero (singlet)
    seniority : int, None
        Seniority (number of unpaired electrons) of the wavefunction
        Seniority is zero
    template_params : np.ndarray
        Template of the wavefunction parameters
        Depends on the attributes given
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, params=None, sd_vec=None, spin=None, seniority=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_nspin(self, nspin)
        Assigns the number of spin orbitals
    assign_params(self, params)
        Assigns the parameters of the wavefunction
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_spin(self, spin=None)
        Assigns the spin of the wavefunction
    assign_seniority(self, seniority=None)
        Assigns the seniority of the wavefunction
    assign_sd_vec(self, sd_vec=None)
        Assigns the tuple of Slater determinants used in the CI wavefunction
    get_overlap(self, sd, deriv=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    """
    def assign_nelec(self, nelec):
        """Set the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons

        Raises
        ------
        TypeError
            If number of electrons is not an integer or long
        ValueError
            If number of electrons is not a positive number
            If number of electrons is not even
        """
        super().assign_nelec(nelec)
        if self.nelec % 2 != 0:
            raise ValueError('`nelec` must be an even number')

    def assign_spin(self, spin=None):
        """Set the spin of each Slater determinant.

        :math:`\frac{1}{2}(N_\alpha - N_\beta)`

        Parameters
        ----------
        spin : float
            Spin of each Slater determinant
            Default is no spin (all spins possible)

        Raises
        ------
        TypeError
            If the spin is not an integer, float, or None
        ValueError
            If the spin is not an integral multiple of 0.5
        ValueError
            If spin is not zero (singlet)
        """
        if spin is None:
            spin = 0
        super().assign_spin(spin)
        if self.spin != 0:
            raise ValueError('DOCI wavefunction can only be singlet')

    def assign_seniority(self, seniority=None):
        """Set the seniority of each Slater determinant of the DOCI wavefunction.

        :math:`\frac{1}{2}(N_\alpha - N_\beta)`

        Parameters
        ----------
        seniority : float
            Seniority of each Slater determinant
            Default is seniority zero

        Raises
        ------
        TypeError
            If the seniority is not an integer, float, or None
        ValueError
            If the seniority is a negative integer
            If the seniority is not zero
        """
        if seniority is None:
            seniority = 0
        super().assign_seniority(seniority)
        if self.seniority != 0:
            raise ValueError('DOCI wavefunction can only be seniority 0')
