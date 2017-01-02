""" DOCI wavefunction (seniority zero)

Seniority zero means that only Slater determinants constructed from alpha-beta paired spatial
orbitals
"""
from __future__ import absolute_import, division, print_function
from .ci_wavefunction import CIWavefunction


class DOCI(CIWavefunction):
    """ Doubly Occupied Configuration Interaction Wavefunction

    CI wavefunction constructed from all seniority zero Slater determinants (alpha-beta paired
    spatial orbitals)

    Attributes
    ----------
    nelec : int
        Number of electrons
    one_int : 1- or 2-tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        1-tuple for spatial (restricted) and generalized orbitals
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
    two_int : 1- or 3-tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        In physicist's notation
        1-tuple for spatial (restricted) and generalized orbitals
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components)
    dtype : {np.float64, np.complex128}
        Numpy data type
    nuc_nuc : float
        Nuclear-nuclear repulsion energy
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    dict_exc_index : dict from int to int
        Dictionary from the excitation order to the column index of the coefficient matrix
    spin : float
        Total spin of the wavefunction
        Default is no spin (all spins possible)
        0 is singlet, 0.5 and -0.5 are doublets, 1 and -1 are triplets, etc
        Positive spin means that there are more alpha orbitals than beta orbitals
        Negative spin means that there are more beta orbitals than alpha orbitals
    civec : tuple of int
        List of Slater determinants used to construct the CI wavefunction

    Properties
    ----------
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals

    Methods
    -------
    __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_nuc_nuc(self, nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_integrals(self, one_int, two_int, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
    assign_excs(self, excs=None)
        Assigns excitations to include in the calculation
    assign_spin(self, spin=None)
        Assigns the spin of the wavefunction
    assign_civec(self, civec=None)
        Assigns the tuple of Slater determinants used in the CI wavefunction
    get_energy(self, include_nuc=True, exc_lvl=0)
        Gets the energy of the CI wavefunction
    compute_density_matrix(self, exc_lvl=0, is_chemist_notation=False, val_threshold=0)
        Constructs the one and two electron density matrices for the given excitation level
    to_proj(self, Other, exc_lvl=0)
        Try to convert the CI wavefunction into the appropriate Projected Wavefunction
    compute_ci_matrix(self)
        Returns CI Hamiltonian matrix in the Slater determinant basis
    generate_civec(self)
        Returns a list of seniority 0 Slater determinants
    """
    def assign_nelec(self, nelec):
        """ Sets the number of electrons

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
        """
        # NOTE: use super?
        if not isinstance(nelec, (int, long)):
            raise TypeError('`nelec` must be of type {0}'.format(int))
        elif nelec <= 0:
            raise ValueError('`nelec` must be a positive integer')
        elif nelec % 2 != 0:
            raise ValueError('`nelec` must be an even number')
        self.nelec = nelec


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

        Raises
        ------
        ValueError
            If given spin is not 0
        """
        # NOTE: use super?
        if spin is None or spin == 0:
            self.spin = 0
        else:
            raise ValueError('DOCI wavefunction can only be singlet')


    def assign_seniority(self, seniority=None):
        """ Sets the seniority of the wavefunction

        Parameters
        ----------
        seniority : int
            Seniority of the wavefunction
            Default is no seniority (all seniority possible)

        Raises
        ------
        TypeError
            If the seniority is not a float or None
        """
        # NOTE: use super?
        if seniority is None or seniority == 0:
            self.seniority = 0
        else:
            raise ValueError('DOCI wavefunction can only be seniority 0')
