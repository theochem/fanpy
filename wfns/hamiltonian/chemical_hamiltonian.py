"""Hamiltonian object that interacts with the wavefunction.

..math::
    \braket{\Phi | H | \Psi}

Functions
---------
hamiltonian(wfn, slater_d, orbtype, deriv=None)
    Computes expectation value of the wavefunction with Hamiltonian projected against a Slater
    determinant
sen0_hamiltonian(wfn, slater_d, orbtype, deriv=None)
    Computes expectation value of the seniority zero wavefunction with Hamiltonian projected against
    a Slater determinant
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from .base_hamiltonian import BaseHamiltonian
from ..backend.integrals import OneElectronIntegrals, TwoElectronIntegrals


class ChemicalHamiltonian(BaseHamiltonian):
    """Hamiltonian used for a typical chemical system.

    ..math::
        \hat{H} &= \\
        &= \sum_{ij} h_{ij} a^\dagger_i a_j + \sum_{ijkl} g_{ijkl} a^\dagger_i a^\dagger_j a_k a_l\\

    Attributes
    ----------
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used.
    energy_nuc_nuc : float
        Nuclear-nuclear repulsion energy
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

    Properties
    ----------
    dtype
        Data type of the integrals.

    Methods
    -------
    assign_orbtype(self, dtype)
        Assigns the orbital type.
    assign_energy_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_integrals(self, one_int, two_int, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
    """

    def __init__(self, one_int, two_int, orbtype=None, energy_nuc_nuc=None):
        """Initialize the Hamiltonian.

        Parameters
        ----------
        one_int : np.ndarray(K,K), 1- or 2-tuple np.ndarray(K,K)
            One-electron integrals
            If orbitals are spatial or generalized, then integrals are given as np.ndarray or
            1-tuple of np.ndarray
            If orbitals are unretricted, then integrals are 2-tuple of np.ndarray
        two_int : np.ndarray(K,K,K,K), 1- or 3-tuple np.ndarray(K,K,K,K)
            If orbitals are spatial or generalized, then integrals are given as np.ndarray or
            1-tuple of np.ndarray
            If orbitals are unretricted, then integrals are 3-tuple of np.ndarray
        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbitals used.
            Default is `'restricted'`
        energy_nuc_nuc : {float, None}
            Nuclear nuclear repulsion energy
            Default is `0.0`

        """
        self.assign_orbtype(orbtype=orbtype)
        self.assign_energy_nuc_nuc(energy_nuc_nuc)
        self.assign_integrals(one_int, two_int)

    @property
    def dtype(self):
        """Return the data type of the integrals."""
        return self.one_int[0].dtype

    #FIXME: getter/setter is not used b/c assign_integrals is a little complicated.
    def assign_orbtype(self, orbtype=None):
        """Assign the orbital type.

        Parameters
        ----------
        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbitals used.
            Default is `'restricted'`

        Raises
        ------
        ValueError
            If orbtype is not one of ['restricted', 'unrestricted', 'generalized']

        Note
        ----
        Should be executed before assign_integrals.
        """
        if orbtype is None:
            orbtype = 'restricted'
        if orbtype not in ['restricted', 'unrestricted', 'generalized']:
            raise TypeError("Orbital type must be one of 'restricted', 'unrestricted', "
                            "and 'generalized'.")
        self.orbtype = orbtype

    def assign_integrals(self, one_int, two_int):
        """Assign the one- and two-electron integrals.

        Parameters
        ----------
        one_int : {np.ndarray(K,K), 1- or 2-tuple np.ndarray(K,K), OneElectronIntegrals}
            One electron integrals for restricted, unrestricted, or generalized orbitals
            1-tuple for spatial (restricted) and generalized orbitals
            2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
        two_int : {np.ndarray(K,K), 1- or 3-tuple np.ndarray(K,K), TwoElectronIntegrals}
            Two electron integrals for restricted, unrestricted, or generalized orbitals
            In physicist's notation
            1-tuple for spatial (restricted) and generalized orbitals
            3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
            beta-beta-beta-beta components)

        Raises
        ------
        TypeError
            If integrals are not provided as a numpy array or a tuple of numpy arrays
            If integral matrices do not have dtype of `float` or `complex`
            If integral matrices do not have the same dtype
            If the number of one-electron integral matrices is not 1 or 2
            If one-electron integral matrices are not two dimensional
            If one-electron integral matrices are not square
            If one-electron integrals (for the unrestricted orbitals) has different number of alpha
            and beta orbitals
            If the number of two-electron integral matrices is not 1 or 3
            If two-electron integral matrices are not four dimensional
            If two-electron integral matrices does not have same dimensionality in all axis
            If two-electron integrals (for the unrestricted orbitals) has different number of alpha
            and beta orbitals
            If one- and two-electron integrals do not have the same number of orbitals
            If one- and two-electron integrals do not correspond to the same orbital type (i.e.
            restricted, unrestricted, generalized).
            If orbital type of the integrals do not match with the given orbital type
        NotImplementedError
            If generalized orbitals and odd number of spin orbitals

        Note
        ----
        Should be executed after assign_orbtype.
        Depending on the orbital type, the form of the integrals can vary:
            Restricted Orbitals
                one_int can be a np.ndarray or a 1-tuple of np.ndarray
                two_int can be a np.ndarray or a 1-tuple of np.ndarray
                orbtype = 'restricted'
            Unrestricted Orbitals
                one_int can be a 2-tuple of np.ndarray
                two_int can be a 3-tuple of np.ndarray
                orbtype = 'unrestricted'
            Generalized Orbitals
                one_int can be a np.ndarray or a 1-tuple of np.ndarray
                two_int can be a np.ndarray or a 1-tuple of np.ndarray
                orbtype = 'generalized'
        """
        one_int = OneElectronIntegrals(one_int)
        two_int = TwoElectronIntegrals(two_int)
        if one_int.num_orbs != two_int.num_orbs:
            raise TypeError('One- and two-electron integrals do not have the same number of '
                            'orbitals.')
        if one_int.possible_orbtypes != two_int.possible_orbtypes:
            raise TypeError('One- and two-electron integrals do not correspond to the same orbital '
                            'type (i.e. restricted, unrestricted, generalized).')
        if self.orbtype not in one_int.possible_orbtypes:
            raise TypeError('Orbital type of the integrals do not match with the given orbital '
                            'type, {0}'.format(self.orbtype))
        if self.orbtype == 'generalized' and one_int[0].shape[0] % 2 == 1:
            raise NotImplementedError('Odd number of "spin" orbitals will cause problems when '
                                      ' constructing Slater determinants.')

        self.one_int = one_int
        self.two_int = two_int
