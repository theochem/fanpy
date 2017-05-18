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
import numpy as np
from .base_hamiltonian import BaseHamiltonian


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

        Raises
        ------
        TypeError
            If `one_int` and `two_int` do not have the same type
            If `one_int` is not a numpy array or 1- or 2-tuple of numpy arrays
            If `two_int` is not a numpy array or 1- or 3-tuple of numpy arrays
            If the orbital type does not match up with the given integrals
            If `one_int` and `two_int` matrices are not numpy arrays
            If `one_int` and `two_int` matrices do not have the same dtype (must be one of float and
            complex)
            If `one_int` and `two_int` matrices do not have the same dimensionality along each axis.
            If `one_int` is not two dimensional
            If `two_int` is not four dimensional
        ValueError
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
        if type(one_int) != type(two_int):
            raise TypeError('Both the one- and two-electron integrals must have the same type.')
        elif not isinstance(one_int, (list, tuple)) and isinstance(one_int, np.ndarray):
            one_int, two_int = (one_int, ), (two_int, )
            possible_orbtype = ('restricted', 'generalized')
        elif isinstance(one_int, (list, tuple)) and len(one_int) == 1 and len(two_int) == 1:
            possible_orbtype = ('restricted', 'generalized')
        elif isinstance(one_int, (list, tuple)) and len(one_int) == 2 and len(two_int) == 3:
            possible_orbtype = ('unrestricted', )
        else:
            raise TypeError('One- and two-electron integrals must be given '
                            'as a numpy array or a 1-tuple of numpy array for restricted and '
                            ' generalized orbitals, '
                            'or as a 2-tuple of numpy arrays (one-electron integrals) and 3-tuple '
                            'of numpy arrays (two-electron integrals) for unrestricted orbitals.')

        if self.orbtype not in possible_orbtype:
            raise TypeError('The orbital type specified, `{0}`, does not match up with the '
                             'possible orbital types of the given integrals.'.format(self.orbtype))

        for matrix in one_int + two_int:
            if isinstance(matrix, np.ndarray):
                raise TypeError('Integral matrices must be a numpy array.')
            if matrix.dtype != one_int[0].dtype in (float, complex):
                raise TypeError('Integral matrices must consistent dtype and be one of float or '
                                'complex.')
            if not all(i == one_int[0].shape[0] for i in matrix.shape):
                raise TypeError('All of the integral matrices must have the same dimensionality '
                                'along each axis.')
        if not all(len(matrix.shape) == 2 for matrix in one_int):
            raise TypeError('One-electron integral matrices must be two dimensional.')
        if not all(len(matrix.shape) == 4 for matrix in two_int):
            raise TypeError('Two-electron integral matrices must be four dimensional.')

        if self.orbtype == 'generalized' and one_int[0].shape[0] % 2 == 1:
            raise NotImplementedError('Odd number of "spin" orbitals will cause problems when '
                                      ' constructing Slater determinants.')
        # TODO: check that two electron integrals are in physicist's notation (check symmetry)

        self.one_int = one_int
        self.two_int = two_int
