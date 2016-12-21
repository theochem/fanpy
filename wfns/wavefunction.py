""" Parent class of the wavefunctions

Contains information that all wavefunctions should have (probably?)
"""
from __future__ import absolute_import, division, print_function
import numpy as np


class Wavefunction(object):
    """ Wavefunction (abstract) class

    Contains the necessary information to projectively solve the wavefunction

    Attributes
    ----------
    nelec : int
        Number of electrons
    H : 1- or 2-tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        1-tuple for spatial (restricted) and generalized orbitals
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
    G : 1- or 3-tuple np.ndarray(K,K)
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

    Properties
    ----------
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals

    Method
    ------
    __init__(self, nelec, H, G, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_integrals(self, H, G, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
    """
    def __init__(self, nelec, H, G, dtype=None, nuc_nuc=None, orbtype=None):
        """ Initializes a wavefunction

        Parameters
        ----------
        nelec : int
            Number of electrons

        H : np.ndarray(K,K), 1- or 2-tuple np.ndarray(K,K)
            One electron integrals
            For spatial and generalized orbitals, np.ndarray or 1-tuple of np.ndarray
            For unretricted spin orbitals, 2-tuple of np.ndarray

        G : np.ndarray(K,K,K,K), 1- or 3-tuple np.ndarray(K,K,K,K)
            For spatial and generalized orbitals, np.ndarray or 1-tuple of np.ndarray
            For unrestricted orbitals, 3-tuple of np.ndarray

        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type
            Default is `np.float64`

        nuc_nuc : {float, None}
            Nuclear nuclear repulsion value
            Default is `0.0`

        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbital used in obtaining the one-electron and two-electron integrals
            Default is `'restricted'`

        """
        # TODO: Implement loading H and G from various file formats
        self.assign_nelec(nelec)
        self.assign_dtype(dtype)
        self.assign_nuc_nuc(nuc_nuc)
        self.assign_integrals(H, G, orbtype=orbtype)


    @property
    def nspin(self):
        """ Number of spin orbitals
        """
        if self.orbtype in ['restricted', 'unrestricted']:
            return 2 * self.H[0].shape[0]
        elif self.orbtype == 'generalized':
            return self.H[0].shape[0]


    @property
    def nspatial(self):
        """ Number of spatial orbitals
        """
        return self.nspin // 2


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
        if not isinstance(nelec, (int, long)):
            raise TypeError("nelec must be of type {0}".format(int))
        elif nelec <= 0:
            raise ValueError("nelec must be a positive integer")
        self.nelec = nelec


    def assign_dtype(self, dtype):
        """ Sets the data type of the parameters

        Parameters
        ----------
        dtype : {float, complex, np.float64, np.complex128}
            Numpy data type
            If None then set to np.float64

        Raises
        ------
        TypeError
            If dtype is not one of float, complex, np.float64, np.complex128
        """

        if dtype is None:
            dtype = np.float64
        elif dtype not in (float, complex, np.float64, np.complex128):
            raise TypeError("dtype must be one of {0}".format((float, complex,
                                                               np.float64, np.complex128)))
        self.dtype = dtype


    def assign_nuc_nuc(self, nuc_nuc=None):
        """ Sets the nuclear nuclear repulsion value

        Parameters
        ----------
        nuc_nuc : float
            Nuclear nuclear repulsion value
            If None then set to 0.0

        Raises
        ------
        TypeError
            If nuclear nuclear repulsion is not a float
        """
        if nuc_nuc is None:
            nuc_nuc = 0.0
        elif not isinstance(nuc_nuc, (float, np.float64)):
            raise TypeError("nuc_nuc integral must be one of {0}".format((float, np.float64)))
        self.nuc_nuc = nuc_nuc


    def assign_integrals(self, H, G, orbtype=None):
        """ Sets one electron and two electron integrals and the orbital type

        Parameters
        ----------
        H : 1- or 2-tuple np.ndarray(K,K)
            One electron integrals for restricted, unrestricted, or generalized orbitals
            1-tuple for spatial (restricted) and generalized orbitals
            2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
        G : 1- or 3-tuple np.ndarray(K,K)
            Two electron integrals for restricted, unrestricted, or generalized orbitals
            In physicist's notation
            1-tuple for spatial (restricted) and generalized orbitals
            3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
            beta-beta-beta-beta components)
        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbital used in obtaining the one-electron and two-electron integrals
            Default is `'restricted'`

        Raises
        ------
        TypeError:
            If H and G are not a numpy array or a tuple of numpy arrays
            If orbital type from H and orbital type from G are not the same
            If orbital type is inconsistent with the integrals given
            If H and G are tuples and its elements are not numpy arrays
            If H and G are tuples and numpy arrays do not have the consistent shapes
            If H and G are tuples and wavefunction data type is float and numpy arrays' data type is
            not float
            If H and G are tuples and wavefunction data type is complex and numpy arrays' data type
            is not float or complex
        NotImplementedError
            If generalized orbitals and odd number of spin orbitals

        Note
        ----
        Restricted Orbitals
            H is a np.ndarray or a 1-tuple of np.ndarray
            G is a np.ndarray or a 1-tuple of np.ndarray
            orbtype is 'restricted'
        Unrestricted Orbitals
            H is a 2-tuple of np.ndarray
            G is a 3-tuple of np.ndarray
            orbtype is 'unrestricted'
        Unrestricted Orbitals
            H is a np.ndarray or a 1-tuple of np.ndarray
            G is a np.ndarray or a 1-tuple of np.ndarray
            orbtype is 'generalized'
        """
        def process_integrals(integrals, is_G=False):
            """ Finds the orbital type of given integrals and convert them to an appropriate format

            Parameters
            ----------
            integrals : np.ndarray, tuple/list of np.ndarray
                Integrals given as numpy array or tuple
                Tuple structure of the integral is used to find the orbital type
            is_G : bool
                Flag to check if integrals are two electron integrals

            Returns
            -------
            integrals : tuple of np.ndarray
                Integrals
            orbtype : {'restricted', 'unrestricted', 'generalized'}
                Orbital type corresponding to the given integrals

            Raises
            ------
            TypeError
                If format of integrals is not supported
            """
            if isinstance(integrals, np.ndarray):
                return (integrals,), ['restricted', 'generalized']
            elif isinstance(integrals, (tuple, list)) and len(integrals) == 1:
                return tuple(integrals), ['restricted', 'generalized']
            elif isinstance(integrals, (tuple, list)) and len(integrals) == 2 + int(is_G):
                return tuple(integrals), ['unrestricted']
            else:
                raise TypeError('{0}-electron integrals must be given as a numpy array or as a 1-'
                                ' or {1}-tuple of numpy arrays'.format(1 + int(is_G),
                                                                       2 + int(is_G)))

        H, orbtype_H = process_integrals(H)
        G, orbtype_G = process_integrals(G, is_G=True)
        if orbtype_H != orbtype_G:
            raise TypeError('One and two electron integrals do not have consistent orbital type')

        if orbtype is None:
            orbtype = orbtype_H[0]
        elif orbtype not in orbtype_H:
            raise TypeError('Orbital type must be one of {0} with the given integrals'
                            ''.format(orbtype_H))

        for matrix in H + G:
            if not isinstance(matrix, np.ndarray):
                raise TypeError('Integrals must be given as a numpy array or a tuple of numpy'
                                ' arrays')
            if not all(i == H[0].shape[0] for i in matrix.shape):
                raise TypeError("All of the integral matrices need to have the same dimensions")
            if self.dtype == np.float64 and matrix.dtype not in (float, np.float64):
                raise TypeError('If the wavefunction data type is float, then the integrals must'
                                ' also be float')
            elif (self.dtype == np.complex128 and
                  matrix.dtype not in (float, np.float64, complex, np.complex128)):
                raise TypeError('If the wavefunction data type is complex, then the integrals must'
                                ' be float or complex')

        if orbtype == 'generalized' and H[0].shape[0] % 2 == 1:
            raise NotImplementedError('Odd number of "spin" orbitals will probably cause problems'
                                      ' somewhere')
        # TODO: check that two electron integrals are in physicist's notation (check symmetry)

        self.H = H
        self.G = G
        self.orbtype = orbtype
