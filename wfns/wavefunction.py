""" Parent class of the wavefunctions

Contains information that all wavefunctions should have (probably?)
"""
from __future__ import absolute_import, division, print_function
import numpy as np


class Wavefunction(object):
    """ Wavefunction class

    Contains the necessary information to solve the wavefunction

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

    Properties
    ----------
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals

    Method
    ------
    __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_integrals(self, one_int, two_int, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
    """
    def __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None):
        """ Initializes a wavefunction

        Parameters
        ----------
        nelec : int
            Number of electrons

        one_int : np.ndarray(K,K), 1- or 2-tuple np.ndarray(K,K)
            One electron integrals
            For spatial and generalized orbitals, np.ndarray or 1-tuple of np.ndarray
            For unretricted spin orbitals, 2-tuple of np.ndarray

        two_int : np.ndarray(K,K,K,K), 1- or 3-tuple np.ndarray(K,K,K,K)
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
        # TODO: Implement loading one_int and two_int from various file formats
        self.assign_nelec(nelec)
        self.assign_dtype(dtype)
        self.assign_nuc_nuc(nuc_nuc)
        self.assign_integrals(one_int, two_int, orbtype=orbtype)


    @property
    def nspin(self):
        """ Number of spin orbitals
        """
        if self.orbtype in ['restricted', 'unrestricted']:
            return 2 * self.one_int[0].shape[0]
        elif self.orbtype == 'generalized':
            return self.one_int[0].shape[0]


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


    def assign_integrals(self, one_int, two_int, orbtype=None):
        """ Sets one electron and two electron integrals and the orbital type

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
        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbital used in obtaining the one-electron and two-electron integrals
            Default is `'restricted'`

        Raises
        ------
        TypeError:
            If `one_int` and `two_int` are not a numpy array or a tuple of numpy arrays
            If orbital type from `one_int` and orbital type from `two_int` are not the same
            If orbital type is inconsistent with the integrals given
            If `one_int` and `two_int` are tuples and its elements are not numpy arrays
            If `one_int` and `two_int` are tuples and numpy arrays do not have the consistent shapes
            If `one_int` and `two_int` are tuples and wavefunction data type is float and numpy
            arrays' data type is not float
            If one_int and two_int are tuples and wavefunction data type is complex and numpy
            arrays' data type is not float or complex
        NotImplementedError
            If generalized orbitals and odd number of spin orbitals

        Note
        ----
        Restricted Orbitals
            one_int is a np.ndarray or a 1-tuple of np.ndarray
            two_int is a np.ndarray or a 1-tuple of np.ndarray
            orbtype is 'restricted'
        Unrestricted Orbitals
            one_int is a 2-tuple of np.ndarray
            two_int is a 3-tuple of np.ndarray
            orbtype is 'unrestricted'
        Unrestricted Orbitals
            one_int is a np.ndarray or a 1-tuple of np.ndarray
            two_int is a np.ndarray or a 1-tuple of np.ndarray
            orbtype is 'generalized'
        """
        def process_integrals(integrals, is_two_int=False):
            """ Finds the orbital type of given integrals and convert them to an appropriate format

            Parameters
            ----------
            integrals : np.ndarray, tuple/list of np.ndarray
                Integrals given as numpy array or tuple
                Tuple structure of the integral is used to find the orbital type
            is_two_int : bool
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
            elif isinstance(integrals, (tuple, list)) and len(integrals) == 2 + int(is_two_int):
                return tuple(integrals), ['unrestricted']
            else:
                raise TypeError('{0}-electron integrals must be given as a numpy array or as a 1-'
                                ' or {1}-tuple of numpy arrays'.format(1 + int(is_two_int),
                                                                       2 + int(is_two_int)))

        one_int, orbtype_one_int = process_integrals(one_int)
        two_int, orbtype_two_int = process_integrals(two_int, is_two_int=True)
        if orbtype_one_int != orbtype_two_int:
            raise TypeError('One and two electron integrals do not have consistent orbital type')

        if orbtype is None:
            orbtype = orbtype_one_int[0]
        elif orbtype not in orbtype_one_int:
            raise TypeError('Orbital type must be one of {0} with the given integrals'
                            ''.format(orbtype_one_int))

        for matrix in one_int + two_int:
            if not isinstance(matrix, np.ndarray):
                raise TypeError('Integrals must be given as a numpy array or a tuple of numpy'
                                ' arrays')
            if not all(i == one_int[0].shape[0] for i in matrix.shape):
                raise TypeError("All of the integral matrices need to have the same dimensions")
            if self.dtype == np.float64 and matrix.dtype not in (float, np.float64):
                raise TypeError('If the wavefunction data type is float, then the integrals must'
                                ' also be float')
            elif (self.dtype == np.complex128 and
                  matrix.dtype not in (float, np.float64, complex, np.complex128)):
                raise TypeError('If the wavefunction data type is complex, then the integrals must'
                                ' be float or complex')

        if orbtype == 'generalized' and one_int[0].shape[0] % 2 == 1:
            raise NotImplementedError('Odd number of "spin" orbitals will probably cause problems'
                                      ' somewhere')
        # TODO: check that two electron integrals are in physicist's notation (check symmetry)

        self.one_int = one_int
        self.two_int = two_int
        self.orbtype = orbtype
