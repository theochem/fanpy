""" Parent class of the wavefunctions

Contains information that all wavefunctions should have (probably?)
"""
from __future__ import absolute_import, division, print_function
import numpy as np

__all__ = []


class Wavefunction(object):
    """ Wavefunction class

    Contains the necessary information to solve the wavefunction

    Attributes
    ----------
    nelec : int
        Number of electrons
    dtype : {np.float64, np.complex128}
        Numpy data type

    Properties
    ----------
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals

    Method
    ------
    __init__(self, nelec, one_int, two_int, dtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    """
    def __init__(self, nelec, one_int, two_int, dtype=None):
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
        """
        # TODO: Implement loading one_int and two_int from various file formats
        self.assign_nelec(nelec)
        self.assign_dtype(dtype)


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

        if dtype is None or dtype in (float, np.float64):
            self.dtype = np.float64
        elif dtype in (complex, np.complex128):
            self.dtype = np.complex128
        else:
            raise TypeError("dtype must be one of {0}".format((float, complex,
                                                               np.float64, np.complex128)))
