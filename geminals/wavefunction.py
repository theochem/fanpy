from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractproperty
import re

import numpy as np

# TODO: add custom exception class
class Wavefunction(object):
    """ Wavefunction class

    Contains the necessary information to projectively solve the wavefunction

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : np.ndarray(K,K) or tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        If tuple of np.ndarray (length 2), one electron integrals for the (alpha, alpha)
        and the (beta, beta) unrestricted orbitals
    G : np.ndarray(K,K,K,K) or tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        If tuple of np.ndarray (length 3), two electron integrals for the
        (alpha, alpha, alpha, alpha), (alpha, beta, alpha, beta), and
        (beta, beta, beta, beta) unrestricted orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nspatial : int
        Number of spatial orbitals
    nspin : int
        Number of spin orbitals (alpha and beta)
    nelec : int
        Number of electrons
    npair : int
        Number of electron pairs (rounded down)
    orb_type : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals

    Private
    -------
    _energy : float
        Electronic energy of the wavefunction

    Abstract Property
    -----------------
    _methods : dict
        Dictionary of methods that is used to solve the wavefunction
    """
    __metaclass__ = ABCMeta

    #
    # Default attribute values
    #
    @abstractproperty
    def _methods(self):
        """ Dictionary of methods for solving the wavefunction

        Returns
        -------
        methods : dict
            Dictionary of the form {'method name':method}
            Must have key "default"
        """
        # this is an example
        def example_method():
            pass
        return {"default": example_method}

    #
    # Special methods
    #

    def __init__(
        self,
        # Mandatory arguments
        nelec=None,
        H=None,
        G=None,
        # Arguments handled by base Wavefunction class
        dtype=None,
        nuc_nuc=None,
        orb_type=None,
    ):
        """

        Required Parameters
        -------------------
        nelec : int
            Number of electrons
            Assumes that the number of electrons is even
        H : np.ndarray(K,K) or tuple np.ndarray(K,K)
            If np.ndarray, one electron integrals for the spatial orbitals
            If tuple of np.ndarray (length 2), one electron integrals for the alpha orbitals
            and one electron integrals for the alpha orbitals
        G : np.ndarray(K,K,K,K) or tuple np.ndarray(K,K)
            If np.ndarray, two electron integrals for the spatial orbitals
            If tuple of np.ndarray (length 2), two electron integrals for the alpha orbitals
            and two electron integrals for the alpha orbitals

        Optional Parameters
        -------------------
        dtype : {float, complex, np.float64, np.complex128}
            Numpy data type
        nuc_nuc : float
            Nuclear nuclear repulsion value
        orb_type : {'restricted', 'unrestricted', 'generalized'}
            Type of the orbital used in obtaining the one-electron and two-electron integrals
            Default is 'restricted'

        """
        # TODO: Implement loading H, G, and x from various file formats
        self.assign_dtype(dtype)
        self.assign_integrals(H, G, orb_type=orb_type)
        self.assign_nuc_nuc(nuc_nuc)
        self.assign_nelec(nelec)
        self._energy = 0.0

    def __call__(self,  method="default", **kwargs):
        """ Optimize coefficients
        """
        if method in self._methods:
            optimizer = self._methods[method.lower()]
            return optimizer(**kwargs)
        else:
            raise ValueError("method must be one of {0}".format(self._methods.keys()))

    #
    # Assignment methods
    #

    def assign_dtype(self, dtype):
        """ Sets the data type of the parameters

        Parameters
        ----------
        dtype : {float, complex, np.float64, np.complex128}
            Numpy data type
            If None then set to np.float64
        """

        if dtype is None:
            dtype = np.float64
        elif dtype not in (float, complex, np.float64, np.complex128):
            raise TypeError("dtype must be one of {0}".format((float, complex, np.float64, np.complex128)))

        self.dtype = dtype

    def assign_integrals(self, H, G, orb_type=None):
        """ Sets one electron and two electron integrals and the orbital type

        Parameters
        ----------
        H : np.ndarray(K,K) or tuple np.ndarray(K,K)
            If np.ndarray, one electron integrals for the spatial or generalized orbitals
            If tuple of np.ndarray (length 2), one electron integrals for the (alpha, alpha)
            and the (beta, beta) unrestricted orbitals
        G : np.ndarray(K,K,K,K) or tuple np.ndarray(K,K)
            If np.ndarray, two electron integrals for the spatial or generalized orbitals
            If tuple of np.ndarray (length 3), two electron integrals for the
            (alpha, alpha, alpha, alpha), (alpha, beta, alpha, beta), and
            (beta, beta, beta, beta) unrestricted orbitals
        orb_type : {'restricted', 'unrestricted', 'generalized'}
            Type of the orbital used in obtaining the one-electron and two-electron integrals
            Default is 'restricted'

        """
        # NOTE: check symmetry?
        # NOTE: assumes number of alpha and beta spin orbitals are equal
        # If numpy array
        if isinstance(H, np.ndarray) and isinstance(G, np.ndarray):
            # must not be unrestricted
            if orb_type == 'unrestricted' :
                raise TypeError('Integrals of unrestricted orbitals must be given as a tuple')
            elif orb_type is None:
                orb_type = 'restricted'
            H = (H,)
            G = (G,)
        # If tuple
        elif isinstance(H, tuple) and isinstance(G, tuple):
            if not all(isinstance(i, np.ndarray) for i in H+G) :
                raise TypeError('H and G as tuples must contain only np.ndarray')
            # tuple of one integrals
            if len(H) == 1 and len(G) == 1:
                # must be restricted or generalized
                if orb_type == 'unrestricted':
                    raise TypeError('Integrals of unrestricted orbitals must be'
                                    'given as a tuple of numpy arrays (2 for H and 3 for G)')
                elif orb_type is None:
                    orb_type = 'unrestricted'
            # tuple of two and three integrals
            elif len(H) == 2 and len(G) == 3:
                # must be unrestricted
                if orb_type in ['restricted', 'generalized']:
                    raise TypeError('Integrals of restricted and generalized'
                                    'orbitals must be given as a numpy array or'
                                    'as a tuple of one numpy array')
                elif orb_type is None:
                    orb_type = 'unrestricted'
            # tuple of other lengths
            else:
                raise TypeError('The tuples of H and G have unsupported lengths,'
                                '{0} and {1}, respectively'.format(len(H), len(G)))
        # If other type
        else:
            raise TypeError('H and G must be of the same type and be one of {0}'
                            ''.format((np.ndarray, tuple)))
        for matrix in H+G:
            if not matrix.dtype in (float, complex, np.float64, np.complex128):
                raise TypeError('Integral matrices dtypes must be one of {0}'
                                ''.format((float, complex, np.float64, np.complex128)))
            if not np.all(np.array(matrix.shape) == matrix.shape[0]):
                raise ValueError("Integral matrices' dimensions must all be equal")
        if orb_type not in ['restricted', 'unrestricted', 'generalized']:
            raise ValueError('Orbital type must be one of {0}'
                             ''.format(['restricted', 'unrestricted', 'generalized']))

        self.H = H
        self.G = G
        self.orb_type = orb_type

    def assign_nuc_nuc(self, nuc_nuc=None):
        """ Sets the nuclear nuclear repulsion value

        Parameters
        ----------
        nuc_nuc : float
            Nuclear nuclear repulsion value
            If None then set to 0.0
        """
        if nuc_nuc is None:
            nuc_nuc = 0.0
        elif not isinstance(nuc_nuc, (float, np.float64)):
            raise TypeError("nuc_nuc integral must be one of {0}".format((float, np.float64)))
        self.nuc_nuc = nuc_nuc

    def assign_nelec(self, nelec):
        """ Sets the number of electrons

        Parameters
        ----------
        nelec : int
            Number of electrons
        """
        if not isinstance(nelec, int):
            raise TypeError("nelec must be of type {0}".format(int))
        elif nelec <= 0:
            raise ValueError("nelec must be a positive integer")
        self.nelec = nelec


    #
    # View methods
    #
    def compute_energy(self, include_nuc=True):
        """ Returns the energy of the system

        Parameters
        ----------
        include_nuc : bool
            Flag to include nuclear nuclear repulsion

        Returns
        -------
        energy : float
            Total energy if include_nuc is True
            Electronic energy if include_nuc is False
        """
        nuc_nuc = self.nuc_nuc if include_nuc else 0.0
        return self._energy + nuc_nuc

    #
    # Properties
    #
    @property
    def nspin(self):
        """ Number of spin orbitals
        """
        if self.orb_type in ['restricted', 'unrestricted']:
            return 2*self.H[0].shape[0]
        elif self.orb_type == 'generalized':
            return self.H[0].shape[0]

    @property
    def nspatial(self):
        """ Number of spatial orbitals (rounded down from number of spin orbitals/2)
        """
        return self.nspin//2

    @property
    def npair(self):
        """ Number of electron pairs (rounded down)
        """
        return self.nelec//2
