"""Class for storing integrals.

Classes
-------
BaseIntegrals

OneElectronIntegrals
TwoElectronIntegrals
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..backend import slater
# TODO: store fraction of integrals
# TODO: check that two electron integrals are in physicist's notation (check symmetry)


class BaseIntegrals:
    """Base class for storing integrals.

    Attributes
    ----------
    integrals : tuple of np.ndarray
        Values of the integrals corresponding to some basis set.
    """
    def __init__(self, integrals):
        """Initialize Integrals.

        Parameters
        ----------
        integrals : {np.ndarray, tuple of np.ndarray}
            Values of the integrals corresponding to some basis set.

        Raises
        ------
        TypeError
            If integrals are not provided as a numpy array or a tuple of numpy arrays
            If integral matrices do not have dtype of `float` or `complex`
            If integral matrices do not have the same dtype
        """
        if isinstance(integrals, np.ndarray):
            integrals = (integrals, )
        elif (isinstance(integrals, (list, tuple)) and
              all(isinstance(matrix, np.ndarray) for matrix in integrals)):
            integrals = tuple(integrals)
        elif isinstance(integrals, self.__class__):
            integrals = integrals.integrals
        else:
            raise TypeError('Unsupported integral format. They must be provided as a numpy array '
                            'or a tuple of numpy arrays')
        for matrix in integrals:
            if matrix.dtype not in [float, complex]:
                raise TypeError('Integral matrices must have dtype of `float` or `complex`.')
            if matrix.dtype != integrals[0].dtype:
                raise TypeError('Integral matrices must have the same dtype.')
        self.integrals = integrals


class OneElectronIntegrals(BaseIntegrals):
    """Class for storing one-electron integrals.

    #FIXME: substitute \hat{h} with real eqn
    ..math::
        h_{ik} = \big< \phi_i \big | \hat{h} | \phi_k \big>

    Attributes
    ----------
    integrals : tuple of np.ndarray
        Values of the one-electron integrals corresponding to some basis set.

    Properties
    ----------
    possible_orbtypes : tuple of str
        Possible orbital types from the given integrals.
    num_orbs : int
        Number of orbitals.

    Methods:
    ---------
    __init__(self, integrals)
    get_value(self, i, k, orbtype)
    """

    def __init__(self, integrals):
        """Initialize Integrals.

        Parameters
        ----------
        integrals : {np.ndarray, tuple of np.ndarray}
            Values of the one-electron integrals corresponding to some basis set.

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
        """
        super().__init__(integrals)
        if len(self.integrals) not in (1, 2):
            raise TypeError('Unsupported number of one-electron integral matrices.')
        for matrix in self.integrals:
            if len(matrix.shape) != 2:
                raise TypeError('One-electron integral matrices must be two dimensional.')
            elif matrix.shape[0] != matrix.shape[1]:
                raise TypeError('One-electron integral matrices must be square (for integration '
                                'within one basis set).')
        if len(self.integrals) == 2 and self.integrals[0].shape != self.integrals[1].shape:
            raise ValueError('Number of alpha and beta orbitals must be the same.')

    @property
    def possible_orbtypes(self):
        """Possible orbital types from the given orbitals."""
        if len(self.integrals) == 1:
            return ('restricted', 'generalized')
        elif len(self.integrals) == 2:
            return ('unrestricted', )
        else:
            raise NotImplementedError('Only one and two one-electron integral matrices are '
                                      'supported.')

    @property
    def num_orbs(self):
        """Number of orbitals.

        Assumes that the integrals have the same dimensionality in all axis
        """
        return self.integrals[0].shape[0]

    def get_value(self, i, k, orbtype):
        """Get value of the one-electron hamiltonian integral with orbitals `i` and `k`.

        ..math::
            h_{ik} = \big< \phi_i \big | \hat{h} | \phi_k \big>

        Parameters
        ----------
        i : int
            Index of the spin orbital
        k : int
            Index of the spin orbital
        orbtype : {'restricted', 'unrestricted', 'generalized'}
            Flag that indicates the type of the orbital

        Returns
        -------
        h_ik : float
            Value of the one-electron integral

        Raises
        ------
        ValueError
            If indices are less than zero
            If indices are greater than the number of spin orbitals
        TypeError
            If orbtype is not one of 'restricted', 'unrestricted', 'generalized'
        """
        # NOTE: NECESARY?
        if i < 0 or k < 0:
            raise ValueError('Indices cannot be negative')

        if orbtype == 'restricted':
            # self.num_orbs is the number of spatial orbitals
            if i >= 2*self.num_orbs or k >= 2*self.num_orbs:
                raise ValueError('Indices cannot be greater than the number of spin orbitals')
            # spatial indices
            I = slater.spatial_index(i, self.num_orbs)
            K = slater.spatial_index(k, self.num_orbs)
            # if spins are the same
            if slater.is_alpha(i, self.num_orbs) == slater.is_alpha(k, self.num_orbs):
                return self.integrals[0][I, K]
        elif orbtype == 'unrestricted':
            # self.num_orbs is the number of spatial orbitals
            if i >= 2*self.num_orbs or k >= 2*self.num_orbs:
                raise ValueError('Indices cannot be greater than the number of spin orbitals')
            # spatial indices
            I = slater.spatial_index(i, self.num_orbs)
            K = slater.spatial_index(k, self.num_orbs)
            # if spins are both alpha
            if slater.is_alpha(i, self.num_orbs) and slater.is_alpha(k, self.num_orbs):
                return self.integrals[0][I, K]
            # if spins are both beta
            elif not slater.is_alpha(i, self.num_orbs) and not slater.is_alpha(k, self.num_orbs):
                return self.integrals[1][I, K]
        elif orbtype == 'generalized':
            if i >= self.num_orbs or k >= self.num_orbs:
                raise ValueError('Indices cannot be greater than the number of spin orbitals')
            return self.integrals[0][i, k]
        else:
            raise TypeError('Unknown orbital type, {0}'.format(orbtype))


class TwoElectronIntegrals(BaseIntegrals):
    """Class for storing two-electron integrals.

    Attributes
    ----------
    integrals : tuple of np.ndarray
        Values of the two-electron integrals corresponding to some basis set.

    Properties
    ----------
    possible_orbtypes : tuple of str
        Possible orbital types from the given integrals.
    num_orbs : int
        Number of orbitals.

    Methods:
    ---------
    __init__(self, integrals)
    get_value(self, i, k, orbtype)
    """

    def __init__(self, integrals, notation='physicist'):
        """Initialize Integrals.

        Parameters
        ----------
        integrals : {np.ndarray, tuple of np.ndarray}
            Values of the one-electron integrals corresponding to some basis set.
        notation : {'physicist', 'chemist'}
            Notation of the integrals.

        Raises
        ------
        TypeError
            If integrals are not provided as a numpy array or a tuple of numpy arrays
            If integral matrices do not have dtype of `float` or `complex`
            If integral matrices do not have the same dtype
            If the number of two-electron integral matrices is not 1 or 3
            If two-electron integral matrices are not four dimensional
            If two-electron integral matrices does not have same dimensionality in all axis
            If two-electron integrals (for the unrestricted orbitals) has different number of alpha
            and beta orbitals
        """
        super().__init__(integrals)
        if len(self.integrals) not in (1, 3):
            raise TypeError('Unsupported number of one-electron integral matrices.')
        for matrix in self.integrals:
            if len(matrix.shape) != 4:
                raise TypeError('Two-electron integral matrices must be four dimensional.')
            elif matrix.shape[0] != matrix.shape[1] != matrix.shape[2] != matrix.shape[3]:
                raise TypeError('Two-electron integral matrices must be square (for integration '
                                'within one basis set).')
        if (len(self.integrals) == 3 and
                not(self.integrals[0].shape == self.integrals[1].shape == self.integrals[2].shape)):
            raise ValueError('Number of alpha and beta orbitals must be the same.')

        if notation not in ('physicist', 'chemist'):
            raise ValueError('Unsupported notation of integrals.')
        self.notation = notation

    @property
    def possible_orbtypes(self):
        """Possible orbital types from the given orbitals."""
        if len(self.integrals) == 1:
            return ('restricted', 'generalized')
        elif len(self.integrals) == 3:
            return ('unrestricted', )
        else:
            raise NotImplementedError('Only one and three two-electron integral matrices are '
                                      'supported.')

    @property
    def num_orbs(self):
        """Number of orbitals.

        Assumes that the integrals have the same dimensionality in all axis
        """
        return self.integrals[0].shape[0]

    def get_value(self, i, j, k, l, orbtype, notation='physicist'):
        """ Gets value of the two-electron hamiltonian integral with orbitals `i`, `j`, `k`, and `l`

        ..math::
            \big< \theta \big | \hat{g} a_i a_j a^\dagger_k a^\dagger_l | \big> =
            \big< \phi_i \phi_j \big | \hat{g} | \phi_k \phi_l \big>

        Parameters
        ----------
        i : int
            Index of the spin orbital
        j : int
            Index of the spin orbital
        k : int
            Index of the spin orbital
        l : int
            Index of the spin orbital
        orbtype : {'restricted', 'unrestricted', 'generalized'}
            Flag that indicates the type of the orbital
        notation : {'physicist', 'chemist'}
            Notation used to access the integrals.

        Returns
        -------
        g_ijkl : float
            Value of the one electron hamiltonian

        Raises
        ------
        ValueError
            If indices are less than zero
            If indices are greater than the number of spin orbitals
        TypeError
            If orbtype is not one of 'restricted', 'unrestricted', 'generalized'
        """
        # NOTE: NECESARY?
        if i < 0 or j < 0 or k < 0 or l < 0:
            raise ValueError('Indices cannot be negative')

        if notation != self.notation:
            j, k = k, j

        if orbtype == 'restricted':
            # self.num_orbs is the number of spatial orbitals
            if any(index >= 2*self.num_orbs for index in (i, j, k, l)):
                raise ValueError('Indices cannot be greater than the number of spin orbitals')
            # if i and k have same spin and j and l have same spin
            if (slater.is_alpha(i, self.num_orbs) == slater.is_alpha(k, self.num_orbs)
                    and slater.is_alpha(j, self.num_orbs) == slater.is_alpha(l, self.num_orbs)):
                # spatial indices
                I = slater.spatial_index(i, self.num_orbs)
                J = slater.spatial_index(j, self.num_orbs)
                K = slater.spatial_index(k, self.num_orbs)
                L = slater.spatial_index(l, self.num_orbs)
                return self.integrals[0][I, J, K, L]
        elif orbtype == 'unrestricted':
            # self.num_orbs is the number of spatial orbitals
            self.num_orbs = self.integrals[0].shape[0]
            if any(index >= 2*self.num_orbs for index in (i, j, k, l)):
                raise ValueError('Indices cannot be greater than the number of spin orbitals')
            # spatial indices
            I = slater.spatial_index(i, self.num_orbs)
            J = slater.spatial_index(j, self.num_orbs)
            K = slater.spatial_index(k, self.num_orbs)
            L = slater.spatial_index(l, self.num_orbs)
            # if alpha alpha alpha alpha
            if all(slater.is_alpha(index, self.num_orbs) for index in (i, j, k, l)):
                return self.integrals[0][I, J, K, L]
            # if alpha beta alpha beta
            elif (slater.is_alpha(i, self.num_orbs) and not slater.is_alpha(j, self.num_orbs) and
                  slater.is_alpha(k, self.num_orbs) and not slater.is_alpha(l, self.num_orbs)):
                return self.integrals[1][I, J, K, L]
            # if beta alpha beta alpha
            elif (not slater.is_alpha(i, self.num_orbs) and slater.is_alpha(j, self.num_orbs) and
                  not slater.is_alpha(k, self.num_orbs) and slater.is_alpha(l, self.num_orbs)):
                # take the appropraite transposition to get the beta alpha beta alpha form
                return self.integrals[1][J, I, L, K]
            # if beta beta beta beta
            elif all(not slater.is_alpha(index, self.num_orbs) for index in (i, j, k, l)):
                return self.integrals[2][I, J, K, L]
        elif orbtype == 'generalized':
            if any(index >= self.num_orbs for index in (i, j, k, l)):
                raise ValueError('Indices cannot be greater than the number of spin orbitals')
            return self.integrals[0][i, j, k, l]
        else:
            raise TypeError('Unknown orbital type, {0}'.format(orbtype))
        return 0.0
