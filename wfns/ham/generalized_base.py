"""Base Hamiltonian for generalized orbitals."""
import numpy as np
from wfns.ham.base import BaseHamiltonian


# FIXME: ordering of the words is not consistent with the GeneralizedChemicalHamiltonian
class BaseGeneralizedHamiltonian(BaseHamiltonian):
    """Base class for Hamiltonian with generalized orbitals.

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
    orb_rotate_jacobi(self, jacobi_indices, theta)
        Rotate orbitals using Jacobi matrix.
    orb_rotate_matrix(self, matrix)
        Rotate orbitals with a transformation matrix.

    Abstract Methods
    ----------------
    integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """

    # pylint: disable=W0223
    def __init__(self, one_int, two_int, energy_nuc_nuc=None):
        """Initialize the Hamiltonian.

        Parameters
        ----------
        one_int : np.ndarray(K, K)
            One electron integrals.
        two_int : np.ndarray(K, K, K, K)
            Two electron integrals.
        energy_nuc_nuc : {float, None}
            Nuclear nuclear repulsion energy.
            Default is `0.0`.

        """
        super().__init__(energy_nuc_nuc=energy_nuc_nuc)
        self.assign_integrals(one_int, two_int)

    @property
    def nspin(self):
        """Return the number of spin orbitals.

        Returns
        -------
        nspin : int
            Number of spin orbitals.

        """
        return self.one_int.shape[0]

    def assign_integrals(self, one_int, two_int):
        """Assign the one- and two-electron integrals.

        Parameters
        ----------
        one_int : np.ndarray(K, K)
            One electron integrals for generalized orbitals.
        two_int : np.ndarray(K, K, K, K)
            Two electron integrals for generalized orbitals.
            Uses physicist's notation.

        Raises
        ------
        TypeError
            If integrals are not provided as a numpy array.
        ValueError
            If one-electron integrals are not given as a (two-dimensional) square matrix.
            If two-electron integrals are not given as a four-dimensional tensor with same
            dimensionality in each axis.
            If one- and two-electron integrals do not have the same number of orbitals.

        """
        if not (
            isinstance(one_int, np.ndarray)
            and isinstance(two_int, np.ndarray)
        ):
            raise TypeError("Integrals must be given as a numpy array")
        if not (one_int.ndim == 2 and one_int.shape[0] == one_int.shape[1]):
            raise ValueError("One-electron integrals be a (two-dimensional) square matrix.")
        if not (
            two_int.ndim == 4
            and two_int.shape[0] == two_int.shape[1] == two_int.shape[2] == two_int.shape[3]
        ):
            raise ValueError(
                "Two-electron integrals must have four-dimensional tensor with equal "
                "number rows in each axis."
            )
        if one_int.shape[0] != two_int.shape[0]:
            raise ValueError(
                "One- and two-electron integrals must have the same number of " "orbitals."
            )

        self.one_int = one_int
        self.two_int = two_int

    def orb_rotate_jacobi(self, jacobi_indices, theta):
        """Rotate orbitals using Jacobi matrix.

        Parameters
        ----------
        jacobi_indices : 2-tuple of ints
            Indices of the orbitals that will be rotated
        theta : float, np.ndarray of float
            Angle with which the orbitals are rotated

        Raises
        ------
        TypeError
            If indices are not given as a tuple/list of two integers.
            If theta is not a flota or a numpy array of floats.
        ValueError
            If any of the pair of indices given are the same.
            If any of the index is less than 0 or greater than the number of rows.

        """
        # pylint: disable=C0103
        if not (
            isinstance(jacobi_indices, (tuple, list))
            and len(jacobi_indices) == 2
            and isinstance(jacobi_indices[0], int)
            and isinstance(jacobi_indices[1], int)
        ):
            raise TypeError("Indices must be given a tuple or list of two integers.")
        if jacobi_indices[0] == jacobi_indices[1]:
            raise ValueError("Indices must be different.")
        if not (
            0 <= jacobi_indices[0] < self.one_int.shape[0]
            and 0 <= jacobi_indices[1] < self.one_int.shape[0]
        ):
            raise ValueError(
                "Indices must be greater than or equal to 0 and less than the number of rows."
            )
        if not (
            isinstance(theta, (int, float))
            or (isinstance(theta, np.ndarray) and theta.dtype in [int, float] and theta.size == 1)
        ):
            raise TypeError("Angle `theta` must be a float or numpy array of one float.")

        p, q = jacobi_indices
        if p > q:
            p, q = q, p

        # one_electron
        p_col = self.one_int[:, p]
        q_col = self.one_int[:, q]
        (self.one_int[:, p], self.one_int[:, q]) = (
            np.cos(theta) * p_col - np.sin(theta) * q_col,
            np.sin(theta) * p_col + np.cos(theta) * q_col,
        )

        p_row = self.one_int[p, :]
        q_row = self.one_int[q, :]
        (self.one_int[p, :], self.one_int[q, :]) = (
            np.cos(theta) * p_row - np.sin(theta) * q_row,
            np.sin(theta) * p_row + np.cos(theta) * q_row,
        )

        # two electron
        p_slice = self.two_int[:, :, :, p]
        q_slice = self.two_int[:, :, :, q]
        (self.two_int[:, :, :, p], self.two_int[:, :, :, q]) = (
            np.cos(theta) * p_slice - np.sin(theta) * q_slice,
            np.sin(theta) * p_slice + np.cos(theta) * q_slice,
        )

        p_slice = self.two_int[:, :, p, :]
        q_slice = self.two_int[:, :, q, :]
        (self.two_int[:, :, p, :], self.two_int[:, :, q, :]) = (
            np.cos(theta) * p_slice - np.sin(theta) * q_slice,
            np.sin(theta) * p_slice + np.cos(theta) * q_slice,
        )

        p_slice = self.two_int[:, p, :, :]
        q_slice = self.two_int[:, q, :, :]
        (self.two_int[:, p, :, :], self.two_int[:, q, :, :]) = (
            np.cos(theta) * p_slice - np.sin(theta) * q_slice,
            np.sin(theta) * p_slice + np.cos(theta) * q_slice,
        )

        p_slice = self.two_int[p, :, :, :]
        q_slice = self.two_int[q, :, :, :]
        (self.two_int[p, :, :, :], self.two_int[q, :, :, :]) = (
            np.cos(theta) * p_slice - np.sin(theta) * q_slice,
            np.sin(theta) * p_slice + np.cos(theta) * q_slice,
        )

    def orb_rotate_matrix(self, matrix):
        r"""Rotate orbitals with a transformation matrix.

        .. math::

            \widetilde{h}_{ab} &= \sum_{ij} C^\dagger_{ai} h_{ij} C_{jb}\\
            \widetilde{g}_{abcd} &= \sum_{ijkl} C^\dagger_{ai} C^\dagger_{bj} g_{ijkl} C_{kc} C_{ld}

        Parameters
        ----------
        matrix : np.ndarray(K, L)
            Transformation matrix.

        Raises
        ------
        TypeError
            If matrix is not a numpy array.
        ValueError
            If shape of matrix does not match up with the shape of the integrals.

        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Transformation matrix must be given as a numpy array.")
        if matrix.ndim != 2:
            raise ValueError("Transformation matrix must be two-dimensional.")
        if matrix.shape[0] != self.one_int.shape[0]:
            raise ValueError(
                "Shape of the transformation matrix must match with the shape of the " "integrals."
            )
        # NOTE: don't need to check that matrix matches up with two_int b/c one_int and two_int have
        #       the same number of rows/columns

        self.one_int = np.einsum("ij,ia->aj", self.one_int, matrix)
        self.one_int = np.einsum("aj,jb->ab", self.one_int, matrix)

        self.two_int = np.einsum("ijkl,ia->ajkl", self.two_int, matrix)
        self.two_int = np.einsum("ajkl,jb->abkl", self.two_int, matrix)
        self.two_int = np.einsum("abkl,kc->abcl", self.two_int, matrix)
        self.two_int = np.einsum("abcl,ld->abcd", self.two_int, matrix)
