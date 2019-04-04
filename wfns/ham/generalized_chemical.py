r"""Hamiltonian used to describe a chemical system expressed wrt generalized orbitals."""
import numpy as np
from wfns.backend import slater, math_tools
from wfns.ham.generalized_base import BaseGeneralizedHamiltonian


class GeneralizedChemicalHamiltonian(BaseGeneralizedHamiltonian):
    r"""Hamiltonian used to describe a typical chemical system expressed wrt generalized orbitals.

    .. math::

        \hat{H} = \sum_{ij} h_{ij} a^\dagger_i a_j
        + \frac{1}{2} \sum_{ijkl} g_{ijkl} a^\dagger_i a^\dagger_j a_l a_k

    where :math:`h_{ik}` is the one-electron integral and :math:`g_{ijkl}` is the two-electron
    integral in Physicists' notation.

    Attributes
    ----------
    energy_nuc_nuc : float
        Nuclear-nuclear repulsion energy.
    one_int : np.ndarray(K, K)
        One-electron integrals.
    two_int : np.ndarray(K, K, K, K)
        Two-electron integrals.
    params : np.ndarray
        Significant elements of the anti-Hermitian matrix.

    Properties
    ----------
    dtype : {np.float64, np.complex128}
        Data type of the Hamiltonian.
    nspin : int
        Number of spin orbitals.
    nparams : int
        Number of parameters.

    Methods
    -------
    __init__(self, one_int, two_int, orbtype=None, energy_nuc_nuc=None)
        Initialize the Hamiltonian
    assign_energy_nuc_nuc(self, energy_nuc_nuc=None)
        Assigns the nuclear nuclear repulsion.
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.
    orb_rotate_jacobi(self, jacobi_indices, theta)
        Rotate orbitals using Jacobi matrix.
    orb_rotate_matrix(self, matrix)
        Rotate orbitals using a transformation matrix.
    clear_cache(self)
        Placeholder function that would clear the cache.
    assign_params(self, params)
        Transform the integrals with a unitary matrix that corresponds to the given parameters.
    integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """

    def __init__(self, one_int, two_int, energy_nuc_nuc=None, params=None):
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
        super().__init__(one_int, two_int, energy_nuc_nuc=energy_nuc_nuc)
        self.set_ref_ints()
        self.cache_two_ints()
        self.assign_params(params=params)

    def set_ref_ints(self):
        """Store the current integrals as the reference from which orbitals will be rotated."""
        self._ref_one_int = np.copy(self.one_int)
        self._ref_two_int = np.copy(self.two_int)

    def cache_two_ints(self):
        """Cache away contractions of the two electron integrals."""
        # store away tensor contractions
        indices = np.arange(self.one_int.shape[0])
        self._cached_two_int_ijij = self.two_int[
            indices[:, None], indices, indices[:, None], indices
        ]
        self._cached_two_int_ijji = self.two_int[
            indices[:, None], indices, indices, indices[:, None]
        ]

    def assign_params(self, params=None):
        """Transform the integrals with a unitary matrix that corresponds to the given parameters.

        Parameters
        ----------
        params : {np.ndarray, None}
            Significant elements of the anti-Hermitian matrix. Integrals will be transformed with
            the Unitary matrix that corresponds to the anti-Hermitian matrix.

        Raises
        ------
        ValueError
            If parameters is not a one-dimensional numpy array with K*(K-1)/2 elements, where K is
            the number of orbitals.

        """
        # NOTE: nspin is not used here because RestrictedChemicalHamiltonian inherits this method
        # and it would need to use the number of spatial orbitals instead of the spin orbitals
        # FIXME: this is too hacky. should the dimension of the antihermitian matrix be stored
        # somewhere?
        num_orbs = self.one_int.shape[0]
        num_params = num_orbs * (num_orbs - 1) // 2

        if params is None:
            params = np.zeros(num_params)

        if not (isinstance(params, np.ndarray) and params.ndim == 1 and params.size == num_params):
            raise ValueError(
                "Parameters for orbital rotation must be a one-dimension numpy array "
                "with {0}=K*(K-1)/2 elements, where K is the number of "
                "orbitals.".format(num_params)
            )

        # assign parameters
        self.params = params

        # revert integrals back to original
        self.assign_integrals(np.copy(self._ref_one_int), np.copy(self._ref_two_int))

        # convert antihermitian part to unitary matrix.
        unitary = math_tools.unitary_matrix(params)

        # transform integrals
        self.orb_rotate_matrix(unitary)

        # cache two electron integrals
        self.cache_two_ints()

    def integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None):
        r"""Integrate the Hamiltonian with against two Slater determinants.

        .. math::

            H_{\mathbf{m}\mathbf{n}} &=
            \left< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \right>\\
            &= \sum_{ij}
               h_{ij} \left< \mathbf{m} \middle| a^\dagger_i a_j \middle| \mathbf{n} \right>
            + \sum_{i<j, k<l} g_{ijkl}
            \left< \mathbf{m} \middle| a^\dagger_i a^\dagger_j a_l a_k \middle| \mathbf{n} \right>\\

        In the first summation involving :math:`h_{ij}`, only the terms where :math:`\mathbf{m}` and
        :math:`\mathbf{n}` are different by at most single excitation will contribute to the
        integral. In the second summation involving :math:`g_{ijkl}`, only the terms where
        :math:`\mathbf{m}` and :math:`\mathbf{n}` are different by at most double excitation will
        contribute to the integral.

        Parameters
        ----------
        sd1 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sign : {1, -1, None}
            Sign change resulting from cancelling out the orbitals shared between the two Slater
            determinants.
            Computes the sign if none is provided.
            Make sure that the provided sign is correct. It will not be checked to see if its
            correct.
        deriv : {int, None}
            Index of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.

        Returns
        -------
        one_electron : float
            One-electron energy.
        coulomb : float
            Coulomb energy.
        exchange : float
            Exchange energy.

        Raises
        ------
        ValueError
            If `sign` is not `1`, `-1` or `None`.

        """
        # pylint: disable=C0103
        if deriv is not None:
            sign = 1 if sign is None else sign
            return sign * self._integrate_sd_sd_deriv(sd1, sd2, deriv)

        sd1 = slater.internal_sd(sd1)
        sd2 = slater.internal_sd(sd2)
        shared_indices = np.array(slater.shared_orbs(sd1, sd2))
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1, sd2)
        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            return 0.0, 0.0, 0.0
        diff_order = len(diff_sd1)
        if diff_order > 2:
            return 0.0, 0.0, 0.0

        if sign is None:
            sign = slater.sign_excite(sd1, diff_sd1, reversed(diff_sd2))
        elif sign not in [1, -1]:
            raise ValueError("The sign associated with the integral must be either `1` or `-1`.")

        one_electron, coulomb, exchange = 0.0, 0.0, 0.0

        # two sd's are the same
        if diff_order == 0 and shared_indices.size > 0:
            one_electron += np.sum(self.one_int[shared_indices, shared_indices])
            coulomb += np.sum(
                np.triu(self._cached_two_int_ijij[shared_indices[:, None], shared_indices], k=1)
            )
            exchange -= np.sum(
                np.triu(self._cached_two_int_ijji[shared_indices[:, None], shared_indices], k=1)
            )

        # two sd's are different by single excitation
        elif diff_order == 1:
            a, = diff_sd1
            b, = diff_sd2
            one_electron += self.one_int[a, b]
            if shared_indices.size > 0:
                coulomb += np.sum(self.two_int[shared_indices, a, shared_indices, b])
                exchange -= np.sum(self.two_int[shared_indices, a, b, shared_indices])

        # two sd's are different by double excitation
        else:
            a, b = diff_sd1
            c, d = diff_sd2
            coulomb += self.two_int[a, b, c, d]
            exchange -= self.two_int[a, b, d, c]

        return sign * one_electron, sign * coulomb, sign * exchange

    def param_ind_to_rowcol_ind(self, param_ind):
        r"""Return the row and column indices of the antihermitian matrix from the parameter index.

        Let :math:`n` be the number of columns and rows in the antihermitian matrix and :math:`x`
        and :math:`y` be the row and column indices of the antihermitian matrix, respectively.
        First, we want to convert the index of the parameter, :math:`i`, to the index of the
        flattened antihermitian matrix, :math:`j`:

        .. math::

            j &= i + 1 + 2 + \dots + (x+1)\\
            &= i + \sum_{k=1}^{x+1} k\\
            &= i + \frac{(x+1)(x+2)}{2}\\

        We can find :math:`x` by finding the smallest :math:`x` such that

        .. math::

            ind - (n-1) - (n-2) - \dots - (n-x-1) &< 0\\
            ind - \sum_{k=1}^{x+1} (n-k) &< 0\\
            ind - n(x+1) + \sum_{k=1}^{x+1} k &< 0\\
            ind - n(x+1) + \frac{(x+1)(x+2)}{2} &< 0\\

        Once we find :math:`x`, we can find :math:`j` and then :math:`y`

        .. math::

            y = j \mod n

        Parameters
        ----------
        param_ind : int
            Index of the parameter.

        Returns
        -------
        matrix_indices : 2-tuple of int
            Indices of row and column of the antihermitian matrix that corresponds to the given
            parameter index.

        Raises
        ------
        ValueError

        """
        # pylint: disable=C0103
        # ind = i
        n = self.nspin
        for k in range(n + 1):  # pragma: no cover
            x = k
            if param_ind - n * (x + 1) + (x + 1) * (x + 2) / 2 < 0:
                break
        # ind_flat = j
        ind_flat = param_ind + (x + 1) * (x + 2) / 2
        y = ind_flat % n

        return int(x), int(y)

    # TODO: Much of the following function can be shortened by using impure functions (function with
    # a side effect) instead
    # FIXME: too many branches, too many statements
    def _integrate_sd_sd_deriv(self, sd1, sd2, deriv):
        r"""Return derivative of the CI matrix element with respect to the antihermitian elements.

        Parameters
        ----------
        sd1 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater Determinant against which the Hamiltonian is integrated.
        deriv : int
            Index of the Hamiltonian parameter against which the integral is derivatized.

        Returns
        -------
        one_electron : float
            One-electron energy derivatized with respect to the given index.
        coulomb : float
            Coulomb energy derivatized with respect to the given index.
        exchange : float
            Exchange energy derivatized with respect to the given index.

        Raises
        ------
        ValueError
            If the given `deriv` is not an integer greater than or equal to 0 and less than the
            number of parameters.

        Notes
        -----
        Integrals are not assumed to be real. The performance benefit (at the moment) for assuming
        real orbitals is not much.

        """
        # pylint: disable=C0103,R0912,R0915
        sd1 = slater.internal_sd(sd1)
        sd2 = slater.internal_sd(sd2)
        # NOTE: shared_indices contains spatial orbital indices
        shared_indices = np.array(slater.shared_orbs(sd1, sd2))
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1, sd2)

        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            return 0.0, 0.0, 0.0
        diff_order = len(diff_sd1)
        if diff_order > 2:
            return 0.0, 0.0, 0.0

        # get sign
        sign = slater.sign_excite(sd1, diff_sd1, reversed(diff_sd2))

        # check deriv
        if not (isinstance(deriv, int) and 0 <= deriv < self.nparams):
            raise ValueError(
                "Given derivative index must be an integer greater than or equal to "
                "zero and less than the number of parameters, "
                "nspatial * (nspatial-1)/2"
            )

        # turn deriv into indices of the matrix, (x, y), where x < y
        x, y = self.param_ind_to_rowcol_ind(deriv)

        one_electron, coulomb, exchange = 0.0, 0.0, 0.0

        # two sd's are the same
        if diff_order == 0:
            # remove orbitals x and y from the shared indices
            shared_indices_no_x = shared_indices[shared_indices != x]
            shared_indices_no_y = shared_indices[shared_indices != y]

            if x in shared_indices:
                one_electron -= 2 * np.real(self.one_int[x, y])
                if shared_indices_no_x.size != 0:
                    coulomb -= 2 * np.sum(
                        np.real(self.two_int[x, shared_indices_no_x, y, shared_indices_no_x])
                    )
                    exchange += 2 * np.sum(
                        np.real(self.two_int[x, shared_indices_no_x, shared_indices_no_x, y])
                    )

            if y in shared_indices:
                one_electron += 2 * np.real(self.one_int[x, y])
                if shared_indices_no_y.size != 0:
                    coulomb += 2 * np.sum(
                        np.real(self.two_int[x, shared_indices_no_y, y, shared_indices_no_y])
                    )
                    exchange -= 2 * np.sum(
                        np.real(self.two_int[x, shared_indices_no_y, shared_indices_no_y, y])
                    )
        # two sd's are different by single excitation
        elif diff_order == 1:
            a, = diff_sd1
            b, = diff_sd2

            if x == a:
                one_electron -= self.one_int[y, b]
                if shared_indices.size != 0:
                    coulomb -= np.sum(self.two_int[y, shared_indices, b, shared_indices])
                    exchange += np.sum(self.two_int[y, shared_indices, shared_indices, b])
            elif x == b:
                one_electron -= self.one_int[a, y]
                if shared_indices.size != 0:
                    coulomb -= np.sum(self.two_int[a, shared_indices, y, shared_indices])
                    exchange += np.sum(self.two_int[a, shared_indices, shared_indices, y])
            elif x in shared_indices:
                # NOTE: we can use the following if we assume that the orbitals are real
                # coulomb -= 2 * self.two_int[x, a, y, b]
                coulomb -= self.two_int[y, a, x, b]
                coulomb -= self.two_int[x, a, y, b]
                exchange += self.two_int[y, a, b, x]
                exchange += self.two_int[x, a, b, y]

            if y == a:
                one_electron += self.one_int[x, b]
                if shared_indices.size != 0:
                    coulomb += np.sum(self.two_int[x, shared_indices, b, shared_indices])
                    exchange -= np.sum(self.two_int[x, shared_indices, shared_indices, b])
            elif y == b:
                one_electron += self.one_int[a, x]
                if shared_indices.size != 0:
                    coulomb += np.sum(self.two_int[a, shared_indices, x, shared_indices])
                    exchange -= np.sum(self.two_int[a, shared_indices, shared_indices, x])
            elif y in shared_indices:
                # NOTE: we can use the following if we assume that the orbitals are real
                # coulomb += 2 * self.two_int[x, a, y, b]
                coulomb += self.two_int[x, a, y, b]
                coulomb += self.two_int[y, a, x, b]
                exchange -= self.two_int[x, a, b, y]
                exchange -= self.two_int[y, a, b, x]

        # two sd's are different by double excitation
        else:
            a, b = diff_sd1
            c, d = diff_sd2

            if x == a:
                coulomb -= self.two_int[y, b, c, d]
                exchange += self.two_int[y, b, d, c]
            elif x == b:
                coulomb -= self.two_int[a, y, c, d]
                exchange += self.two_int[a, y, d, c]
            elif x == c:
                coulomb -= self.two_int[a, b, y, d]
                exchange += self.two_int[a, b, d, y]
            elif x == d:
                coulomb -= self.two_int[a, b, c, y]
                exchange += self.two_int[a, b, y, c]

            if y == a:
                coulomb += self.two_int[x, b, c, d]
                exchange -= self.two_int[x, b, d, c]
            elif y == b:
                coulomb += self.two_int[a, x, c, d]
                exchange -= self.two_int[a, x, d, c]
            elif y == c:
                coulomb += self.two_int[a, b, x, d]
                exchange -= self.two_int[a, b, d, x]
            elif y == d:
                coulomb += self.two_int[a, b, c, x]
                exchange -= self.two_int[a, b, x, c]

        return sign * one_electron, sign * coulomb, sign * exchange
