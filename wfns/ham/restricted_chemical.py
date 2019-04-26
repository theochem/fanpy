r"""Hamiltonian used to describe a chemical system expressed wrt restricted orbitals."""
import itertools as it

import numpy as np
from wfns.backend import slater
from wfns.ham.generalized_chemical import GeneralizedChemicalHamiltonian


class RestrictedChemicalHamiltonian(GeneralizedChemicalHamiltonian):
    r"""Hamiltonian used to describe a typical chemical system expressed wrt restricted orbitals.

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
    nspatial : int
        Number of spatial orbitals.
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

    @property
    def nspin(self):
        """Return the number of spin orbitals.

        Returns
        -------
        nspin : int
            Number of spin orbitals.

        """
        return 2 * self.one_int.shape[0]

    # FIXME: remove sign?
    # FIXME: too many branches, too many statements
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
        # pylint: disable=C0103,R0912,R0915
        if deriv is not None:
            sign = 1 if sign is None else sign
            return sign * self._integrate_sd_sd_deriv(sd1, sd2, deriv)

        nspatial = self.one_int.shape[0]

        sd1 = slater.internal_sd(sd1)
        sd2 = slater.internal_sd(sd2)
        shared_alpha_sd, shared_beta_sd = slater.split_spin(slater.shared_sd(sd1, sd2), nspatial)
        shared_alpha = np.array(slater.occ_indices(shared_alpha_sd))
        shared_beta = np.array(slater.occ_indices(shared_beta_sd))
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

        # two sd's are the same
        if diff_order == 0:
            one_electron, coulomb, exchange = self._integrate_sd_sd_zero(shared_alpha, shared_beta)

        # two sd's are different by single excitation
        elif diff_order == 1:
            one_electron, coulomb, exchange = self._integrate_sd_sd_one(
                diff_sd1, diff_sd2, shared_alpha, shared_beta
            )

        # two sd's are different by double excitation
        else:
            one_electron, coulomb, exchange = self._integrate_sd_sd_two(diff_sd1, diff_sd2)

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
        matrix_indices : 3-tuple of int
            Spin of the orbital and the row and column indices of the antihermitian matrix that
            corresponds to the given parameter index.

        """
        # pylint: disable=C0103
        nspatial = self.one_int[0].shape[0]
        # ind = i
        for k in range(nspatial + 1):  # pragma: no branch
            x = k
            if param_ind - nspatial * (x + 1) + (x + 1) * (x + 2) / 2 < 0:
                break
        # ind_flat = j
        ind_flat = param_ind + (x + 1) * (x + 2) / 2
        y = ind_flat % nspatial

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

            Index of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.

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
        # pylint: disable=C0103
        nspatial = self.one_int[0].shape[0]

        sd1 = slater.internal_sd(sd1)
        sd2 = slater.internal_sd(sd2)
        # NOTE: following are spatial orbital indices
        shared_alpha, shared_beta = map(
            lambda shared_sd: np.array(slater.occ_indices(shared_sd)),
            slater.split_spin(slater.shared_sd(sd1, sd2), nspatial),
        )
        # NOTE: following are spin orbital indices
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
        # NOTE: x and y are spatial orbitals
        x, y = self.param_ind_to_rowcol_ind(deriv)

        one_electron, coulomb, exchange = 0.0, 0.0, 0.0

        # two sd's are the same
        if diff_order == 0:
            one_electron, coulomb, exchange = self._integrate_sd_sd_deriv_zero(
                x, y, shared_alpha, shared_beta
            )

        # two sd's are different by single excitation
        elif diff_order == 1:
            one_electron, coulomb, exchange = self._integrate_sd_sd_deriv_one(
                diff_sd1, diff_sd2, x, y, shared_alpha, shared_beta
            )

        # two sd's are different by double excitation
        else:
            one_electron, coulomb, exchange = self._integrate_sd_sd_deriv_two(
                diff_sd1, diff_sd2, x, y
            )

        return sign * one_electron, sign * coulomb, sign * exchange

    def _integrate_sd_sd_zero(self, shared_alpha, shared_beta):
        """Return integrals of the given Slater determinant with itself.

        Parameters
        ----------
        shared_indices : np.ndarray
            Integer indices of the orbitals that are occupied in the Slater determinant.

        Returns
        -------
        integrals : 3-tuple of float
            Integrals of the given Slater determinant with itself.
            The one-electron (first element), coulomb (second element), and exchange (third element)
            integrals of the given Slater determinant with itself.

        """
        one_electron, coulomb, exchange = 0, 0, 0

        if shared_alpha.size != 0:
            one_electron += np.sum(self.one_int[shared_alpha, shared_alpha])
            coulomb += np.sum(
                np.triu(self._cached_two_int_ijij[shared_alpha[:, None], shared_alpha], k=1)
            )
            exchange -= np.sum(
                np.triu(self._cached_two_int_ijji[shared_alpha[:, None], shared_alpha], k=1)
            )
        if shared_beta.size != 0:
            one_electron += np.sum(self.one_int[shared_beta, shared_beta])
            coulomb += np.sum(
                np.triu(self._cached_two_int_ijij[shared_beta[:, None], shared_beta], k=1)
            )
            exchange -= np.sum(
                np.triu(self._cached_two_int_ijji[shared_beta[:, None], shared_beta], k=1)
            )
        if shared_alpha.size != 0 and shared_beta.size != 0:
            coulomb += np.sum(self._cached_two_int_ijij[shared_alpha[:, None], shared_beta])

        return one_electron, coulomb, exchange

    def _integrate_sd_sd_one(self, diff_sd1, diff_sd2, shared_alpha, shared_beta):
        """Return integrals of the given Slater determinant with its first order excitation.

        Parameters
        ----------
        diff_sd1 : 1-tuple of int
            Index of the orbital that is occupied in the first Slater determinant and not occupied
            in the second.
        diff_sd2 : 1-tuple of int
            Index of the orbital that is occupied in the second Slater determinant and not occupied
            in the first.
        shared_alpha : np.ndarray
            Integer indices of the alpha orbitals that are shared between the first and second
            Slater determinants.
        shared_beta : np.ndarray
            Integer indices of the beta orbitals that are shared between the first and second Slater
            determinants.

        Returns
        -------
        integrals : 3-tuple of float
            The one-electron (first element), coulomb (second element), and exchange (third element)
            integrals of the given Slater determinant with its first order excitations.

        """
        # pylint:disable=C0103
        one_electron, coulomb, exchange = 0, 0, 0
        nspatial = self.nspin // 2

        a, = diff_sd1
        b, = diff_sd2
        # get spatial indices
        spatial_a = slater.spatial_index(a, nspatial)
        spatial_b = slater.spatial_index(b, nspatial)

        if slater.is_alpha(a, nspatial) != slater.is_alpha(b, nspatial):
            return 0.0, 0.0, 0.0

        one_electron += self.one_int[spatial_a, spatial_b]
        if shared_alpha.size != 0:
            coulomb += np.sum(self.two_int[shared_alpha, spatial_a, shared_alpha, spatial_b])
            if slater.is_alpha(a, nspatial):
                exchange -= np.sum(self.two_int[shared_alpha, spatial_a, spatial_b, shared_alpha])
        if shared_beta.size != 0:
            coulomb += np.sum(self.two_int[shared_beta, spatial_a, shared_beta, spatial_b])
            if not slater.is_alpha(a, nspatial):
                exchange -= np.sum(self.two_int[shared_beta, spatial_a, spatial_b, shared_beta])

        return one_electron, coulomb, exchange

    def _integrate_sd_sd_two(self, diff_sd1, diff_sd2):
        """Return integrals of the given Slater determinant with its second order excitation.

        Parameters
        ----------
        diff_sd1 : 2-tuple of int
            Indices of the orbitals that are occupied in the first Slater determinant and not
            occupied in the second.
        diff_sd2 : 2-tuple of int
            Indices of the orbitals that are occupied in the second Slater determinant and not
            occupied in the first.

        Returns
        -------
        integrals : 3-tuple of float
            The one-electron (first element), coulomb (second element), and exchange (third element)
            integrals of the given Slater determinant with itself.

        """
        # pylint:disable=C0103
        one_electron, coulomb, exchange = 0, 0, 0
        nspatial = self.nspin // 2

        a, b = diff_sd1
        c, d = diff_sd2
        # get spatial indices
        spatial_a = slater.spatial_index(a, nspatial)
        spatial_b = slater.spatial_index(b, nspatial)
        spatial_c = slater.spatial_index(c, nspatial)
        spatial_d = slater.spatial_index(d, nspatial)

        if slater.is_alpha(b, nspatial) == slater.is_alpha(d, nspatial) and slater.is_alpha(
            a, nspatial
        ) == slater.is_alpha(c, nspatial):
            coulomb += self.two_int[spatial_a, spatial_b, spatial_c, spatial_d]
        if slater.is_alpha(b, nspatial) == slater.is_alpha(c, nspatial) and slater.is_alpha(
            a, nspatial
        ) == slater.is_alpha(d, nspatial):
            exchange -= self.two_int[spatial_a, spatial_b, spatial_d, spatial_c]

        return one_electron, coulomb, exchange

    def _integrate_sd_sd_deriv_zero(self, x, y, shared_alpha, shared_beta):
        """Return the derivative of the integrals of the given Slater determinant with itself.

        Parameters
        ----------
        x : int
            Row of the antihermitian matrix (of the given spin) at which the integral will be
            derivatized.
        y : int
            Column of the antihermitian matrix (of the given spin) at which the integral will be
            derivatized.
        shared_alpha : np.ndarray
            Integer indices of the alpha orbitals that are occupied by the Slater determinant.
            Dtype must be int.
        shared_beta : np.ndarray
            Integer indices of the beta orbitals that are occupied by the Slater determinant.
            Dtype must be int.

        Returns
        -------
        integrals : 3-tuple of float
            The derivatives (with respect to the given parameter) of the one-electron (first
            element), coulomb (second element), and exchange (third element) integrals of the given
            Slater determinant with itself.

        """
        # pylint:disable=R0912,R0915
        one_electron, coulomb, exchange = 0, 0, 0

        # remove orbitals x and y from the shared indices
        shared_alpha_no_x = shared_alpha[shared_alpha != x]
        shared_alpha_no_y = shared_alpha[shared_alpha != y]
        shared_beta_no_x = shared_beta[shared_beta != x]
        shared_beta_no_y = shared_beta[shared_beta != y]

        if x in shared_alpha:
            one_electron -= 2 * np.real(self.one_int[x, y])
            if shared_beta.size != 0:
                coulomb -= 2 * np.sum(np.real(self.two_int[x, shared_beta, y, shared_beta]))
            if shared_alpha_no_x.size != 0:
                coulomb -= 2 * np.sum(
                    np.real(self.two_int[x, shared_alpha_no_x, y, shared_alpha_no_x])
                )
                exchange += 2 * np.sum(
                    np.real(self.two_int[x, shared_alpha_no_x, shared_alpha_no_x, y])
                )
        if x in shared_beta:
            one_electron -= 2 * np.real(self.one_int[x, y])
            if shared_alpha.size != 0:
                coulomb -= 2 * np.sum(np.real(self.two_int[shared_alpha, x, shared_alpha, y]))
            if shared_beta_no_x.size != 0:
                coulomb -= 2 * np.sum(
                    np.real(self.two_int[x, shared_beta_no_x, y, shared_beta_no_x])
                )
                exchange += 2 * np.sum(
                    np.real(self.two_int[x, shared_beta_no_x, shared_beta_no_x, y])
                )

        if y in shared_alpha:
            one_electron += 2 * np.real(self.one_int[x, y])
            if shared_beta.size != 0:
                coulomb += 2 * np.sum(np.real(self.two_int[x, shared_beta, y, shared_beta]))
            if shared_alpha_no_y.size != 0:
                coulomb += 2 * np.sum(
                    np.real(self.two_int[x, shared_alpha_no_y, y, shared_alpha_no_y])
                )
                exchange -= 2 * np.sum(
                    np.real(self.two_int[x, shared_alpha_no_y, shared_alpha_no_y, y])
                )
        if y in shared_beta:
            one_electron += 2 * np.real(self.one_int[x, y])
            if shared_alpha.size != 0:
                coulomb += 2 * np.sum(np.real(self.two_int[shared_alpha, x, shared_alpha, y]))
            if shared_beta_no_y.size != 0:
                coulomb += 2 * np.sum(
                    np.real(self.two_int[x, shared_beta_no_y, y, shared_beta_no_y])
                )
                exchange -= 2 * np.sum(
                    np.real(self.two_int[x, shared_beta_no_y, shared_beta_no_y, y])
                )

        return one_electron, coulomb, exchange

    def _integrate_sd_sd_deriv_one(self, diff_sd1, diff_sd2, x, y, shared_alpha, shared_beta):
        """Return derivative of integrals of given Slater determinant with its first excitation.

        Parameters
        ----------
        diff_sd1 : 1-tuple of int
            Index of the orbital that is occupied in the first Slater determinant and not occupied
            in the second.
        diff_sd2 : 1-tuple of int
            Index of the orbital that is occupied in the second Slater determinant and not occupied
            in the first.
        x : int
            Row of the antihermitian matrix at which the integral will be derivatized.
        y : int
            Column of the antihermitian matrix at which the integral will be derivatized.
        shared_alpha : np.ndarray
            Integer indices of the alpha orbitals that are shared between the first and second
            Slater determinant.
            Dtype must be int.
        shared_beta : np.ndarray
            Integer indices of the beta orbitals that are shared between the first and second Slater
            Dtype must be int.

        Returns
        -------
        integrals : 3-tuple of float
            The derivatives (with respect to the given parameter) of the one-electron (first
            element), coulomb (second element), and exchange (third element) integrals of the given
            Slater determinant with its first order excitation.

        """
        # pylint:disable=C0103,R0912,R0915
        one_electron, coulomb, exchange = 0, 0, 0
        nspatial = self.nspin // 2

        a, = diff_sd1
        b, = diff_sd2
        spatial_a, spatial_b = map(lambda i: slater.spatial_index(i, nspatial), [a, b])
        spin_a, spin_b = map(lambda i: int(not slater.is_alpha(i, nspatial)), [a, b])

        if spin_a == 0 and spin_b == 0:
            shared_alpha_no_ab = shared_alpha[~np.in1d(shared_alpha, [a, b])]
            shared_beta_no_ab = shared_beta
        elif spin_a == 0 and spin_b == 1:
            shared_alpha_no_ab = shared_alpha[shared_alpha != a]
            shared_beta_no_ab = shared_beta[shared_beta != b]
        elif spin_a == 1 and spin_b == 0:
            shared_alpha_no_ab = shared_alpha[shared_alpha != b]
            shared_beta_no_ab = shared_beta[shared_beta != a]
        else:
            shared_alpha_no_ab = shared_alpha
            shared_beta_no_ab = shared_beta[~np.in1d(shared_beta, [a, b])]

        # selected (spin orbital) x = a
        if x == spatial_a and spin_a == spin_b:
            one_electron -= self.one_int[y, spatial_b]
            # spin of a, b = alpha
            if spin_b == 0:
                if shared_beta_no_ab.size != 0:
                    coulomb -= np.sum(
                        self.two_int[y, shared_beta_no_ab, spatial_b, shared_beta_no_ab]
                    )
                if shared_alpha_no_ab.size != 0:
                    coulomb -= np.sum(
                        self.two_int[y, shared_alpha_no_ab, spatial_b, shared_alpha_no_ab]
                    )
                    exchange += np.sum(
                        self.two_int[y, shared_alpha_no_ab, shared_alpha_no_ab, spatial_b]
                    )
            # spin of a, b = beta
            else:
                if shared_alpha_no_ab.size != 0:
                    coulomb -= np.sum(
                        self.two_int[shared_alpha_no_ab, y, shared_alpha_no_ab, spatial_b]
                    )
                if shared_beta_no_ab.size != 0:
                    coulomb -= np.sum(
                        self.two_int[y, shared_beta_no_ab, spatial_b, shared_beta_no_ab]
                    )
                    exchange += np.sum(
                        self.two_int[y, shared_beta_no_ab, shared_beta_no_ab, spatial_b]
                    )
        # selected (spin orbital) x = b
        elif x == spatial_b and spin_a == spin_b:
            one_electron -= self.one_int[spatial_a, y]
            # spin of a, b = alpha
            if spin_a == 0:
                if shared_beta_no_ab.size != 0:
                    coulomb -= np.sum(
                        self.two_int[spatial_a, shared_beta_no_ab, y, shared_beta_no_ab]
                    )
                if shared_alpha_no_ab.size != 0:
                    coulomb -= np.sum(
                        self.two_int[spatial_a, shared_alpha_no_ab, y, shared_alpha_no_ab]
                    )
                    exchange += np.sum(
                        self.two_int[spatial_a, shared_alpha_no_ab, shared_alpha_no_ab, y]
                    )
            # spin of a, b = beta
            else:
                if shared_alpha_no_ab.size != 0:
                    coulomb -= np.sum(
                        self.two_int[shared_alpha_no_ab, spatial_a, shared_alpha_no_ab, y]
                    )
                if shared_beta_no_ab.size != 0:
                    coulomb -= np.sum(
                        self.two_int[spatial_a, shared_beta_no_ab, y, shared_beta_no_ab]
                    )
                    exchange += np.sum(
                        self.two_int[spatial_a, shared_beta_no_ab, shared_beta_no_ab, y]
                    )
        # non selected (spin orbital) x, spin of a, b = 0
        if spin_a == spin_b == 0:
            if x in shared_alpha:
                coulomb -= self.two_int[x, spatial_a, y, spatial_b]
                coulomb -= self.two_int[x, spatial_b, y, spatial_a]
                exchange += self.two_int[x, spatial_b, spatial_a, y]
                exchange += self.two_int[x, spatial_a, spatial_b, y]
            if x in shared_beta:
                coulomb -= self.two_int[spatial_a, x, spatial_b, y]
                coulomb -= self.two_int[spatial_b, x, spatial_a, y]
        # non selected (spin orbital) x, spin of a, b = 1
        elif spin_a == spin_b == 1:
            if x in shared_beta:
                coulomb -= self.two_int[x, spatial_a, y, spatial_b]
                coulomb -= self.two_int[x, spatial_b, y, spatial_a]
                exchange += self.two_int[x, spatial_b, spatial_a, y]
                exchange += self.two_int[x, spatial_a, spatial_b, y]
            if x in shared_alpha:
                coulomb -= self.two_int[x, spatial_a, y, spatial_b]
                coulomb -= self.two_int[x, spatial_b, y, spatial_a]

        # selected (spin orbital) y = a
        if y == spatial_a and spin_a == spin_b:
            one_electron += self.one_int[x, spatial_b]
            # spin of a, b = alpha
            if spin_b == 0:
                if shared_beta_no_ab.size != 0:
                    coulomb += np.sum(
                        self.two_int[x, shared_beta_no_ab, spatial_b, shared_beta_no_ab]
                    )
                if shared_alpha_no_ab.size != 0:
                    coulomb += np.sum(
                        self.two_int[x, shared_alpha_no_ab, spatial_b, shared_alpha_no_ab]
                    )
                    exchange -= np.sum(
                        self.two_int[x, shared_alpha_no_ab, shared_alpha_no_ab, spatial_b]
                    )
            # spin of a, b = beta
            else:
                if shared_alpha_no_ab.size != 0:
                    coulomb += np.sum(
                        self.two_int[shared_alpha_no_ab, x, shared_alpha_no_ab, spatial_b]
                    )
                if shared_beta_no_ab.size != 0:
                    coulomb += np.sum(
                        self.two_int[x, shared_beta_no_ab, spatial_b, shared_beta_no_ab]
                    )
                    exchange -= np.sum(
                        self.two_int[x, shared_beta_no_ab, shared_beta_no_ab, spatial_b]
                    )
        # selected (spin orbital) x = b
        elif y == spatial_b and spin_a == spin_b:
            one_electron += self.one_int[spatial_a, x]
            # spin of a, b = alpha
            if spin_a == 0:
                if shared_beta_no_ab.size != 0:
                    coulomb += np.sum(
                        self.two_int[spatial_a, shared_beta_no_ab, x, shared_beta_no_ab]
                    )
                if shared_alpha_no_ab.size != 0:
                    coulomb += np.sum(
                        self.two_int[spatial_a, shared_alpha_no_ab, x, shared_alpha_no_ab]
                    )
                    exchange -= np.sum(
                        self.two_int[spatial_a, shared_alpha_no_ab, shared_alpha_no_ab, x]
                    )
            # spin of a, b = beta
            else:
                if shared_alpha_no_ab.size != 0:
                    coulomb += np.sum(
                        self.two_int[shared_alpha_no_ab, spatial_a, shared_alpha_no_ab, x]
                    )
                if shared_beta_no_ab.size != 0:
                    coulomb += np.sum(
                        self.two_int[spatial_a, shared_beta_no_ab, x, shared_beta_no_ab]
                    )
                    exchange -= np.sum(
                        self.two_int[spatial_a, shared_beta_no_ab, shared_beta_no_ab, x]
                    )
        # non selected (spin orbital) x, spin of a, b = 0
        if spin_a == spin_b == 0:
            if y in shared_alpha:
                coulomb += self.two_int[x, spatial_a, y, spatial_b]
                coulomb += self.two_int[x, spatial_b, y, spatial_a]
                exchange -= self.two_int[x, spatial_a, spatial_b, y]
                exchange -= self.two_int[x, spatial_b, spatial_a, y]
            if y in shared_beta:
                coulomb += self.two_int[spatial_a, x, spatial_b, y]
                coulomb += self.two_int[spatial_b, x, spatial_a, y]
        # non selected (spin orbital) x, spin of a, b = 1
        elif spin_a == spin_b == 1:
            if y in shared_beta:
                coulomb += self.two_int[x, spatial_a, y, spatial_b]
                coulomb += self.two_int[x, spatial_b, y, spatial_a]
                exchange -= self.two_int[x, spatial_a, spatial_b, y]
                exchange -= self.two_int[x, spatial_b, spatial_a, y]
            if y in shared_alpha:
                coulomb += self.two_int[x, spatial_a, y, spatial_b]
                coulomb += self.two_int[x, spatial_b, y, spatial_a]

        return one_electron, coulomb, exchange

    def _integrate_sd_sd_deriv_two(self, diff_sd1, diff_sd2, x, y):
        """Return derivative of integrals of given Slater determinant with its second excitation.

        Parameters
        ----------
        diff_sd1 : 2-tuple of int
            Indices of the orbitals that are occupied in the first Slater determinant and not
            occupied in the second.
        diff_sd2 : 2-tuple of int
            Indices of the orbitals that are occupied in the second Slater determinant and not
            occupied in the first.
        x : int
            Row of the antihermitian matrix at which the integral will be derivatized.
        y : int
            Column of the antihermitian matrix at which the integral will be derivatized.

        Returns
        -------
        integrals : 3-tuple of float
            The derivatives (with respect to the given parameter) of the one-electron (first
            element), coulomb (second element), and exchange (third element) integrals of the given
            Slater determinant with its first order excitation.

        """
        # pylint: disable=C0103,R0912,R0915
        one_electron, coulomb, exchange = 0, 0, 0
        nspatial = self.nspin // 2

        a, b = diff_sd1
        c, d = diff_sd2
        (spatial_a, spatial_b, spatial_c, spatial_d) = map(
            lambda i: slater.spatial_index(i, nspatial), [a, b, c, d]
        )
        spin_a, spin_b, spin_c, spin_d = map(
            lambda i: int(not slater.is_alpha(i, nspatial)), [a, b, c, d]
        )

        if x == spatial_a:
            if spin_a == spin_c == spin_b == spin_d == 0:
                coulomb -= self.two_int[y, spatial_b, spatial_c, spatial_d]
                exchange += self.two_int[y, spatial_b, spatial_d, spatial_c]
            elif spin_a == spin_c == 0 and spin_b == spin_d == 1:
                coulomb -= self.two_int[y, spatial_b, spatial_c, spatial_d]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: d will not be alpha if c is beta
            # elif spin_a == spin_c == 1 and spin_b == spin_d == 0:
            #     coulomb -= self.two_int[spatial_b, y, spatial_d, spatial_c]
            # elif spin_a == spin_d == 0 and spin_b == spin_c == 1:
            #     exchange += self.two_int[y, spatial_b, spatial_d, spatial_c]
            # NOTE: b will not be alpha if a is beta (spin of x = spin of a)
            # elif spin_a == spin_d == 1 and spin_b == spin_c == 0:
            #     exchange += self.two_int[spatial_b, y, spatial_c, spatial_d]
            elif spin_a == spin_c == spin_b == spin_d == 1:
                coulomb -= self.two_int[y, spatial_b, spatial_c, spatial_d]
                exchange += self.two_int[y, spatial_b, spatial_d, spatial_c]
        if x == spatial_b:
            if spin_b == spin_c == spin_a == spin_d == 0:
                exchange += self.two_int[y, spatial_a, spatial_c, spatial_d]
                coulomb -= self.two_int[y, spatial_a, spatial_d, spatial_c]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
            # elif spin_b == spin_c == 0 and spin_a == spin_d == 1:
            #     exchange += self.two_int[y, spatial_a, spatial_c, spatial_d]
            # NOTE: d will not be alpha if c is beta
            # elif spin_b == spin_c == 1 and spin_a == spin_d == 0:
            #     exchange += self.two_int[spatial_a, y, spatial_d, spatial_c]
            # elif spin_b == spin_d == 0 and spin_a == spin_c == 1:
            #     coulomb -= self.two_int[y, spatial_a, spatial_d, spatial_c]
            elif spin_b == spin_d == 1 and spin_a == spin_c == 0:
                coulomb -= self.two_int[spatial_a, y, spatial_c, spatial_d]
            elif spin_b == spin_c == spin_a == spin_d == 1:
                exchange += self.two_int[y, spatial_a, spatial_c, spatial_d]
                coulomb -= self.two_int[y, spatial_a, spatial_d, spatial_c]
        if x == spatial_c:
            if spin_c == spin_a == spin_b == spin_d == 0:
                coulomb -= self.two_int[spatial_a, spatial_b, y, spatial_d]
                exchange += self.two_int[spatial_b, spatial_a, y, spatial_d]
            elif spin_c == spin_a == 0 and spin_b == spin_d == 1:
                coulomb -= self.two_int[spatial_a, spatial_b, y, spatial_d]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # elif spin_c == spin_a == 1 and spin_b == spin_d == 0:
            #     coulomb -= self.two_int[spatial_b, spatial_a, spatial_d, y]
            # elif spin_c == spin_b == 0 and spin_a == spin_d == 1:
            #     exchange += self.two_int[spatial_b, spatial_a, y, spatial_d]
            # NOTE: b will not be alpha if a is beta
            # elif spin_c == spin_b == 1 and spin_a == spin_d == 0:
            #     exchange += self.two_int[spatial_a, spatial_b, spatial_d, y]
            elif spin_c == spin_a == spin_b == spin_d == 1:
                coulomb -= self.two_int[spatial_a, spatial_b, y, spatial_d]
                exchange += self.two_int[spatial_b, spatial_a, y, spatial_d]
        if x == spatial_d:
            if spin_d == spin_a == spin_b == spin_c == 0:
                exchange += self.two_int[spatial_a, spatial_b, y, spatial_c]
                coulomb -= self.two_int[spatial_b, spatial_a, y, spatial_c]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
            # elif spin_d == spin_a == 0 and spin_b == spin_c == 1:
            #     exchange += self.two_int[spatial_a, spatial_b, y, spatial_c]
            # NOTE: b will not be alpha if a is beta
            # elif spin_d == spin_a == 1 and spin_b == spin_c == 0:
            #     exchange += self.two_int[spatial_b, spatial_a, spatial_c, y]
            # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
            # elif spin_d == spin_b == 0 and spin_a == spin_c == 1:
            #     coulomb -= self.two_int[spatial_b, spatial_a, y, spatial_c]
            elif spin_d == spin_b == 1 and spin_a == spin_c == 0:
                coulomb -= self.two_int[spatial_a, spatial_b, spatial_c, y]
            elif spin_d == spin_a == spin_b == spin_c == 1:
                exchange += self.two_int[spatial_a, spatial_b, y, spatial_c]
                coulomb -= self.two_int[spatial_b, spatial_a, y, spatial_c]

        if y == spatial_a:
            if spin_a == spin_c == spin_b == spin_d == 0:
                coulomb += self.two_int[x, spatial_b, spatial_c, spatial_d]
                exchange -= self.two_int[x, spatial_b, spatial_d, spatial_c]
            elif spin_a == spin_c == 0 and spin_b == spin_d == 1:
                coulomb += self.two_int[x, spatial_b, spatial_c, spatial_d]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: d will not be alpha if c is beta
            # elif spin_a == spin_c == 1 and spin_b == spin_d == 0:
            #     coulomb += self.two_int[spatial_b, x, spatial_d, spatial_c]
            # elif spin_a == spin_d == 0 and spin_b == spin_c == 1:
            #     exchange -= self.two_int[x, spatial_b, spatial_d, spatial_c]
            # NOTE: b will not be alpha if a is beta (spin of x = spin of a)
            # elif spin_a == spin_d == 1 and spin_b == spin_c == 0:
            #     exchange -= self.two_int[spatial_b, x, spatial_c, spatial_d]
            elif spin_a == spin_c == spin_b == spin_d == 1:
                coulomb += self.two_int[x, spatial_b, spatial_c, spatial_d]
                exchange -= self.two_int[x, spatial_b, spatial_d, spatial_c]
        if y == spatial_b:
            if spin_b == spin_c == spin_a == spin_d == 0:
                exchange -= self.two_int[x, spatial_a, spatial_c, spatial_d]
                coulomb += self.two_int[x, spatial_a, spatial_d, spatial_c]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
            # elif spin_b == spin_c == 0 and spin_a == spin_d == 1:
            #     exchange -= self.two_int[x, spatial_a, spatial_c, spatial_d]
            # NOTE: d will not be alpha if c is beta
            # elif spin_b == spin_c == 1 and spin_a == spin_d == 0:
            #     exchange -= self.two_int[spatial_a, x, spatial_d, spatial_c]
            # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
            # elif spin_b == spin_d == 0 and spin_a == spin_c == 1:
            #     coulomb += self.two_int[x, spatial_a, spatial_d, spatial_c]
            elif spin_b == spin_d == 1 and spin_a == spin_c == 0:
                coulomb += self.two_int[spatial_a, x, spatial_c, spatial_d]
            elif spin_b == spin_c == spin_a == spin_d == 1:
                exchange -= self.two_int[x, spatial_a, spatial_c, spatial_d]
                coulomb += self.two_int[x, spatial_a, spatial_d, spatial_c]
        if y == spatial_c:
            if spin_c == spin_a == spin_b == spin_d == 0:
                coulomb += self.two_int[spatial_a, spatial_b, x, spatial_d]
                exchange -= self.two_int[spatial_b, spatial_a, x, spatial_d]
            elif spin_c == spin_a == 0 and spin_b == spin_d == 1:
                coulomb += self.two_int[spatial_a, spatial_b, x, spatial_d]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: d will not be alpha if c is beta (spin of x = spin of c)
            # elif spin_c == spin_a == 1 and spin_b == spin_d == 0:
            #     coulomb += self.two_int[spatial_b, spatial_a, spatial_d, x]
            # NOTE: a will not be alpha if b is beta
            # elif spin_c == spin_b == 0 and spin_a == spin_d == 1:
            #     exchange -= self.two_int[spatial_b, spatial_a, x, spatial_d]
            # NOTE: d will not be alpha if c is beta (spin of x = spin of c)
            # elif spin_c == spin_b == 1 and spin_a == spin_d == 0:
            #     exchange -= self.two_int[spatial_a, spatial_b, spatial_d, x]
            elif spin_c == spin_a == spin_b == spin_d == 1:
                coulomb += self.two_int[spatial_a, spatial_b, x, spatial_d]
                exchange -= self.two_int[spatial_b, spatial_a, x, spatial_d]
        if y == spatial_d:
            if spin_d == spin_a == spin_b == spin_c == 0:
                exchange -= self.two_int[spatial_a, spatial_b, x, spatial_c]
                coulomb += self.two_int[spatial_b, spatial_a, x, spatial_c]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
            # elif spin_d == spin_a == 0 and spin_b == spin_c == 1:
            #     exchange -= self.two_int[spatial_a, spatial_b, x, spatial_c]
            # NOTE: b will not be alpha if a is beta
            # elif spin_d == spin_a == 1 and spin_b == spin_c == 0:
            #     exchange -= self.two_int[spatial_b, spatial_a, spatial_c, x]
            # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
            # elif spin_d == spin_b == 0 and spin_a == spin_c == 1:
            #     coulomb += self.two_int[spatial_b, spatial_a, x, spatial_c]
            elif spin_d == spin_b == 1 and spin_a == spin_c == 0:
                coulomb += self.two_int[spatial_a, spatial_b, spatial_c, x]
            elif spin_d == spin_a == spin_b == spin_c == 1:
                exchange -= self.two_int[spatial_a, spatial_b, x, spatial_c]
                coulomb += self.two_int[spatial_b, spatial_a, x, spatial_c]

        return one_electron, coulomb, exchange

    def _integrate_sd_sds_zero(self, occ_alpha, occ_beta):
        """Return the integrals of the given Slater determinant with itself.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, 1)
            Integrals of the given Slater determinant with itself.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.

        """
        one_electron = np.sum(self.one_int[occ_alpha, occ_alpha])
        one_electron += np.sum(self.one_int[occ_beta, occ_beta])

        coulomb = np.sum(np.triu(self._cached_two_int_ijij[occ_alpha[:, None], occ_alpha], k=1))
        coulomb += np.sum(self._cached_two_int_ijij[occ_alpha[:, None], occ_beta])
        coulomb += np.sum(np.triu(self._cached_two_int_ijij[occ_beta[:, None], occ_beta], k=1))

        exchange = -np.sum(np.triu(self._cached_two_int_ijji[occ_alpha[:, None], occ_alpha], k=1))
        exchange -= np.sum(np.triu(self._cached_two_int_ijji[occ_beta[:, None], occ_beta], k=1))

        return np.array([[one_electron], [coulomb], [exchange]])

    def _integrate_sd_sds_one_alpha(self, occ_alpha, occ_beta, vir_alpha):
        """Return the integrals of the given Slater determinant with its first order excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, M)
            Integrals of the given Slater determinant with its first order excitations involving the
            alpha spin orbitals.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.
            Second index corresponds to the first order excitations of the given Slater determinant.
            The excitations are ordered by the occupied orbital then the virtual orbital. For
            example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the ordering of the
            excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number of first order
            excitations of the given Slater determinants.

        """
        shared_alpha = slater.shared_indices_remove_one_index(occ_alpha)

        nspatial = self.nspin // 2
        occ_indices = np.hstack(
            [
                slater.spatial_to_spin_indices(occ_alpha, nspatial, to_beta=False),
                slater.spatial_to_spin_indices(occ_beta, nspatial, to_beta=True),
            ]
        )
        sign_a = slater.sign_excite_array(
            occ_indices,
            slater.spatial_to_spin_indices(occ_alpha[:, None], nspatial, to_beta=False),
            slater.spatial_to_spin_indices(vir_alpha[:, None], nspatial, to_beta=False),
            self.nspin,
        ).ravel()

        one_electron_a = self.one_int[occ_alpha[:, np.newaxis], vir_alpha[np.newaxis, :]].ravel()

        coulomb_a = np.sum(
            self.two_int[
                shared_alpha[:, :, np.newaxis],
                occ_alpha[:, np.newaxis, np.newaxis],
                shared_alpha[:, :, np.newaxis],
                vir_alpha[np.newaxis, np.newaxis, :],
            ],
            axis=1,
        ).ravel()
        coulomb_a += np.sum(
            self.two_int[
                occ_alpha[:, np.newaxis, np.newaxis],
                occ_beta[np.newaxis, :, np.newaxis],
                vir_alpha[np.newaxis, np.newaxis, :],
                occ_beta[np.newaxis, :, np.newaxis],
            ],
            axis=1,
        ).ravel()

        exchange_a = -np.sum(
            self.two_int[
                shared_alpha[:, :, np.newaxis],
                occ_alpha[:, np.newaxis, np.newaxis],
                vir_alpha[np.newaxis, np.newaxis, :],
                shared_alpha[:, :, np.newaxis],
            ],
            axis=1,
        ).ravel()

        return sign_a[None, :] * np.array([one_electron_a, coulomb_a, exchange_a])

    def _integrate_sd_sds_one_beta(self, occ_alpha, occ_beta, vir_beta):
        """Return the integrals of the given Slater determinant with its first order excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, M)
            Integrals of the given Slater determinant with its first order excitations involving the
            beta spin orbitals.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.
            Second index corresponds to the first order excitations of the given Slater determinant.
            The excitations are ordered by the occupied orbital then the virtual orbital. For
            example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the ordering of the
            excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number of first order
            excitations of the given Slater determinants.

        """
        shared_beta = slater.shared_indices_remove_one_index(occ_beta)

        nspatial = self.nspin // 2
        occ_indices = np.hstack(
            [
                slater.spatial_to_spin_indices(occ_alpha, nspatial, to_beta=False),
                slater.spatial_to_spin_indices(occ_beta, nspatial, to_beta=True),
            ]
        )
        sign_b = slater.sign_excite_array(
            occ_indices,
            slater.spatial_to_spin_indices(occ_beta[:, None], nspatial, to_beta=True),
            slater.spatial_to_spin_indices(vir_beta[:, None], nspatial, to_beta=True),
            self.nspin,
        ).ravel()

        one_electron_b = self.one_int[occ_beta[:, np.newaxis], vir_beta[np.newaxis, :]].ravel()

        coulomb_b = np.sum(
            self.two_int[
                shared_beta[:, :, np.newaxis],
                occ_beta[:, np.newaxis, np.newaxis],
                shared_beta[:, :, np.newaxis],
                vir_beta[np.newaxis, np.newaxis, :],
            ],
            axis=1,
        ).ravel()
        coulomb_b += np.sum(
            self.two_int[
                occ_alpha[np.newaxis, :, np.newaxis],
                occ_beta[:, np.newaxis, np.newaxis],
                occ_alpha[np.newaxis, :, np.newaxis],
                vir_beta[np.newaxis, np.newaxis, :],
            ],
            axis=1,
        ).ravel()

        exchange_b = -np.sum(
            self.two_int[
                shared_beta[:, :, np.newaxis],
                occ_beta[:, np.newaxis, np.newaxis],
                vir_beta[np.newaxis, np.newaxis, :],
                shared_beta[:, :, np.newaxis],
            ],
            axis=1,
        ).ravel()

        return sign_b[None, :] * np.array([one_electron_b, coulomb_b, exchange_b])

    def _integrate_sd_sds_two_aa(self, occ_alpha, occ_beta, vir_alpha):
        """Return the integrals of a Slater determinant with its second order (alpha) excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(2, M)
            Integrals of the given Slater determinant with its second order excitations involving
            the alpha spin orbitals.
            First index corresponds to the coulomb (index 0) and exchange (index 1) integrals.
            Second index corresponds to the second order excitations of the given Slater
            determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
            the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
            3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
            the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=C0103
        nspatial = self.nspin // 2
        occ_indices = np.hstack(
            [
                slater.spatial_to_spin_indices(occ_alpha, nspatial, to_beta=False),
                slater.spatial_to_spin_indices(occ_beta, nspatial, to_beta=True),
            ]
        )

        annihilators = np.array(list(it.combinations(occ_alpha, 2)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.combinations(vir_alpha, 2)))
        c = creators[:, 0]
        d = creators[:, 1]

        sign = slater.sign_excite_array(
            occ_indices,
            slater.spatial_to_spin_indices(annihilators, nspatial, to_beta=False),
            slater.spatial_to_spin_indices(creators, nspatial, to_beta=False),
            self.nspin,
        ).ravel()

        coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()
        exchange = -self.two_int[a[:, None], b[:, None], d[None, :], c[None, :]].ravel()

        return sign[None, :] * np.array([coulomb, exchange])

    def _integrate_sd_sds_two_ab(self, occ_alpha, occ_beta, vir_alpha, vir_beta):
        """Return the integrals of a SD with its second order (alpha and beta) excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(M,)
            Coulomb integrals of the given Slater determinant with its second order excitations
            involving both alpha and beta orbitals
            Second index corresponds to the second order excitations of the given Slater
            determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
            the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
            3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
            the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=C0103
        nspatial = self.nspin // 2
        occ_indices = np.hstack(
            [
                slater.spatial_to_spin_indices(occ_alpha, nspatial, to_beta=False),
                slater.spatial_to_spin_indices(occ_beta, nspatial, to_beta=True),
            ]
        )

        annihilators = np.array(list(it.product(occ_alpha, occ_beta)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.product(vir_alpha, vir_beta)))
        c = creators[:, 0]
        d = creators[:, 1]

        sign = slater.sign_excite_array(
            occ_indices,
            np.array(
                [
                    slater.spatial_to_spin_indices(a, nspatial, to_beta=False),
                    slater.spatial_to_spin_indices(b, nspatial, to_beta=True),
                ]
            ).T,
            np.array(
                [
                    slater.spatial_to_spin_indices(c, nspatial, to_beta=False),
                    slater.spatial_to_spin_indices(d, nspatial, to_beta=True),
                ]
            ).T,
            self.nspin,
        ).ravel()
        coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()

        return sign * coulomb

    def _integrate_sd_sds_two_bb(self, occ_alpha, occ_beta, vir_beta):
        """Return the integrals of a Slater determinant with its second order (beta) excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(2, M)
            Integrals of the given Slater determinant with its second order excitations involving
            the beta spin orbitals.
            First index corresponds to the coulomb (index 0) and exchange (index 1) integrals.
            Second index corresponds to the second order excitations of the given Slater
            determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
            the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
            3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
            the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=C0103
        nspatial = self.nspin // 2
        occ_indices = np.hstack(
            [
                slater.spatial_to_spin_indices(occ_alpha, nspatial, to_beta=False),
                slater.spatial_to_spin_indices(occ_beta, nspatial, to_beta=True),
            ]
        )

        annihilators = np.array(list(it.combinations(occ_beta, 2)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.combinations(vir_beta, 2)))
        c = creators[:, 0]
        d = creators[:, 1]

        sign = slater.sign_excite_array(
            occ_indices,
            slater.spatial_to_spin_indices(annihilators, nspatial, to_beta=True),
            slater.spatial_to_spin_indices(creators, nspatial, to_beta=True),
            self.nspin,
        ).ravel()
        coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()
        exchange = -self.two_int[a[:, None], b[:, None], d[None, :], c[None, :]].ravel()

        return sign[None, :] * np.array([coulomb, exchange])

