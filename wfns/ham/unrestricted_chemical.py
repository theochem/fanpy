r"""Hamiltonian used to describe a chemical system expressed wrt unrestricted orbitals."""
import numpy as np
from wfns.backend import slater, math_tools
from wfns.ham.unrestricted_base import BaseUnrestrictedHamiltonian


class UnrestrictedChemicalHamiltonian(BaseUnrestrictedHamiltonian):
    r"""Hamiltonian used to describe a typical chemical system expressed wrt unrestricted orbitals.

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
        one_int : 2-tuple of np.ndarray(K, K)
            One electron integrals.
        two_int : 3-tuple of np.ndarray(K, K, K, K)
            Two electron integrals.
        energy_nuc_nuc : {float, None}
            Nuclear nuclear repulsion energy.
            Default is `0.0`.

        """
        super().__init__(one_int, two_int, energy_nuc_nuc=energy_nuc_nuc)
        self.set_ref_ints()
        self.assign_params(params=params)

    def set_ref_ints(self):
        """Store the current integrals as the reference from which orbitals will be rotated."""
        self._ref_one_int = [np.copy(self.one_int[0]), np.copy(self.one_int[1])]
        self._ref_two_int = [
            np.copy(self.two_int[0]),
            np.copy(self.two_int[1]),
            np.copy(self.two_int[2]),
        ]

    def cache_two_ints(self):
        """Cache away contractions of the two electron integrals."""
        # store away tensor contractions
        indices = np.arange(self.one_int[0].shape[0])
        self._cached_two_int_0_ijij = self.two_int[0][
            indices[:, None], indices, indices[:, None], indices
        ]
        self._cached_two_int_1_ijij = self.two_int[1][
            indices[:, None], indices, indices[:, None], indices
        ]
        self._cached_two_int_2_ijij = self.two_int[2][
            indices[:, None], indices, indices[:, None], indices
        ]
        self._cached_two_int_0_ijji = self.two_int[0][
            indices[:, None], indices, indices, indices[:, None]
        ]
        self._cached_two_int_2_ijji = self.two_int[2][
            indices[:, None], indices, indices, indices[:, None]
        ]

    def assign_params(self, params=None):
        """Transform the integrals with a unitary matrix that corresponds to the given parameters.

        Parameters
        ----------
        params : {np.ndarray, None}
            Significant elements of the anti-Hermitian matrix. Integrals will be transformed with
            the Unitary matrix that corresponds to the anti-Hermitian matrix.
            First `K*(K-1)/2` elements correspond to the transformation of the alpha orbitals.
            Last `K*(K-1)/2` elements correspond to the transformation of the beta orbitals.

        Raises
        ------
        ValueError
            If parameters is not a one-dimensional numpy array with K*(K-1) elements, where K is the
            number of orbitals.

        """
        num_orbs = self.one_int[0].shape[0]
        num_params = num_orbs * (num_orbs - 1)

        if params is None:
            params = np.zeros(num_params)

        if not (isinstance(params, np.ndarray) and params.ndim == 1 and params.size == num_params):
            raise ValueError(
                "Parameters for orbital rotation must be a one-dimension numpy array "
                "with {0}=K*(K-1) elements, where K is the number of "
                "orbitals.".format(num_params)
            )

        # assign parameters
        self.params = params

        # revert integrals back to original
        self.assign_integrals(
            [np.copy(self._ref_one_int[0]), np.copy(self._ref_one_int[1])],
            [
                np.copy(self._ref_two_int[0]),
                np.copy(self._ref_two_int[1]),
                np.copy(self._ref_two_int[2]),
            ],
        )

        # convert antihermitian part to unitary matrix.
        unitary_alpha = math_tools.unitary_matrix(params[: num_params // 2])
        unitary_beta = math_tools.unitary_matrix(params[num_params // 2 :])

        # transform integrals
        self.orb_rotate_matrix([unitary_alpha, unitary_beta])

        # cache two electron integrals
        self.cache_two_ints()

    # FIXME: remove sign?
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
        if deriv is not None:
            sign = 1 if sign is None else sign
            return sign * self._integrate_sd_sd_deriv(sd1, sd2, deriv)

        nspatial = self.one_int[0].shape[0]

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

        one_electron, coulomb, exchange = 0.0, 0.0, 0.0

        # two sd's are the same
        if diff_order == 0:
            if shared_alpha.size != 0:
                one_electron += np.sum(self.one_int[0][shared_alpha, shared_alpha])
                coulomb += np.sum(
                    np.triu(self._cached_two_int_0_ijij[shared_alpha[:, None], shared_alpha], k=1)
                )
                exchange -= np.sum(
                    np.triu(self._cached_two_int_0_ijji[shared_alpha[:, None], shared_alpha], k=1)
                )
            if shared_beta.size != 0:
                one_electron += np.sum(self.one_int[1][shared_beta, shared_beta])
                coulomb += np.sum(
                    np.triu(self._cached_two_int_2_ijij[shared_beta[:, None], shared_beta], k=1)
                )
                exchange -= np.sum(
                    np.triu(self._cached_two_int_2_ijji[shared_beta[:, None], shared_beta], k=1)
                )
            if shared_alpha.size != 0 and shared_beta.size != 0:
                coulomb += np.sum(self._cached_two_int_1_ijij[shared_alpha[:, None], shared_beta])

        # two sd's are different by single excitation
        elif diff_order == 1:
            a, = diff_sd1
            b, = diff_sd2
            spatial_a = slater.spatial_index(a, nspatial)
            spatial_b = slater.spatial_index(b, nspatial)

            if slater.is_alpha(a, nspatial) != slater.is_alpha(b, nspatial):
                return 0.0, 0.0, 0.0

            if slater.is_alpha(a, nspatial):
                one_electron += self.one_int[0][spatial_a, spatial_b]
                if shared_alpha.size != 0:
                    coulomb += np.sum(
                        self.two_int[0][shared_alpha, spatial_a, shared_alpha, spatial_b]
                    )
                    exchange -= np.sum(
                        self.two_int[0][shared_alpha, spatial_a, spatial_b, shared_alpha]
                    )
                if shared_beta.size != 0:
                    coulomb += np.sum(
                        self.two_int[1][spatial_a, shared_beta, spatial_b, shared_beta]
                    )
            else:
                one_electron += self.one_int[1][spatial_a, spatial_b]
                if shared_alpha.size != 0:
                    coulomb += np.sum(
                        self.two_int[1][shared_alpha, spatial_a, shared_alpha, spatial_b]
                    )
                if shared_beta.size != 0:
                    coulomb += np.sum(
                        self.two_int[2][shared_beta, spatial_a, shared_beta, spatial_b]
                    )
                    exchange -= np.sum(
                        self.two_int[2][shared_beta, spatial_a, spatial_b, shared_beta]
                    )

        # two sd's are different by double excitation
        else:
            a, b = diff_sd1
            c, d = diff_sd2

            if slater.is_alpha(a, nspatial) and slater.is_alpha(b, nspatial):
                spin_index = 0
            elif slater.is_alpha(a, nspatial) and not slater.is_alpha(b, nspatial):
                spin_index = 1
            elif not slater.is_alpha(a, nspatial) and slater.is_alpha(
                b, nspatial
            ):  # pragma: no cover
                # NOTE: Since a < b by construction from slater.diff_orbs and alpha orbitals are
                #       indexed (by convention) to be less than the beta orbitals, `a` cannot be a
                #       beta orbital if `b` is an alpha orbital.
                #       However, in the case that the conventions change, this block will ensure
                #       that the code can still work and no assumption will be made regarding the
                #       structure of the slater determinant
                spin_index = 1
                # swap indices for the alpha-beta-alpha-beta integrals
                a, b, c, d = b, a, d, c
            else:
                spin_index = 2

            spatial_a = slater.spatial_index(a, nspatial)
            spatial_b = slater.spatial_index(b, nspatial)
            spatial_c = slater.spatial_index(c, nspatial)
            spatial_d = slater.spatial_index(d, nspatial)

            if slater.is_alpha(b, nspatial) == slater.is_alpha(d, nspatial) and slater.is_alpha(
                a, nspatial
            ) == slater.is_alpha(c, nspatial):
                coulomb += self.two_int[spin_index][spatial_a, spatial_b, spatial_c, spatial_d]
            if slater.is_alpha(b, nspatial) == slater.is_alpha(c, nspatial) and slater.is_alpha(
                a, nspatial
            ) == slater.is_alpha(d, nspatial):
                exchange -= self.two_int[spin_index][spatial_a, spatial_b, spatial_d, spatial_c]

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
        nspatial = self.one_int[0].shape[0]
        if param_ind < nspatial * (nspatial - 1) / 2:
            spin_ind = 0
        else:
            param_ind -= nspatial * (nspatial - 1) // 2
            spin_ind = 1

        # ind = i
        for k in range(nspatial + 1):  # pragma: no branch
            x = k
            if param_ind - nspatial * (x + 1) + (x + 1) * (x + 2) / 2 < 0:
                break
        # ind_flat = j
        ind_flat = param_ind + (x + 1) * (x + 2) / 2
        y = ind_flat % nspatial

        return spin_ind, int(x), int(y)

    # TODO: Much of the following function can be shortened by using impure functions (function with
    # a side effect) instead
    def _integrate_sd_sd_deriv(self, sd1, sd2, deriv):  # pylint: disable=too-many-branches
        r"""Derivative of the CI matrix element with respect to the antihermitian elements.

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
        nspatial = self.one_int[0].shape[0]

        sd1 = slater.internal_sd(sd1)
        sd2 = slater.internal_sd(sd2)
        # NOTE: shared_alpha and shared_beta contain spatial orbital indices
        shared_alpha, shared_beta = map(
            lambda shared_sd: np.array(slater.occ_indices(shared_sd)),
            slater.split_spin(slater.shared_sd(sd1, sd2), nspatial),
        )
        # NOTE: diff_sd1 and diff_sd2 contain spin orbitals
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
        spin_ind, x, y = self.param_ind_to_rowcol_ind(deriv)

        one_electron, coulomb, exchange = 0.0, 0.0, 0.0

        # two sd's are the same
        if diff_order == 0:
            # remove orbitals x and y from the shared indices
            if spin_ind == 0:
                shared_alpha_no_x = shared_alpha[shared_alpha != x]
                shared_alpha_no_y = shared_alpha[shared_alpha != y]
            if spin_ind == 1:
                shared_beta_no_x = shared_beta[shared_beta != x]
                shared_beta_no_y = shared_beta[shared_beta != y]

            if spin_ind == 0 and x in shared_alpha:
                one_electron -= 2 * np.real(self.one_int[0][x, y])
                if shared_beta.size != 0:
                    coulomb -= 2 * np.sum(np.real(self.two_int[1][x, shared_beta, y, shared_beta]))
                if shared_alpha_no_x.size != 0:
                    coulomb -= 2 * np.sum(
                        np.real(self.two_int[0][x, shared_alpha_no_x, y, shared_alpha_no_x])
                    )
                    exchange += 2 * np.sum(
                        np.real(self.two_int[0][x, shared_alpha_no_x, shared_alpha_no_x, y])
                    )
            elif spin_ind == 1 and x in shared_beta:
                one_electron -= 2 * np.real(self.one_int[1][x, y])
                if shared_alpha.size != 0:
                    coulomb -= 2 * np.sum(
                        np.real(self.two_int[1][shared_alpha, x, shared_alpha, y])
                    )
                if shared_beta_no_x.size != 0:
                    coulomb -= 2 * np.sum(
                        np.real(self.two_int[2][x, shared_beta_no_x, y, shared_beta_no_x])
                    )
                    exchange += 2 * np.sum(
                        np.real(self.two_int[2][x, shared_beta_no_x, shared_beta_no_x, y])
                    )

            if spin_ind == 0 and y in shared_alpha:
                one_electron += 2 * np.real(self.one_int[0][x, y])
                if shared_beta.size != 0:
                    coulomb += 2 * np.sum(np.real(self.two_int[1][x, shared_beta, y, shared_beta]))
                if shared_alpha_no_y.size != 0:
                    coulomb += 2 * np.sum(
                        np.real(self.two_int[0][x, shared_alpha_no_y, y, shared_alpha_no_y])
                    )
                    exchange -= 2 * np.sum(
                        np.real(self.two_int[0][x, shared_alpha_no_y, shared_alpha_no_y, y])
                    )
            elif spin_ind == 1 and y in shared_beta:
                one_electron += 2 * np.real(self.one_int[1][x, y])
                if shared_alpha.size != 0:
                    coulomb += 2 * np.sum(
                        np.real(self.two_int[1][shared_alpha, x, shared_alpha, y])
                    )
                if shared_beta_no_y.size != 0:
                    coulomb += 2 * np.sum(
                        np.real(self.two_int[2][x, shared_beta_no_y, y, shared_beta_no_y])
                    )
                    exchange -= 2 * np.sum(
                        np.real(self.two_int[2][x, shared_beta_no_y, shared_beta_no_y, y])
                    )
        # two sd's are different by single excitation
        elif diff_order == 1:
            a, = diff_sd1
            b, = diff_sd2
            spatial_a, spatial_b = map(lambda i: slater.spatial_index(i, nspatial), [a, b])
            spin_a, spin_b = map(lambda i: int(not slater.is_alpha(i, nspatial)), [a, b])

            # selected (spin orbital) x = a
            if x == spatial_a and spin_ind == spin_b == spin_a:
                # spin of x, y, a, b = alpha
                if spin_ind == 0:
                    one_electron -= self.one_int[0][y, spatial_b]
                    if shared_beta.size != 0:
                        coulomb -= np.sum(self.two_int[1][y, shared_beta, spatial_b, shared_beta])
                    if shared_alpha.size != 0:
                        coulomb -= np.sum(self.two_int[0][y, shared_alpha, spatial_b, shared_alpha])
                        exchange += np.sum(
                            self.two_int[0][y, shared_alpha, shared_alpha, spatial_b]
                        )
                # spin of x, y, a, b = beta
                else:
                    one_electron -= self.one_int[1][y, spatial_b]
                    if shared_alpha.size != 0:
                        coulomb -= np.sum(self.two_int[1][shared_alpha, y, shared_alpha, spatial_b])
                    if shared_beta.size != 0:
                        coulomb -= np.sum(self.two_int[2][y, shared_beta, spatial_b, shared_beta])
                        exchange += np.sum(self.two_int[2][y, shared_beta, shared_beta, spatial_b])
            # selected (spin orbital) x = b
            elif x == spatial_b and spin_ind == spin_a == spin_b:
                # spin of x, y, a, b = alpha
                if spin_ind == 0:
                    one_electron -= self.one_int[0][spatial_a, y]
                    if shared_beta.size != 0:
                        coulomb -= np.sum(self.two_int[1][spatial_a, shared_beta, y, shared_beta])
                    if shared_alpha.size != 0:
                        coulomb -= np.sum(self.two_int[0][spatial_a, shared_alpha, y, shared_alpha])
                        exchange += np.sum(
                            self.two_int[0][spatial_a, shared_alpha, shared_alpha, y]
                        )
                # spin of x, y, a, b = beta
                else:
                    one_electron -= self.one_int[1][spatial_a, y]
                    if shared_alpha.size != 0:
                        coulomb -= np.sum(self.two_int[1][shared_alpha, spatial_a, shared_alpha, y])
                    if shared_beta.size != 0:
                        coulomb -= np.sum(self.two_int[2][spatial_a, shared_beta, y, shared_beta])
                        exchange += np.sum(self.two_int[2][spatial_a, shared_beta, shared_beta, y])
            # spin of x, y, a, b = alpha and selected (spin orbital) != a, b
            elif (
                spin_ind == 0
                and spin_a == spin_b == 0
                and x in shared_alpha
                and x not in [spatial_a, spatial_b]
            ):
                coulomb -= self.two_int[0][x, spatial_a, y, spatial_b]
                coulomb -= self.two_int[0][x, spatial_b, y, spatial_a]
                exchange += self.two_int[0][x, spatial_b, spatial_a, y]
                exchange += self.two_int[0][x, spatial_a, spatial_b, y]
            # spin of x, y = alpha and spin of a, b = alpha and selected (spin orbital) != a, b
            elif spin_ind == 0 and spin_a == spin_b == 1 and x in shared_alpha:
                coulomb -= self.two_int[1][x, spatial_a, y, spatial_b]
                coulomb -= self.two_int[1][x, spatial_b, y, spatial_a]
            # spin of x, y = beta and spin of a, b = beta and selected (spin orbital) != a, b
            elif spin_ind == 1 and spin_a == spin_b == 0 and x in shared_beta:
                coulomb -= self.two_int[1][spatial_a, x, spatial_b, y]
                coulomb -= self.two_int[1][spatial_b, x, spatial_a, y]
            # spin of x, y, a, b = beta and selected (spin orbital) != a, b
            elif (
                spin_ind == 1
                and spin_a == spin_b == 1
                and x in shared_beta
                and x not in [spatial_a, spatial_b]
            ):
                coulomb -= self.two_int[2][x, spatial_a, y, spatial_b]
                coulomb -= self.two_int[2][x, spatial_b, y, spatial_a]
                exchange += self.two_int[2][x, spatial_b, spatial_a, y]
                exchange += self.two_int[2][x, spatial_a, spatial_b, y]

            # selected (spin orbital) y = a
            if y == spatial_a and spin_ind == spin_b == spin_a:
                # spin of x, y, a, b = alpha
                if spin_ind == 0:
                    one_electron += self.one_int[0][x, spatial_b]
                    if shared_beta.size != 0:
                        coulomb += np.sum(self.two_int[1][x, shared_beta, spatial_b, shared_beta])
                    if shared_alpha.size != 0:
                        coulomb += np.sum(self.two_int[0][x, shared_alpha, spatial_b, shared_alpha])
                        exchange -= np.sum(
                            self.two_int[0][x, shared_alpha, shared_alpha, spatial_b]
                        )
                # spin of x, y, a, b = beta
                else:
                    one_electron += self.one_int[1][x, spatial_b]
                    if shared_alpha.size != 0:
                        coulomb += np.sum(self.two_int[1][shared_alpha, x, shared_alpha, spatial_b])
                    if shared_beta.size != 0:
                        coulomb += np.sum(self.two_int[2][x, shared_beta, spatial_b, shared_beta])
                        exchange -= np.sum(self.two_int[2][x, shared_beta, shared_beta, spatial_b])
            # selected (spin orbital) y = b
            elif y == spatial_b and spin_ind == spin_a == spin_b:
                # spin of x, y, a, b = alpha
                if spin_ind == 0:
                    one_electron += self.one_int[0][spatial_a, x]
                    if shared_beta.size != 0:
                        coulomb += np.sum(self.two_int[1][spatial_a, shared_beta, x, shared_beta])
                    if shared_alpha.size != 0:
                        coulomb += np.sum(self.two_int[0][spatial_a, shared_alpha, x, shared_alpha])
                        exchange -= np.sum(
                            self.two_int[0][spatial_a, shared_alpha, shared_alpha, x]
                        )
                # spin of x, y, a, b = beta
                else:
                    one_electron += self.one_int[1][spatial_a, x]
                    if shared_alpha.size != 0:
                        coulomb += np.sum(self.two_int[1][shared_alpha, spatial_a, shared_alpha, x])
                    if shared_beta.size != 0:
                        coulomb += np.sum(self.two_int[2][spatial_a, shared_beta, x, shared_beta])
                        exchange -= np.sum(self.two_int[2][spatial_a, shared_beta, shared_beta, x])
            # spin of x, y, a, b = alpha and selected (spin orbital) != a, b
            elif (
                spin_ind == 0
                and spin_a == spin_b == 0
                and y in shared_alpha
                and y not in [spatial_a, spatial_b]
            ):
                coulomb += self.two_int[0][x, spatial_a, y, spatial_b]
                coulomb += self.two_int[0][x, spatial_b, y, spatial_a]
                exchange -= self.two_int[0][x, spatial_a, spatial_b, y]
                exchange -= self.two_int[0][x, spatial_b, spatial_a, y]
            # spin of x, y = alpha and spin of a, b = beta and selected (spin orbital) != a, b
            elif spin_ind == 0 and spin_a == spin_b == 1 and y in shared_alpha:
                coulomb += self.two_int[1][x, spatial_a, y, spatial_b]
                coulomb += self.two_int[1][x, spatial_b, y, spatial_a]
            # spin of x, y = beta and spin of a, b = alpha and selected (spin orbital) != a, b
            elif spin_ind == 1 and spin_a == spin_b == 0 and y in shared_beta:
                coulomb += self.two_int[1][spatial_a, x, spatial_b, y]
                coulomb += self.two_int[1][spatial_b, x, spatial_a, y]
            # spin of x, y, a, b = beta and selected (spin orbital) != a, b
            elif (
                spin_ind == 1
                and spin_a == spin_b == 1
                and y in shared_beta
                and y not in [spatial_a, spatial_b]
            ):
                coulomb += self.two_int[2][x, spatial_a, y, spatial_b]
                coulomb += self.two_int[2][x, spatial_b, y, spatial_a]
                exchange -= self.two_int[2][x, spatial_a, spatial_b, y]
                exchange -= self.two_int[2][x, spatial_b, spatial_a, y]

        # two sd's are different by double excitation
        else:
            a, b = diff_sd1
            c, d = diff_sd2
            (spatial_a, spatial_b, spatial_c, spatial_d) = map(
                lambda i: slater.spatial_index(i, nspatial), [a, b, c, d]
            )
            spin_a, spin_b, spin_c, spin_d = map(
                lambda i: int(not slater.is_alpha(i, nspatial)), [a, b, c, d]
            )

            if x == spatial_a and spin_ind == spin_a:
                if spin_ind == spin_c == spin_b == spin_d == 0:
                    coulomb -= self.two_int[0][y, spatial_b, spatial_c, spatial_d]
                    exchange += self.two_int[0][y, spatial_b, spatial_d, spatial_c]
                elif spin_ind == spin_c == 0 and spin_b == spin_d == 1:
                    coulomb -= self.two_int[1][y, spatial_b, spatial_c, spatial_d]
                # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
                # returns orbitals in increasing order (which means second index cannot be alpha if
                # the first is beta)
                # NOTE: d will not be alpha if c is beta
                # elif spin_ind == spin_c == 1 and spin_b == spin_d == 0:
                #     coulomb -= self.two_int[1][spatial_b, y, spatial_d, spatial_c]
                # elif spin_ind == spin_d == 0 and spin_b == spin_c == 1:
                #     exchange += self.two_int[1][y, spatial_b, spatial_d, spatial_c]
                # NOTE: b will not be alpha if a is beta (spin of x = spin of a)
                # elif spin_ind == spin_d == 1 and spin_b == spin_c == 0:
                #     exchange += self.two_int[1][spatial_b, y, spatial_c, spatial_d]
                elif spin_ind == spin_c == spin_b == spin_d == 1:
                    coulomb -= self.two_int[2][y, spatial_b, spatial_c, spatial_d]
                    exchange += self.two_int[2][y, spatial_b, spatial_d, spatial_c]
            elif x == spatial_b and spin_ind == spin_b:
                if spin_ind == spin_c == spin_a == spin_d == 0:
                    exchange += self.two_int[0][y, spatial_a, spatial_c, spatial_d]
                    coulomb -= self.two_int[0][y, spatial_a, spatial_d, spatial_c]
                # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
                # returns orbitals in increasing order (which means second index cannot be alpha if
                # the first is beta)
                # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
                # elif spin_ind == spin_c == 0 and spin_a == spin_d == 1:
                #     exchange += self.two_int[1][y, spatial_a, spatial_c, spatial_d]
                # NOTE: d will not be alpha if c is beta
                # elif spin_ind == spin_c == 1 and spin_a == spin_d == 0:
                #     exchange += self.two_int[1][spatial_a, y, spatial_d, spatial_c]
                # elif spin_ind == spin_d == 0 and spin_a == spin_c == 1:
                #     coulomb -= self.two_int[1][y, spatial_a, spatial_d, spatial_c]
                elif spin_ind == spin_d == 1 and spin_a == spin_c == 0:
                    coulomb -= self.two_int[1][spatial_a, y, spatial_c, spatial_d]
                elif spin_ind == spin_c == spin_a == spin_d == 1:
                    exchange += self.two_int[2][y, spatial_a, spatial_c, spatial_d]
                    coulomb -= self.two_int[2][y, spatial_a, spatial_d, spatial_c]
            elif x == spatial_c and spin_ind == spin_c:
                if spin_ind == spin_a == spin_b == spin_d == 0:
                    coulomb -= self.two_int[0][spatial_a, spatial_b, y, spatial_d]
                    exchange += self.two_int[0][spatial_b, spatial_a, y, spatial_d]
                elif spin_ind == spin_a == 0 and spin_b == spin_d == 1:
                    coulomb -= self.two_int[1][spatial_a, spatial_b, y, spatial_d]
                # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
                # returns orbitals in increasing order (which means second index cannot be alpha if
                # the first is beta)
                # NOTE: d will not be alpha if c is beta (spin of x = spin of c)
                # elif spin_ind == spin_a == 1 and spin_b == spin_d == 0:
                #     coulomb -= self.two_int[1][spatial_b, spatial_a, spatial_d, y]
                # elif spin_ind == spin_b == 0 and spin_a == spin_d == 1:
                #     exchange += self.two_int[1][spatial_b, spatial_a, y, spatial_d]
                # NOTE: b will not be alpha if a is beta
                # elif spin_ind == spin_b == 1 and spin_a == spin_d == 0:
                #     exchange += self.two_int[1][spatial_a, spatial_b, spatial_d, y]
                elif spin_ind == spin_a == spin_b == spin_d == 1:
                    coulomb -= self.two_int[2][spatial_a, spatial_b, y, spatial_d]
                    exchange += self.two_int[2][spatial_b, spatial_a, y, spatial_d]
            elif x == spatial_d and spin_ind == spin_d:
                if spin_ind == spin_a == spin_b == spin_c == 0:
                    exchange += self.two_int[0][spatial_a, spatial_b, y, spatial_c]
                    coulomb -= self.two_int[0][spatial_b, spatial_a, y, spatial_c]
                # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
                # returns orbitals in increasing order (which means second index cannot be alpha if
                # the first is beta)
                # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
                # elif spin_ind == spin_a == 0 and spin_b == spin_c == 1:
                #     exchange += self.two_int[1][spatial_a, spatial_b, y, spatial_c]
                # NOTE: b will not be alpha if a is beta
                # elif spin_ind == spin_a == 1 and spin_b == spin_c == 0:
                #     exchange += self.two_int[1][spatial_b, spatial_a, spatial_c, y]
                # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
                # elif spin_ind == spin_b == 0 and spin_a == spin_c == 1:
                #     coulomb -= self.two_int[1][spatial_b, spatial_a, y, spatial_c]
                elif spin_ind == spin_b == 1 and spin_a == spin_c == 0:
                    coulomb -= self.two_int[1][spatial_a, spatial_b, spatial_c, y]
                elif spin_ind == spin_a == spin_b == spin_c == 1:
                    exchange += self.two_int[2][spatial_a, spatial_b, y, spatial_c]
                    coulomb -= self.two_int[2][spatial_b, spatial_a, y, spatial_c]

            if y == spatial_a and spin_ind == spin_a:
                if spin_ind == spin_c == spin_b == spin_d == 0:
                    coulomb += self.two_int[0][x, spatial_b, spatial_c, spatial_d]
                    exchange -= self.two_int[0][x, spatial_b, spatial_d, spatial_c]
                elif spin_ind == spin_c == 0 and spin_b == spin_d == 1:
                    coulomb += self.two_int[1][x, spatial_b, spatial_c, spatial_d]
                # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
                # returns orbitals in increasing order (which means second index cannot be alpha if
                # the first is beta)
                # NOTE: d will not be alpha if c is beta
                # elif spin_ind == spin_c == 1 and spin_b == spin_d == 0:
                #     coulomb += self.two_int[1][spatial_b, x, spatial_d, spatial_c]
                # elif spin_ind == spin_d == 0 and spin_b == spin_c == 1:
                #     exchange -= self.two_int[1][x, spatial_b, spatial_d, spatial_c]
                # NOTE: b will not be alpha if a is beta (spin of x = spin of a)
                # elif spin_ind == spin_d == 1 and spin_b == spin_c == 0:
                #     exchange -= self.two_int[1][spatial_b, x, spatial_c, spatial_d]
                elif spin_ind == spin_c == spin_b == spin_d == 1:
                    coulomb += self.two_int[2][x, spatial_b, spatial_c, spatial_d]
                    exchange -= self.two_int[2][x, spatial_b, spatial_d, spatial_c]
            elif y == spatial_b and spin_ind == spin_b:
                if spin_ind == spin_c == spin_a == spin_d == 0:
                    exchange -= self.two_int[0][x, spatial_a, spatial_c, spatial_d]
                    coulomb += self.two_int[0][x, spatial_a, spatial_d, spatial_c]
                # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
                # returns orbitals in increasing order (which means second index cannot be alpha if
                # the first is beta)
                # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
                # elif spin_ind == spin_c == 0 and spin_a == spin_d == 1:
                #     exchange -= self.two_int[1][x, spatial_a, spatial_c, spatial_d]
                # NOTE: d will not be alpha if c is beta
                # elif spin_ind == spin_c == 1 and spin_a == spin_d == 0:
                #     exchange -= self.two_int[1][spatial_a, x, spatial_d, spatial_c]
                # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
                # elif spin_ind == spin_d == 0 and spin_a == spin_c == 1:
                #     coulomb += self.two_int[1][x, spatial_a, spatial_d, spatial_c]
                elif spin_ind == spin_d == 1 and spin_a == spin_c == 0:
                    coulomb += self.two_int[1][spatial_a, x, spatial_c, spatial_d]
                elif spin_ind == spin_c == spin_a == spin_d == 1:
                    exchange -= self.two_int[2][x, spatial_a, spatial_c, spatial_d]
                    coulomb += self.two_int[2][x, spatial_a, spatial_d, spatial_c]
            elif y == spatial_c and spin_ind == spin_c:
                if spin_ind == spin_a == spin_b == spin_d == 0:
                    coulomb += self.two_int[0][spatial_a, spatial_b, x, spatial_d]
                    exchange -= self.two_int[0][spatial_b, spatial_a, x, spatial_d]
                elif spin_ind == spin_a == 0 and spin_b == spin_d == 1:
                    coulomb += self.two_int[1][spatial_a, spatial_b, x, spatial_d]
                # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
                # returns orbitals in increasing order (which means second index cannot be alpha if
                # the first is beta)
                # NOTE: d will not be alpha if c is beta (spin of x = spin of c)
                # elif spin_ind == spin_a == 1 and spin_b == spin_d == 0:
                #     coulomb += self.two_int[1][spatial_b, spatial_a, spatial_d, x]
                # NOTE: a will not be alpha if b is beta
                # elif spin_ind == spin_b == 0 and spin_a == spin_d == 1:
                #     exchange -= self.two_int[1][spatial_b, spatial_a, x, spatial_d]
                # NOTE: d will not be alpha if c is beta (spin of x = spin of c)
                # elif spin_ind == spin_b == 1 and spin_a == spin_d == 0:
                #     exchange -= self.two_int[1][spatial_a, spatial_b, spatial_d, x]
                elif spin_ind == spin_a == spin_b == spin_d == 1:
                    coulomb += self.two_int[2][spatial_a, spatial_b, x, spatial_d]
                    exchange -= self.two_int[2][spatial_b, spatial_a, x, spatial_d]
            elif y == spatial_d and spin_ind == spin_d:
                if spin_ind == spin_a == spin_b == spin_c == 0:
                    exchange -= self.two_int[0][spatial_a, spatial_b, x, spatial_c]
                    coulomb += self.two_int[0][spatial_b, spatial_a, x, spatial_c]
                # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
                # returns orbitals in increasing order (which means second index cannot be alpha if
                # the first is beta)
                # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
                # elif spin_ind == spin_a == 0 and spin_b == spin_c == 1:
                #     exchange -= self.two_int[1][spatial_a, spatial_b, x, spatial_c]
                # NOTE: b will not be alpha if a is beta
                # elif spin_ind == spin_a == 1 and spin_b == spin_c == 0:
                #     exchange -= self.two_int[1][spatial_b, spatial_a, spatial_c, x]
                # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
                # elif spin_ind == spin_b == 0 and spin_a == spin_c == 1:
                #     coulomb += self.two_int[1][spatial_b, spatial_a, x, spatial_c]
                elif spin_ind == spin_b == 1 and spin_a == spin_c == 0:
                    coulomb += self.two_int[1][spatial_a, spatial_b, spatial_c, x]
                elif spin_ind == spin_a == spin_b == spin_c == 1:
                    exchange -= self.two_int[2][spatial_a, spatial_b, x, spatial_c]
                    coulomb += self.two_int[2][spatial_b, spatial_a, x, spatial_c]

        return sign * one_electron, sign * coulomb, sign * exchange
