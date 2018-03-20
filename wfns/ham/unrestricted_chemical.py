r"""Hamiltonian used to describe a chemical system expressed wrt unrestricted orbitals."""
import numpy as np
from wfns.backend import slater, math_tools
from wfns.ham.unrestricted_base import BaseUnrestrictedHamiltonian
from wfns.ham.generalized_chemical import GeneralizedChemicalHamiltonian


class UnrestrictedChemicalHamiltonian(BaseUnrestrictedHamiltonian, GeneralizedChemicalHamiltonian):
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
    _update_integrals(self, wfn, sd, sd_m, wfn_deriv, ham_deriv, one_electron, coulomb, exchange)
        Update integrals for the given Slater determinant.
        Used to simplify `integrate_wfn_sd`.
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
        BaseUnrestrictedHamiltonian.__init__(self, one_int, two_int, energy_nuc_nuc=energy_nuc_nuc)
        self.set_ref_ints()
        self.assign_params(params=params)

    def set_ref_ints(self):
        """Store the current integrals as the reference from which orbitals will be rotated."""
        self._ref_one_int = [np.copy(self.one_int[0]), np.copy(self.one_int[1])]
        self._ref_two_int = [np.copy(self.two_int[0]),
                             np.copy(self.two_int[1]), np.copy(self.two_int[2])]

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

        if not(isinstance(params, np.ndarray) and params.ndim == 1 and params.size == num_params):
            raise ValueError('Parameters for orbital rotation must be a one-dimension numpy array '
                             'with {0}=K*(K-1) elements, where K is the number of '
                             'orbitals.'.format(num_params))

        # assign parameters
        self.params = params

        # revert integrals back to original
        self.assign_integrals([np.copy(self._ref_one_int[0]), np.copy(self._ref_one_int[1])],
                              [np.copy(self._ref_two_int[0]),
                               np.copy(self._ref_two_int[1]), np.copy(self._ref_two_int[2])])

        # convert antihermitian part to unitary matrix.
        unitary_alpha = math_tools.unitary_matrix(params[:num_params//2])
        unitary_beta = math_tools.unitary_matrix(params[num_params//2:])

        # transform integrals
        self.orb_rotate_matrix([unitary_alpha, unitary_beta])

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
        NotImplementedError
            If `deriv` is not `None`.

        """
        if deriv is not None:
            raise NotImplementedError('Orbital rotation is not implemented properly: you cannot '
                                      'take the derivative of CI matrix elements with respect to '
                                      'orbital rotation coefficients.')

        nspatial = self.one_int[0].shape[0]

        sd1 = slater.internal_sd(sd1)
        sd2 = slater.internal_sd(sd2)
        shared_alpha_sd, shared_beta_sd = slater.split_spin(slater.shared_sd(sd1, sd2), nspatial)
        shared_alpha = slater.occ_indices(shared_alpha_sd)
        shared_beta = slater.occ_indices(shared_beta_sd)
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
            raise ValueError('The sign associated with the integral must be either `1` or `-1`.')

        one_electron, coulomb, exchange = 0.0, 0.0, 0.0

        # two sd's are the same
        if diff_order == 0:
            one_electron = sign * np.sum(self.one_int[0][shared_alpha, shared_alpha])
            one_electron += sign * np.sum(self.one_int[1][shared_beta, shared_beta])
            coulomb = np.sum(np.triu(self.two_int[0][shared_alpha, :, shared_alpha, :]
                                                    [:, shared_alpha, shared_alpha], k=1))
            coulomb += np.sum(np.triu(self.two_int[2][shared_beta, :, shared_beta, :]
                                                     [:, shared_beta, shared_beta], k=1))
            coulomb += np.sum(self.two_int[1][shared_alpha, :, shared_alpha, :]
                                             [:, shared_beta, shared_beta])
            exchange = -np.sum(np.triu(self.two_int[0][shared_alpha, :, :, shared_alpha]
                                                      [:, shared_alpha, shared_alpha], k=1))
            exchange += -np.sum(np.triu(self.two_int[1][shared_beta, :, :, shared_beta]
                                                       [:, shared_beta, shared_beta], k=1))

        # two sd's are different by single excitation
        elif diff_order == 1:
            a, = diff_sd1
            b, = diff_sd2
            spatial_a = slater.spatial_index(a, nspatial)
            spatial_b = slater.spatial_index(b, nspatial)

            if slater.is_alpha(a, nspatial) != slater.is_alpha(b, nspatial):
                return 0.0, 0.0, 0.0

            if slater.is_alpha(a, nspatial):
                one_electron = self.one_int[0][spatial_a, spatial_b]
                coulomb = np.sum(self.two_int[0][shared_alpha, spatial_a, shared_alpha, spatial_b])
                coulomb += np.sum(self.two_int[1][spatial_a, shared_beta, spatial_b, shared_beta])
                exchange = -np.sum(self.two_int[0]
                                               [shared_alpha, spatial_a, spatial_b, shared_alpha])
            else:
                one_electron = self.one_int[1][spatial_a, spatial_b]
                coulomb = np.sum(self.two_int[1][shared_alpha, spatial_a, shared_alpha, spatial_b])
                coulomb += np.sum(self.two_int[2][shared_beta, spatial_a, shared_beta, spatial_b])
                exchange = -np.sum(self.two_int[2][shared_beta, spatial_a, spatial_b, shared_beta])

        # two sd's are different by double excitation
        else:
            a, b = diff_sd1
            c, d = diff_sd2

            if slater.is_alpha(a, nspatial) and slater.is_alpha(b, nspatial):
                spin_index = 0
            elif slater.is_alpha(a, nspatial) and not slater.is_alpha(b, nspatial):
                spin_index = 1
            elif not slater.is_alpha(a, nspatial) and slater.is_alpha(b, nspatial):
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

            if (slater.is_alpha(b, nspatial) == slater.is_alpha(d, nspatial) and
                    slater.is_alpha(a, nspatial) == slater.is_alpha(c, nspatial)):
                coulomb = self.two_int[spin_index][spatial_a, spatial_b, spatial_c, spatial_d]
            if (slater.is_alpha(b, nspatial) == slater.is_alpha(c, nspatial) and
                    slater.is_alpha(a, nspatial) == slater.is_alpha(d, nspatial)):
                exchange = -self.two_int[spin_index][spatial_a, spatial_b, spatial_d, spatial_c]

        return sign*one_electron, sign*coulomb, sign*exchange
