r"""Hamiltonian used to describe a chemical system expressed wrt restricted orbitals."""
import numpy as np
from wfns.backend import slater
from wfns.ham.restricted_base import BaseRestrictedHamiltonian
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
    _update_integrals(self, wfn, sd, sd_m, wfn_deriv, ham_deriv, one_electron, coulomb, exchange)
        Update integrals for the given Slater determinant.
        Used to simplify `integrate_wfn_sd`.
    integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """
    # inherit from BaseRestrictedHamiltonian
    nspin = BaseRestrictedHamiltonian.nspin

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

        nspatial = self.one_int.shape[0]

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
            one_electron = np.sum(self.one_int[shared_alpha, shared_alpha])
            one_electron += np.sum(self.one_int[shared_beta, shared_beta])
            coulomb = np.sum(np.triu(self.two_int[shared_alpha, :, shared_alpha, :]
                                                 [:, shared_alpha, shared_alpha], k=1))
            coulomb += np.sum(np.triu(self.two_int[shared_beta, :, shared_beta, :]
                                                  [:, shared_beta, shared_beta], k=1))
            coulomb += np.sum(self.two_int[shared_alpha, :, shared_alpha, :]
                                          [:, shared_beta, shared_beta])
            exchange = -np.sum(np.triu(self.two_int[shared_alpha, :, :, shared_alpha]
                                                   [:, shared_alpha, shared_alpha], k=1))
            exchange += -np.sum(np.triu(self.two_int[shared_beta, :, :, shared_beta]
                                                    [:, shared_beta, shared_beta], k=1))

        # two sd's are different by single excitation
        elif diff_order == 1:
            a, = diff_sd1
            b, = diff_sd2
            # get spatial indices
            spatial_a = slater.spatial_index(a, nspatial)
            spatial_b = slater.spatial_index(b, nspatial)

            if slater.is_alpha(a, nspatial) != slater.is_alpha(b, nspatial):
                return 0.0, 0.0, 0.0

            one_electron = self.one_int[spatial_a, spatial_b]
            coulomb = np.sum(self.two_int[shared_alpha, spatial_a, shared_alpha, spatial_b])
            coulomb += np.sum(self.two_int[shared_beta, spatial_a, shared_beta, spatial_b])
            if slater.is_alpha(a, nspatial):
                exchange = -np.sum(self.two_int[shared_alpha, spatial_a, spatial_b, shared_alpha])
            else:
                exchange = -np.sum(self.two_int[shared_beta, spatial_a, spatial_b, shared_beta])

        # two sd's are different by double excitation
        else:
            a, b = diff_sd1
            c, d = diff_sd2
            # get spatial indices
            spatial_a = slater.spatial_index(a, nspatial)
            spatial_b = slater.spatial_index(b, nspatial)
            spatial_c = slater.spatial_index(c, nspatial)
            spatial_d = slater.spatial_index(d, nspatial)

            if (slater.is_alpha(b, nspatial) == slater.is_alpha(d, nspatial) and
                    slater.is_alpha(a, nspatial) == slater.is_alpha(c, nspatial)):
                coulomb = self.two_int[spatial_a, spatial_b, spatial_c, spatial_d]
            if (slater.is_alpha(b, nspatial) == slater.is_alpha(c, nspatial) and
                    slater.is_alpha(a, nspatial) == slater.is_alpha(d, nspatial)):
                exchange = -self.two_int[spatial_a, spatial_b, spatial_d, spatial_c]

        return sign*one_electron, sign*coulomb, sign*exchange
