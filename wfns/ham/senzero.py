r"""Seniority-zero Hamiltonian object that interacts with the wavefunction."""
from wfns.ham.chemical import ChemicalHamiltonian
from wfns.backend import slater


class SeniorityZeroHamiltonian(ChemicalHamiltonian):
    r"""Hamiltonian that involves only the zero-seniority terms.

    .. math::

        \hat{H} =
        \sum_{i} \left(
            h_{ii} a^\dagger_i a_i + h_{\bar{\i}\bar{\i}} a^\dagger_{\bar{\i}} a_{\bar{\i}}
        \right)
        + \sum_{i<j} \left(
            g_{ijij} a^\dagger_i a^\dagger_j a_j a_i
            + g_{i\bar{\j}i\bar{\j}} a^\dagger_i a^\dagger_{\bar{\j}} a_{\bar{\j}} a_i
            + g_{\bar{\i}j\bar{\i}j} a^\dagger_{\bar{\i}} a^\dagger_j  a_j a_{\bar{\i}}
            + g_{\bar{\i}\bar{\j}\bar{\i}\bar{\j}}
            a^\dagger_{\bar{\i}} a^\dagger_{\bar{\j}} a_{\bar{\j}} a_{\bar{\i}}
            + g_{i\bar{\j}\bar{\j}i} a^\dagger_i a^\dagger_{\bar{\j}} a_{\bar{\j} a_i}
        \right)

    where :math:`i` and :math:`\bar{\i}` are the indices that correspond to the ith alpha- and beta-
    spin orbtials, :math:`h_{ik}` is the one-electron integral and :math:`g_{ijkl}` is the two-
    electron integral.

    Seniority zero means that there are no unpaired electrons.

    Attributes
    ----------
    params : np.ndarray
        Parameters of the Hamitonian.
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used.
    energy_nuc_nuc : float
        Nuclear-nuclear repulsion energy.
    one_int : {1- or 2-tuple np.ndarray(K, K)}
        One electron integrals for restricted, unrestricted, or generalized orbitals.
        1-tuple for spatial (restricted) and generalized orbitals.
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components).
    two_int : {1- or 3-tuple np.ndarray(K, K)}
        Two electron integrals for restricted, unrestricted, or generalized orbitals.
        Uses the physicist's notation.
        1-tuple for spatial (restricted) and generalized orbitals.
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components).

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspin : int
        Number of spin orbitals.
    dtype : {'restricted', 'unrestricted', 'generalized'}
        Number of spin orbitals.

    Methods
    -------
    __init__(self, one_int, two_int, orbtype=None, energy_nuc_nuc=None)
        Initialize the Hamiltonian
    clear_cache(self)
        Placeholder function that would clear the cache.
    assign_orbtype(self, orbtype=None)
        Assign the orbital type.
    assign_energy_nuc_nuc(self, energy_nuc_nuc=None)
        Assigns the nuclear nuclear repulsion.
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.
    assign_params(self, params)
        Assign parameters of the Hamiltonian.
    orb_rotate_jacobi(self, jacobi_indices, theta)
        Rotate orbitals using Jacobi matrix.
    orb_rotate_matrix(self, matrix)
        Rotate orbitals using a transformation matrix.
    _update_integrals(self, wfn, sd, sd_m, wfn_deriv, ham_deriv, one_electron, coulomb, exchange)
        Update integrals for the given Slater determinant.
        Used to simplify `integrate_wfn_sd`.
    integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """
    def assign_orbtype(self, orbtype=None):
        """Assign the orbital type.

        Raises
        ------
        ValueError
            If orbtype is not one of 'restricted' or 'unrestricted'.
        NotImplementedError
            If orbtype is 'generalized'

        """
        super().assign_orbtype(orbtype)
        if orbtype == 'generalized':
            raise NotImplementedError('Generalized orbitals are not supported in seniority-zero '
                                      'Hamiltonian.')

    # FIXME: need to speed up
    def integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None):
        r"""Integrate the Hamiltonian with against a wavefunction and Slater determinant.

        .. math::

            \braket{\Phi | \hat{H} | \Psi}
            &= \sum_{\mathbf{m} \in S_\Phi} f(\mathbf{m}) \braket{\Phi | \hat{H} | \mathbf{m}}

        where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
        :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
        determinants for which :math:`\braket{\Phi | \hat{H} | \mathbf{m}}` is not zero, which are
        the :math:`\Phi` and its first order orbital-pair excitations for a chemical Hamiltonian.

        """
        if wfn_deriv is not None and ham_deriv is not None:
            raise ValueError('Integral can be derivatized with respect to at most one out of the '
                             'wavefunction and Hamiltonian parameters.')

        nspatial = self.nspin // 2
        sd = slater.internal_sd(sd)
        sd_alpha, sd_beta = slater.split_spin(sd, nspatial)
        occ_spatial_indices = slater.occ_indices(sd_alpha)
        vir_spatial_indices = slater.vir_indices(sd_alpha, nspatial)

        one_electron = 0.0
        coulomb = 0.0
        exchange = 0.0

        def update_integrals(sd_m):
            """Wrapped function for updating the integral values."""
            return self._update_integrals(wfn, sd, sd_m, wfn_deriv, ham_deriv,
                                          one_electron, coulomb, exchange)

        one_electron, coulomb, exchange = update_integrals(sd)

        # if slater determinant is not seniority zero
        if sd_alpha != sd_beta:
            return one_electron, coulomb, exchange

        for i in occ_spatial_indices:
            for a in vir_spatial_indices:
                # FIXME: assumes that the orbitals are organized blockwise (alpha then beta)
                sd_m = slater.excite(sd, i, i + nspatial, a + nspatial, a)
                one_electron, coulomb, exchange = update_integrals(sd_m)

        return one_electron, coulomb, exchange

    def integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None):
        r"""Integrate the Hamiltonian with against two Slater determinants.

        .. math::

            H_{\mathbf{m}\mathbf{n}} &= \braket{\mathbf{m} | \hat{H} | \mathbf{n}}\\

        In the first summation involving :math:`h_{ij}`, only the terms where :math:`\mathbf{m}` and
        :math:`\mathbf{n}` are the same will contribute to the integral. In the second summation
        involving :math:`g_{ijkl}`, only the terms where :math:`\mathbf{m}` and :math:`\mathbf{n}`
        are different by at most single pair-wise excitation will contribute to the integral.

        Raises
        ------
        ValueError
            If `sign` is not `1`, `-1` or `None`.

        """
        nspatial = self.nspin // 2
        sd1 = slater.internal_sd(sd1)
        sd2 = slater.internal_sd(sd2)
        sd1_alpha, sd1_beta = slater.split_spin(sd1, nspatial)
        sd2_alpha, sd2_beta = slater.split_spin(sd2, nspatial)

        # if either of the two Slater determinants are not seniority zero and they're different
        if (sd1_alpha != sd1_beta or sd2_alpha != sd2_beta) and sd1 != sd2:
            return 0.0, 0.0, 0.0

        # FIXME: up to 4 times slower for spatial orbitals (b/c looping over 2K instead of K)
        # two Slater determinants are the same
        return super().integrate_sd_sd(sd1, sd2, sign=sign, deriv=deriv)
