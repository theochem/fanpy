r"""Hamiltonian used to describe a chemical system."""
from __future__ import absolute_import, division, print_function, unicode_literals
from wfns.backend import slater
from wfns.hamiltonian.base_hamiltonian import BaseHamiltonian
from wfns.wrapper.docstring import docstring_class


@docstring_class(indent_level=1)
class ChemicalHamiltonian(BaseHamiltonian):
    r"""Hamiltonian used to describe a typical chemical system.

    .. math::

        \hat{H} = \sum_{ij} h_{ij} a^\dagger_i a_j
        + \sum_{i<j,k<l} g_{ijkl} a^\dagger_i a^\dagger_j a_l a_k

    where :math:`h_{ik}` is the one-electron integral and :math:`g_{ijkl}` is the two-electron
    integral in Physicists' notation.

    """
    def _update_integrals(self, wfn, sd, sd_m, wfn_deriv, ham_deriv, one_electron, coulomb,
                          exchange):
        """Update integrals for the given Slater determinant.

        Add the term :math:`f(\mathbf{m}) \braket{\Phi | \hat{H} | \mathbf{m}}` to the provided
        integrals.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction.
        sd : int
            Slater determinant.
        sd_m : int
            Slater determinant.
        wfn_deriv : {int, None}
            Index of the wavefunction parameter against which the integral is derivatized.
            `None` results in no derivatization.
        ham_deriv : {int, None}
            Index of the Hamiltonian parameter against which the integral is derivatized.
            `None` results in no derivatization.
        one_electron : float
            One-electron energy.
        coulomb : float
            Coulomb energy.
        exchange : float
            Exchange energy.

        Returns
        -------
        one_electron : float
            Updated one-electron energy.
        coulomb : float
            Updated coulomb energy.
        exchange : float
            Updated exchange energy.

        """
        coeff = wfn.get_overlap(sd_m, deriv=wfn_deriv)
        sd_energy = self.integrate_sd_sd(sd, sd_m, deriv=ham_deriv)
        one_electron += coeff * sd_energy[0]
        coulomb += coeff * sd_energy[1]
        exchange += coeff * sd_energy[2]
        return one_electron, coulomb, exchange

    # FIXME: need to speed up
    def integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None):
        r"""Integrate the Hamiltonian with against a wavefunction and Slater determinant.

        .. math::

            \braket{\Phi | \hat{H} | \Psi}
            &= \sum_{\mathbf{m} \in S_\Phi} f(\mathbf{m}) \braket{\Phi | \hat{H} | \mathbf{m}}

        where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
        :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
        determinants for which :math:`\braket{\Phi | \hat{H} | \mathbf{m}}` is not zero, which are
        the :math:`\Phi` and its first and second order excitations for a chemical Hamiltonian.

        Parameters
        ----------
        wfn : Wavefunction
            Wavefunction against which the Hamiltonian is integrated.
            Needs to have the following in `__dict__`: `get_overlap`.
        sd : int
            Slater Determinant against which the Hamiltonian is integrated.
        wfn_deriv : {int, None}
            Index of the wavefunction parameter against which the integral is derivatized.
            Default is no derivatization.
        ham_deriv : {int, None}
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
            If integral is derivatized to both wavefunction and Hamiltonian parameters.

        """
        if wfn_deriv is not None and ham_deriv is not None:
            raise ValueError('Integral can be derivatized with respect to at most one out of the '
                             'wavefunction and Hamiltonian parameters.')

        sd = slater.internal_sd(sd)
        occ_indices = slater.occ_indices(sd)
        vir_indices = slater.vir_indices(sd, self.nspin)

        one_electron = 0.0
        coulomb = 0.0
        exchange = 0.0

        def update_integrals(sd_m):
            return self._update_integrals(wfn, sd, sd_m, wfn_deriv, ham_deriv,
                                          one_electron, coulomb, exchange)

        one_electron, coulomb, exchange = update_integrals(sd)
        for counter_i, i in enumerate(occ_indices):
            for counter_a, a in enumerate(vir_indices):
                sd_m = slater.excite(sd, i, a)
                one_electron, coulomb, exchange = update_integrals(sd_m)
                for j in occ_indices[counter_i+1:]:
                    for b in vir_indices[counter_a+1:]:
                        sd_m = slater.excite(sd, i, j, b, a)
                        one_electron, coulomb, exchange = update_integrals(sd_m)

        return one_electron, coulomb, exchange

    # FIXME: need to speed up
    def integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None):
        r"""Integrate the Hamiltonian with against two Slater determinants.

        .. math::

            H_{\mathbf{m}\mathbf{n}} &= \braket{\mathbf{m} | \hat{H} | \mathbf{n}}\\
            &= \sum_{ij} h_{ij} \braket{\mathbf{m}| a^\dagger_i a_j | \mathbf{n}}
            + \sum_{i<j, k<l} g_{ijkl}
            \braket{\mathbf{m}| a^\dagger_i a^\dagger_j a_l a_k | \mathbf{n}}\\

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
            raise NotImplementedError('Current Hamiltonian does not have any parameters.')

        one_electron = 0.0
        coulomb = 0.0
        exchange = 0.0

        sd1 = slater.internal_sd(sd1)
        sd2 = slater.internal_sd(sd2)
        shared_indices = slater.shared_orbs(sd1, sd2)
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1, sd2)
        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            return 0.0, 0.0, 0.0
        diff_order = len(diff_sd1)

        # Assume the creators are ordered from smallest to largest (left to right) and
        # annihilators are ordered from largest to smallest.
        # Move all of the creators and annihilators that are shared between the two Slater
        # determinants towards the middle starting from the smallest index.
        # e.g. Given left Slater determinant a_8 a_6 a_3 a_2 a_1 and
        #      right Slater determinant a^\dagger_1 a^\dagger_2 a^dagger_5 a^\dagger_7 a^\dagger_8,
        #      annihilators and creators of orbitals 1, 2, and 8 must move towards one another for
        #      mutual destruction
        #    0    a_8 a_6 a_3 a_2 a_1    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    1    a_8 a_6 a_3 a_1 a_2    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    2    a_6 a_8 a_3 a_1 a_2    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    3    a_6 a_3 a_8 a_1 a_2    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    4    a_6 a_3 a_1 a_8 a_2    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    5    a_6 a_3 a_1 a_2 a_8    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    6    a_6 a_3 a_1 a_2 a_8    a^\dagger_2 a^\dagger_1 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    7    a_6 a_3 a_1 a_2 a_8    a^\dagger_2 a^\dagger_1 a^\dagger_5 a^\dagger_8 a^\dagger_7
        #    8    a_6 a_3 a_1 a_2 a_8    a^\dagger_2 a^\dagger_1 a^\dagger_8 a^\dagger_5 a^\dagger_7
        #    9    a_6 a_3 a_1 a_2 a_8    a^\dagger_2 a^\dagger_8 a^\dagger_1 a^\dagger_5 a^\dagger_7
        #   10    a_6 a_3 a_1 a_2 a_8    a^\dagger_8 a^\dagger_2 a^\dagger_1 a^\dagger_5 a^\dagger_7
        #   10    a_6 a_3                                                    a^\dagger_5 a^\dagger_7
        # where first column is the number of transpositions, second column is the ordering of the
        # annhilators (left SD) and third column is the ordering of the creators (right SD)

        # NOTE: after moving all of the different creators toward the middle, the unshared creators
        # will be ordered from smallest to largest and the shared creators will be ordered from
        # largest to smallest
        if sign is None:
            sign = slater.sign_excite(sd1, diff_sd1, diff_sd2)
        elif sign not in [1, -1]:
            raise ValueError('The sign associated with the integral must be either `1` or `-1`.')

        # two sd's are the same
        if diff_order == 0:
            for count, i in enumerate(shared_indices):
                one_electron += sign * self.one_int.get_value(i, i, self.orbtype)
                for j in shared_indices[count + 1:]:
                    coulomb += sign * self.two_int.get_value(i, j, i, j, self.orbtype)
                    exchange -= sign * self.two_int.get_value(i, j, j, i, self.orbtype)

        # two sd's are different by single excitation
        elif diff_order == 1:
            i, = diff_sd1
            a, = diff_sd2
            one_electron += sign * self.one_int.get_value(i, a, self.orbtype)
            for j in shared_indices:
                coulomb += sign * self.two_int.get_value(i, j, a, j, self.orbtype)
                exchange -= sign * self.two_int.get_value(i, j, j, a, self.orbtype)

        # two sd's are different by double excitation
        elif diff_order == 2:
            i, j = diff_sd1
            a, b = diff_sd2
            coulomb += sign * self.two_int.get_value(i, j, a, b, self.orbtype)
            exchange -= sign * self.two_int.get_value(i, j, b, a, self.orbtype)

        return one_electron, coulomb, exchange
