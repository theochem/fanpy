"""Seniority-zero Hamiltonian object that interacts with the wavefunction.

#FIXME: this needs to be checked
..math::
    \hat{H} = \sum_{i} \big( h_{ii} a^\dagger_i a_i
                             + h_{\bar{\i}\bar{\i}} a^\dagger_{\bar{\i}} a_{\bar{\i}} \big)
    + \sum_{ij} \big( g_{ijij} a^\dagger_i a^\dagger_j a_i a_j
                      + g_{i\bar{\j}i\bar{\j}} a^\dagger_i a^\dagger_{\bar{\j}} a_i a_{\bar{\j}}
                      + g_{\bar{\i}j\bar{\i}j} a^\dagger_{\bar{\i}} a^\dagger_j a_{\bar{\i}} a_j
                      + g_{\bar{\i}\bar{\i}\bar{\i}\bar{\j}}
                      a^\dagger_{\bar{\i}} a^\dagger_{\bar{\j}} a_{\bar{\i}} a_{\bar{\j}}
                      + g_{j\bar{\j}i\bar{\i}} a^\dagger_j a^\dagger_{\bar{\j}} a_i a_{\bar{\i}}
                \big)
where :math:`i` and :math:`\bar{\i}` are the indices that correspond to the ith alpha and beta spin
orbtials, :math:`h_{ik}` is the one-electron integral and :math:`g_{ijkl}` is the two-electron
integral.

Class
-----
SeniorityZeroHamiltonian(one_int, two_int, orbtype=None, energy_nuc_nuc=None)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from .chemical_hamiltonian import ChemicalHamiltonian
from ..backend import slater


class SeniorityZeroHamiltonian(ChemicalHamiltonian):
    """

    Seniority zero means that there are no unpaired electrons
    """
    def assign_orbtype(self, orbtype=None):
        """Assign the orbital type.

        Parameters
        ----------
        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbitals used.
            Default is `'restricted'`

        Raises
        ------
        ValueError
            If orbtype is not one of ['restricted', 'unrestricted']
        NotImplementedError
            If orbtype is 'generalized'

        Note
        ----
        Should be executed before assign_integrals.
        """
        super().assign_orbtype(orbtype)
        if orbtype == 'generalized':
            raise NotImplementedError('Generalized orbitals are not supported in seniority-zero '
                                      'Hamiltonian.')

    def integrate_wfn_sd(self, wfn, sd, deriv=None):
        """Integrates the seniority-zero Hamiltonian with against a wavefunction and Slater
        determinant.

        ..math::
            \big< \Psi \big| \hat{H} \big| \Phi \big>

        where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the seniority-zero Hamiltonian
        operator, and :math:`\Phi` is the Slater determinant.

        Parameters
        ----------
        wfn : Wavefunction
            Wavefunction against which the Hamiltonian is integrated.
            Needs to have the following in `__dict__`: `nspin`, `one_int`, `two_int`, `overlap`.
        sd : int
            Slater Determinant against which the Hamiltonian is integrated.
        deriv : int, None
            Index of the parameter against which the expectation value is derivatized.
            Default is no derivatization

        Returns
        -------
        one_electron : float
            One electron energy
        coulomb : float
            Coulomb energy
        exchange : float
            Exchange energy
        """
        # FIXME: incredibly slow/bad approach
        spatial_sd = slater.split_spin(sd, wfn.nspatial)[0]
        occ_spatial_indices = slater.occ_indices(spatial_sd)
        vir_spatial_indices = slater.vir_indices(spatial_sd, wfn.nspatial)

        one_electron = 0.0
        coulomb = 0.0
        exchange = 0.0

        # sum over zeroth order excitation
        coeff = wfn.get_overlap(sd, deriv=deriv)
        for counter, i in enumerate(occ_spatial_indices):
            one_electron += 2 * coeff * self.one_int.get_val(i, i, self.orbtype)
            coulomb += coeff * self.two_int.get_val(i, i, i, i, self.orbtype)
            for j in occ_spatial_indices[counter+1:]:
                coulomb += 4 * coeff * self.two_int.get_val(i, j, i, j, self.orbtype)
                exchange -= 2 * coeff * self.two_int.get_val(i, j, j, i, self.orbtype)

        # sum over pair wise excitation (seniority zero)
        for i in occ_spatial_indices:
            for a in vir_spatial_indices:
                j = i + wfn.nspatial
                b = a + wfn.nspatial
                coeff = wfn.get_overlap(slater.excite(sd, i, j, a, b), deriv=deriv)
                coulomb += coeff * self.two_int.get_val(i, j, a, b, self.orbtype)
                exchange -= coeff * self.two_int.get_val(i, j, b, a, self.orbtype)

        return one_electron, coulomb, exchange
