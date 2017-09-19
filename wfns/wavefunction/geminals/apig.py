"""Antisymmeterized Product of Interacting Geminals (APIG) wavefunction."""
from __future__ import absolute_import, division, print_function
import numpy as np
from wfns.wavefunction.geminals.base_geminal import BaseGeminal
from wfns.wrapper.docstring import docstring_class

__all__ = []


@docstring_class(indent_level=1)
class APIG(BaseGeminal):
    r"""Antisymmetrized Product of Interacting Geminals (APIG) Wavefunction.

    Each geminal is a linear combination of orbital pairs that do not share any orbitals with one
    another. Since there are :math:`2K` spin orbitals, there can only be :math:`K` orbital pairs.

    .. math::

        G^\dagger_p = \sum_{i=1}^{K} C_{pi} a^\dagger_i a^\dagger_{\bar{i}}

    where :math:`\bar{i}` corresponds to the spin orbital index that uniquely corresponds to the
    index :math:`i`.

    Then, there will be, at most, one way to represent a Slater determinant in terms of orbital
    pairs. The combinatorial sum present in more complex geminal flavors, such as APG and APsetG,
    reduces down to a single permanent.

    .. math::
        \ket{\Psi_{\mathrm{APIG}}}
        &= \prod_{p=1}^P G^\dagger_p \ket{\theta}\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}}
        |C(\mathbf{m})|^+ \ket{\mathbf{m}}

    By default, the spin orbitals that belong to the same spatial orbital, i.e. the alpha and beta
    spin orbitals, will be paired up. Then, the resulting wavefunction will only contain seniority
    zero (i.e. no unpaired electrons) Slater determinants.

    """
    @property
    def spin(self):
        """

        Notes
        -----
        Spin is not zero if you change the pairing scheme.

        """
        return 0.0

    @property
    def seniority(self):
        """

        Notes
        -----
        Seniority is not zero if you change the pairing scheme.

        """
        return 0

    def assign_orbpairs(self, orbpairs=None):
        """

        Raises
        ------
        TypeError
            If `orbpairs` is not an iterable.
            If an orbital pair is not given as a list or a tuple.
            If an orbital pair does not contain exactly two elements.
            If an orbital index is not an integer.
        ValueError
            If an orbital pair has the same integer.
            If an orbital pair occurs more than once.
            If any two orbital pair shares an orbital.
            If any orbital is not included in any orbital pair.

        """
        if orbpairs is None:
            orbpairs = ((i, i+self.nspatial) for i in range(self.nspatial))
        super().assign_orbpairs(orbpairs)

        all_orbs = [j for i in self.dict_orbpair_ind.keys() for j in i]
        if len(all_orbs) != len(set(all_orbs)):
            raise ValueError('At least two orbital pairs share an orbital')
        elif len(all_orbs) != self.nspin:
            raise ValueError('Not all of the orbitals are included in orbital pairs')

    def generate_possible_orbpairs(self, occ_indices):
        """

        The APIG wavefunction only contains one pairing scheme for each Slater determinant (because
        no two orbital pairs share the orbital). By default, the alpha and beta spin orbitals that
        correspond to the same spatial orbital will be paired up.

        Raises
        ------
        ValueError
            If the number of electrons in the Slater determinant does not match up with the number
            of electrons in the wavefunction.
            If the Slater determinant cannot be constructed using the APIG pairing scheme.

        """
        if len(occ_indices) != self.nelec:
            raise ValueError('The number of electrons in the Slater determinant does not match up '
                             'with the number of electrons in the wavefunction.')
        # ASSUME each orbital is associated with exactly one orbital pair
        dict_orb_ind = {orbpair[0]: ind for orbpair, ind in self.dict_orbpair_ind.items()}
        orbpairs = []
        for i in occ_indices:
            try:
                ind = dict_orb_ind[i]
            except KeyError:
                continue
            else:
                orbpairs.append(self.dict_ind_orbpair[ind])

        # signature to turn orbpairs into strictly INCREASING order.
        sign = (-1)**((self.npair//2) % 2)

        # if Slater determinant cannot be constructed from the orbital pairing scheme
        if len(occ_indices) != 2 * len(orbpairs):
            yield [], 1
        else:
            yield tuple(orbpairs), sign

    # TODO: refactor when APG is set up
    def to_apg(self):
        """Convert APIG wavefunction to APG wavefunction.

        Returns
        -------
        apg : APG
            APG wavefunction that corresponds to the given APIG wavefunction.

        """
        # import APG (because nasty cyclic import business)
        from .apg import APG

        # make apig coefficients
        apig_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)
        # make apg coefficients
        apg = APG(self.nelec, self.one_int, self.two_int, dtype=self.dtype, nuc_nuc=self.nuc_nuc,
                  orbtype=self.orbtype, ref_sds=self.ref_sds, ngem=self.ngem)
        apg_coeffs = apg.params[:-1].reshape(apg.template_coeffs.shape)
        # assign apg coefficients
        for orbpair, ind in self.dict_orbpair_ind.items():
            apg_coeffs[:, apg.dict_orbpair_ind[orbpair]] = apig_coeffs[:, ind]
        apg.assign_params(np.hstack((apg_coeffs.flat, self.get_energy())))

        # make wavefunction
        return apg
