"""DOCI wavefunction."""
from __future__ import absolute_import, division, print_function
from wfns.wfn.ci.ci_wavefunction import CIWavefunction
from wfns.wrapper.docstring import docstring_class

__all__ = []


@docstring_class(indent_level=1)
class DOCI(CIWavefunction):
    r"""Doubly Occupied Configuration Interaction (DOCI) Wavefunction.

    CI wavefunction constructed from all of the seniority zero Slater determinants within the given
    basis. Seniority zero means that only doubly occupied spatial orbitals are used in the
    construction of each Slater determinant.

    """
    def assign_nelec(self, nelec):
        """

        Cannot create seniority zero Slater determinants with odd number of electrons.

        Raises
        ------
        ValueError
            If number of electrons is not a positive number.
            If number of electrons is not even.
        """
        super().assign_nelec(nelec)
        if self.nelec % 2 != 0:
            raise ValueError('`nelec` must be an even number')

    def assign_spin(self, spin=None):
        r"""

        Since the number of alpha and beta electrons are always equal (b/c seniority zero),
        wavefunction will always be a singlet.

        Raises
        ------
        ValueError
            If spin is not zero (singlet).
        """
        if spin is None:
            spin = 0
        super().assign_spin(spin)
        if self.spin != 0:
            raise ValueError('DOCI wavefunction can only be singlet')

    def assign_seniority(self, seniority=None):
        r"""

        Raises
        ------
        ValueError
            If the seniority is a negative integer.
            If the seniority is not zero.
        """
        if seniority is None:
            seniority = 0
        super().assign_seniority(seniority)
        if self.seniority != 0:
            raise ValueError('DOCI wavefunction can only be seniority 0')
