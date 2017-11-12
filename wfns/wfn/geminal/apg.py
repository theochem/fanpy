"""Antisymmeterized Product of Geminals (APG) Wavefunction."""
from __future__ import absolute_import, division, print_function
from wfns.wfn.geminal.base_geminal import BaseGeminal
from wfns.backend.graphs import generate_complete_pmatch
from wfns.wrapper.docstring import docstring_class

__all__ = []


@docstring_class(indent_level=1)
class APG(BaseGeminal):
    r"""Antisymmeterized Product of Geminals (APG) Wavefunction.

    Each geminal is a linear combination of all possible (:math:`\binom{2K}{2}`) spin orbital pairs.

    .. math::

        G^\dagger_p = \sum_{i=1}^{2K} \sum_{j>i}^{2K} C_{pij} a^\dagger_i a^\dagger_j

    where all possible orbital pairs, :math:`a^\dagger_i a^\dagger_j`, are allowed to contribute to
    each geminal.

    """
    def generate_possible_orbpairs(self, occ_indices):
        """

        Generates all possible orbital pairing schemes from a given set of occupied orbitals. This
        is equivalent to finding all the perfect matchings (pairing schemes) within a complete graph
        with the given set of vertices (occupied orbitals).

        """
        yield from generate_complete_pmatch(occ_indices, is_decreasing=False)
