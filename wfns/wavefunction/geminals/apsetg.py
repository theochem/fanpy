"""Antisymmeterized Product of Set divided Geminals (APsetG) Wavefunction."""
from __future__ import absolute_import, division, print_function
from wfns.backend import graphs
from wfns.wavefunction.geminals.base_geminal import BaseGeminal
from wfns.wrapper.pydocstring import docstring_class

__all__ = []


@docstring_class(indent_level=1)
class APsetG(BaseGeminal):
    r"""Antisymmeterized Product of Set divided Geminals (APsetG) Wavefunction.

    .. math::

        G^\dagger_p = \sum_{i \in S_A} \sum_{j \in S_B} C_{pij} a^\dagger_i a^\dagger_j

    where :math:`S_A` and :math:`S_B` are two sets of orbitals that are mutually exclusive (no
    shared orbitals) and exhaustive (form the complete basis set as a whole).

    """
    def generate_possible_orbpairs(self, occ_indices):
        """

        Generates all possible orbital pairing schemes where one orbital is selected from one set
        and the other orbital is selected from the other. This is equivalent to finding all the
        perfect matchings (pairing schemes) within a bipartite graph with the given two sets
        (:math:`S_A` and :mat:`S_B`) of vertices (occupied orbitals).

        """
        alpha_occ_indices = []
        beta_occ_indices = []
        for i in occ_indices:
            if i < self.nspatial:
                alpha_occ_indices.append(i)
            else:
                beta_occ_indices.append(i)
        yield from graphs.generate_biclique_pmatch(alpha_occ_indices, beta_occ_indices,
                                                   is_decreasing=False)
