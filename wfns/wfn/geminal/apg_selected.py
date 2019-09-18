"""Antisymmeterized Product of Selected Geminals (APG) Wavefunction."""
import numpy as np

from wfns.backend.graphs import generate_general_pmatch
from wfns.wfn.geminal.apg import APG


class APGSelected(APG):
    def generate_possible_orbpairs(self, occ_indices):
        """Yield all possible orbital pairs that can construct the given Slater determinant.

        Generates all possible orbital pairing schemes from a given set of occupied orbitals. This
        is equivalent to finding all the perfect matchings (pairing schemes) within a complete graph
        with the given set of vertices (occupied orbitals).

        Parameters
        ----------
        occ_indices : N-tuple of int
            Indices of the orbitals from which the Slater determinant is constructed.
            Must be strictly increasing.

        Yields
        ------
        orbpairs : P-tuple of 2-tuple of ints
            Indices of the creation operators (grouped by orbital pairs) that construct the Slater
            determinant.
        sign : int
            Signature of the transpositions required to shuffle the `orbpairs` back into the
            original order in `occ_indices`.

        """
        dead_indices = np.where(np.sum(np.abs(self.params), axis=0) < self.tol)[0]
        dead_orbpairs = set([self.get_orbpair(ind) for ind in dead_indices])
        for pmatch in generate_general_pmatch(
                occ_indices, np.ones((len(occ_indices), len(occ_indices)))
        ):
            if any(i in dead_orbpairs for i in pmatch):
                continue
            yield pmatch
