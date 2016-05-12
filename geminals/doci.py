from __future__ import absolute_import, division, print_function

from itertools import combinations, product

import numpy as np

from .fullci import FullCI
from .math import binomial


class DOCI(FullCI):

    #
    # Default attribute values
    #

    @property
    def _nproj_default(self):

        return binomial(self.nspatial, self.npair)

    #
    # Computation methods
    #

    def compute_civec(self):

        civec = np.zeros((self.nproj, self.nspin), dtype=bool)
        civec[:, :self.nelec] = True
        civeca = civec[:, 0::2]
        civecb = civec[:, 1::2]

        count = 1
        for nexc in range(1, self.npair + 1):
            for occ, vir in product(
                combinations(range(self.npair - 1, -1, -1), nexc),
                combinations(range(self.npair, self.nspatial), nexc),
            ):
                civeca[count, occ] = False
                civeca[count, vir] = True
                count += 1
                if count == self.nproj:
                    break

        civecb[:] = civeca

        return civec

    def compute_projection(self, sd=0, vec=0, deriv=None):

        if vec == 0:
            index = self.civec_index[sd]
        elif vec == 1:
            index = self.Hvec_index[sd]
        elif vec == 2:
            index = self.Gvec_index[sd]

        if deriv is None:
            try:
                return self.C[index]
            except IndexError:
                return 0.0
            return proj
        elif deriv == index:
            return 1.0
        else:
            return 0.0

    def is_valid_sd(self, sd):

        if np.all(sd[0::2] == sd[1::2]):
            return True
        else:
            return False
