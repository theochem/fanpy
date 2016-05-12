from __future__ import absolute_import, division, print_function

from itertools import combinations, product

import numpy as np
import gmpy2

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

        nelec = self.nelec
        nspatial = self.nspatial
        npair = self.npair

        civec = []

        # ASSUME: certain structure for civec
        # spin orbitals are shuffled (alpha1, beta1, alph2, beta2, etc)
        # spin orbitals are ordered by energy
        ground = gmpy2.mpz(0)
        for i in range(nelec):
            # Add electrons
            ground |= 1 << i
        civec.append(ground)

        count = 1
        for nexc in range(1, npair + 1):
            occ_combinations = combinations(reversed(range(npair)), nexc)
            vir_combinations = combinations(range(npair, nspatial), nexc)
            for occ, vir in product(occ_combinations, vir_combinations):
                sd = ground
                for i in occ:
                    # Remove electrons
                    sd &= ~(1 << (2 * i + 1))
                    sd &= ~(1 << (2 * i))
                for a in vir:
                    # Add electrons
                    sd |= 1 << (2 * a + 1)
                    sd |= 1 << (2 * a)
                civec.append(sd)
                count += 1
                if count == self.nproj:
                    return civec
        else:
            return civec
