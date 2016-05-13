from __future__ import absolute_import, division, print_function

from itertools import combinations, product

import numpy as np

from .fullci import FullCI
from .math import binomial
from . import slater


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
        ground = slater.ground(nelec)
        civec.append(ground)

        count = 1
        for nexc in range(1, npair + 1):
            occ_combinations = combinations(reversed(range(npair)), nexc)
            vir_combinations = combinations(range(npair, nspatial), nexc)
            for occ, vir in product(occ_combinations, vir_combinations):
                occ = [2*i for i in occ] + [2*i+1 for i in occ]
                vir = [2*a for a in vir] + [2*a+1 for a in vir]
                sd = slater.annihilate(ground, *occ)
                sd = slater.create(ground, *vir)
                civec.append(sd)
                count += 1
                if count == self.nproj:
                    return civec
        else:
            return civec
