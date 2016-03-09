from __future__ import absolute_import, division, print_function

from itertools import combinations
import numpy as np
from scipy.misc import comb
from ..utils import slater



def generate_pspace(self):
    """
    Generate an appropriate projection space for solving the coefficient vector `x`.

    Returns
    -------
    pspace : list
        List of Python ints representing Slater determinants.

    """
    # Find the ground state and occupied/virtual indices
    ground = sum(2 ** i for i in range(self.n))
    pspace = [ground]

    # Get occupied (HOMOs first!) and virtual indices
    occ = list(range(self.n//2-1, -1, -1))
    vir = list(range(self.n//2, self.k))

    # Add nth order pair excitation
    for n in range(1, self.k+1):
        for excite_from in combinations(occ, n):
            for excite_to in combinations(vir, n):
                # TODO: Need excite_multiple_pairs function
                sd = ground
                for i in excite_from:
                    sd = slater.annihilate_pair(sd, i)
                for a in excite_to:
                    sd = slater.create_pair(sd, a)
                pspace.append(sd)
    # Return the sorted `pspace`
    pspace.sort()
    return pspace
