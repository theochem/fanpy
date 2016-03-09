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


def density_matrix(self, val_threshold=1e-4):
    """ Returns the first and second order density matrices

    Second order density matrix uses the Physicist's notation:
    ..math::
        \Gamma_{ijkl} = < \Psi | a_i^\dagger a_k^\dagger a_l a_j | \Psi >
    Chemist's notation is also implemented
    ..math::
        \Gamma_{ijkl} = < \Psi | a_i^\dagger a_j^\dagger a_k a_l | \Psi >
    but is commented out

    Paramaters
    ----------
    val_threshold : float
        If the term has weight that is less than this threshold, it is discarded

    Returns
    -------
    one_density : np.ndarray(self.k, self.k)
        One electron density matrix
    two_density : np.ndarray(self.k, self.k, self.k, self.k)
        Two electron density matrix
    """
    temp_sorting = sorted(zip(self.x, self.pspace), key=lambda x: x[0], reverse=True)
    sorted_x, sorted_sd = zip(*temp_sorting)

    one_density = np.zeros([self.k]*2)
    two_density = np.zeros([self.k]*4)
    for i in range(self.k):
        for sd in self.pspace:
            if slater.occupation_pair(sd, i):
                olp = self.overlap(sd)
                one_density[i, i] += 2*olp**2
        for j in range(i, self.k):
            for sd in self.pspace:
                # i is always occupied
                if not slater.occupation_pair(sd, i):
                    continue
                num = 0
                # if i and j are equal and both occupied
                if i == j:
                    num = 2*self.overlap(sd)**2
                    two_density[i, i, i, i] += num
                # if i is occupied and j is virtual
                elif not slater.occupation_pair(sd, j):
                    exc = slater.excite_pair(sd, i, j)
                    num = 2*self.overlap(sd)*self.overlap(exc)
                    #\Gamma_{ijkl} = < \Psi | a_i^\dagger a_k^\dagger a_l a_j | \Psi >
                    #two_density[i, j, i, j] += num
                    #two_density[j, i, j, i] += num
                    #\Gamma_{ijkl} = < \Psi | a_i^\dagger a_j^\dagger a_k a_l | \Psi >
                    two_density[i, i, j, j] += num
                    two_density[j, j, i, i] += num
                # if i and j are occupied and i < j
                else:
                    num = self.overlap(sd)**2
                    #\Gamma_{ijkl} = < \Psi | a_i^\dagger a_k^\dagger a_l a_j | \Psi >
                    #two_density[i, i, j, j] += 4*num
                    #two_density[j, i, i, j] -= 2*num
                    #two_density[i, j, j, i] -= 2*num
                    #two_density[j, j, i, i] += 4*num
                    #\Gamma_{ijkl} = < \Psi | a_i^\dagger a_j^\dagger a_k a_l | \Psi >
                    two_density[i, j, i, j] += 4*num
                    two_density[j, j, i, i] -= 2*num
                    two_density[i, i, j, j] -= 2*num
                    two_density[j, i, j, i] += 4*num
                if abs(num) < val_threshold:
                    break
    return one_density, two_density
