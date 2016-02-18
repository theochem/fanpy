from __future__ import absolute_import, division, print_function

from itertools import combinations
import numpy as np
from scipy.misc import comb
from ..utils import permanent
from ..utils import slater


def __init__(self, n, q, H, G, *args, **kwargs):
    """
    Initialize the particle number symmetry-broken APIG class.

    See Base.__init__().

    """

    # Number of geminals generally != self.n // 2 for nsb geminals
    self.p = q
    super(self.__class__, self).__init__(n, H, G, *args, **kwargs)


def jacobian(self, x):
    """
    Compute the partial derivative of the objective function.

    Parameters
    ----------
    x : 1-index np.ndarray
        The coefficient vector.

    """

    # Update the coefficient vector
    self.x[:] = x
    jac = np.empty((len(self.pspace) + 0, self.x.size), dtype=x.dtype)

    # Loop through all coefficients
    c = 0
    for i in range(self.p):
        for j in range(self.k):

            # Intialize needed variables
            energy = sum(self.hamiltonian(self.ground))
            d_olp = self.overlap_deriv(self.ground, i, j)
            d_energy = sum(self.hamiltonian_deriv(self.ground, i, j))

            # Impose d<HF|Psi> == 1 + 0j
            #jac[-1, c] = d_olp

            # Impose dC[0, 0] == 1 + 0j
            #if c == 0:
                #jac[-2, c] = 1
            #else:
                #jac[-2, c] = 0

            # Impose (for all SDs in `pspace`) <SD|H|Psi> - E<SD|H|Psi> == 0
            for k, sd in enumerate(self.pspace):
                d_tmp = sum(self.hamiltonian_deriv(sd, i, j)) \
                    - energy * self.overlap_deriv(sd, i, j) - d_energy * self.overlap(sd)
                jac[k, c] = d_tmp

            # Move to the next coefficient
            c += 1

    return jac * 0.5


def objective(self, x):
    """
    Compute the objective function for solving the coefficient vector.
    The function is of the form "<sd|H|Psi> - E<sd|Psi> == 0 for all sd in pspace".

    Parameters
    ----------
    x : 1-index np.ndarray
        The coefficient vector.

    """

    # Update the coefficient vector
    self.x[:] = x

    # Intialize needed variables
    olp = self.overlap(self.ground)
    energy = sum(self.hamiltonian(self.ground))
    obj = np.empty((len(self.pspace) + 0), dtype=x.dtype)

    # Impose <HF|Psi> == 1
    #obj[-1] = olp - 1.0

    # Impose C[0, 0] == 1 + 0j
    #obj[-2] = self.x[0] - 1.0

    # Impose (for all SDs in `pspace`) <SD|H|Psi> - E<SD|H|Psi> == 0
    for i, sd in enumerate(self.pspace):
        obj[i] = sum(self.hamiltonian(sd)) - energy * self.overlap(sd)

    return obj
