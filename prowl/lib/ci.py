from __future__ import absolute_import, division, print_function

from itertools import combinations
import numpy as np
from scipy.misc import comb
from ..utils import slater



def __init__(self, n, H, G, sds=None, dtype=None, pspace=None, x=None):
    """
    Initialize the base wavefunction class.

    Parameters
    ----------
    n : int
        The number of electrons.
    H : 2-index np.ndarray, dtype=np.float64
        The one-electron Hamiltonian array in the MO basis.
    G : 4-index np.ndarray, dtype=np.float64
        The two-electron Hamiltonian array in the MO basis.
    sds : list of int
        List of Python ints representing Slater determinants that make up the
        N electron basis set
        If not specified, the Slater determinants for the CISD (all single and
        double excitations) are used
    dtype : {np.float64, np.complex128}
    pspace : list, optional
        A list of Python ints representing Slater Determinants that make up the
        projection space
        If not specified, it is made to be the same as the sds
    x : 1-index np.ndarray, dtype=np.float64, optional
        The coefficient vector with initial guess values.
        If not specified, an appropriate guess `x` is generated.
    """

    # System attributes
    self.n = n
    self.k = H.shape[0]
    self.H = H
    self.G = G

    # Projection space attributes
    self.sds = self.generate_sds() if sds is None else sds
    self.pspace = self.sds if pspace is None else pspace
    self.ground = min(self.pspace)

    # Coefficient attributes
    self.dtype = self.dtype if dtype is None else dtype
    self.x = self.generate_guess() if x is None else x


def generate_guess(self):
    """
    Generate an appropriate random guess for the coefficient vector `x`.

    Returns
    -------
    x : 1-index np.ndarray

    """
    num_sds = len(self.sds)
    # Generate an empty array of the appropriate shape
    x = np.zeros(num_sds, dtype=self.dtype)
    x[0] += 1.0

    # Add random noise
    x += (0.2 / x.size) * (np.random.rand(num_sds) - 0.5)
    if self.dtype == np.complex128:
        x += (0.2 / x.size) * (np.random.rand(num_sds) - 0.5) * 1j

    # Normalize
    x[1:] /= np.max(x)

    return x


def generate_sds(self):
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
    occ = list(range(self.n-1, -1, -1))
    vir = list(range(self.n, self.k))
    # Add Single excitations
    for i in occ:
        for j in vir:
            pspace.append(slater.excite(ground, i, j))
    # Add Double excitations
    for i,j in combinations(occ, 2):
        for k,l in combinations(vir, 2):
            sd = slater.excite(ground, i, k)
            pspace.append(slater.excite(sd, j, l))

    # Return the sorted `pspace`
    pspace.sort()
    return pspace
