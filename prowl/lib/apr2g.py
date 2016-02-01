from __future__ import absolute_import, division, print_function

import numpy as np
from ..utils import permanent, slater


def generate_guess(self):
    """
    Generate an appropriate random guess for the coefficient vector `x`.

    Returns
    -------
    x : 1-index np.ndarray

    """

    x = np.empty(4 * self.k + 2 * self.p)
    x[:(4 * self.k)] = np.random.rand(4 * self.k)
    x[(4 * self.k):] = np.random.rand(2 * self.p)

    return x


def generate_view(self):
    """
    Generate a view of `x` corresponding to the shape of the coefficient array.

    Returns
    -------
    view : 2-index np.ndarray

    """

    # Initialize `_C`, the cached value of the recomposed Cauchy matrix
    self._C = np.empty((self.p, self.k), dtype=np.complex128)

    #return self.x.reshape(self.p + self.k, 2).view(np.complex128)[:, 0]
    return self.x


def overlap(self, sd):
    """
    Compute the overlap of the wavefunction with `sd`.

    Parameters
    ----------
    sd : int
        The Slater Determinant against which to project.

    """

    # If the Slater determinant is bad
    if sd is None:
        return 0
    # If the SD and the wavefunction have a different number of electrons
    elif slater.number(sd) != self.n:
        return 0
    # If the SD is not a closed-shell singlet
    elif any(slater.occupation(sd, 2 * i)  != slater.occupation(sd, 2 * i + 1) for i in range(self.k)):
        return 0

    # Evaluate the overlap
    occ = [i for i in range(self.k) if slater.occupation_pair(sd, i)]
    return permanent.borchardt(self._C[:, occ])


def objective(self, x):
    """
    Compute the objective function for solving the coefficient vector.
    The function is of the form "<sd|H|Psi> - E<sd|Psi> == 0 for all sd in pspace".

    Parameters
    ----------
    x : 1-index np.ndarray
        The coefficient vector.

    """

    # Update the coefficient vector (and the `_C` matrix cache in APr2G)
    self.x[:] = x
    self._update_C()

    # Intialize needed variables
    olp = self.overlap(self.ground)
    coeff = self._C[0, 0]
    energy = sum(self.hamiltonian(self.ground))
    obj = np.empty(2 * len(self.pspace) + 3)

    # Impose <HF|Psi> == 1
    obj[-1] = np.real(olp) - 1.0
    obj[-2] = np.imag(olp)

    # Impose C[0, 0] == 1
    obj[-3] = np.real(coeff) - 1.0
    obj[-4] = np.imag(coeff)

    # Impose (for all SDs in `pspace`) <SD|H|Psi> - E<SD|H|Psi> == 0
    for i, sd in enumerate(self.pspace[1:]):
        tmp = sum(self.hamiltonian(sd)) - energy * self.overlap(sd)
        obj[2 * i] = np.real(tmp) - 1.0
        obj[2 * i + 1] = np.imag(tmp)

    print(np.sum(obj ** 2))
    return obj


def _update_C(self):
    """
    Compute the matrix corresponding to the shape of the
    Borchardt-decomposed coefficient array.

    """

    for i in range(self.p):
        for j in range(self.k):
            self._C[i, j] = (self.x[j] + self.x[self.k + j] * 1j) \
            / (self.x[4 * self.k + i] \
             + self.x[4 * self.k + self.p + i] * 1j \
             - self.x[2 * self.k + j] \
             - self.x[3 * self.k + j] * 1j)
