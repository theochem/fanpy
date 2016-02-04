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

    x = np.empty((2 * self.k + self.p), dtype=self.dtype)
    x += np.random.rand(2 * self.k + self.p)
    if self.dtype == np.complex128:
        x += np.random.rand(2 * self.k + self.p) * 1j
        x /= np.max(np.abs(np.real(x)))

    return x


def generate_view(self):
    """
    Generate a view of `x` corresponding to the shape of the coefficient array.

    Returns
    -------
    view : 2-index np.ndarray

    """

    # Initialize `_C`, the cached value of the recomposed Cauchy matrix
    self._C = np.empty((self.p, self.k), dtype=self.dtype)
    return self.x


def hamiltonian_deriv(self, sd, c):
    """
    Compute the partial derivative of the Hamiltonian with respect to parameter `(x, y)`.

    Parameters
    ----------
    sd : int
        The Slater Determinant against which to project.
    c : int

    Returns
    -------
    energy : tuple
        Tuple of floats corresponding to the one electron, coulomb, and exchange
        energies.

    """

    d_olp = self.overlap_deriv(sd, c)

    one_electron = 0.0
    coulomb = 0.0
    coulomb_tmp = 0.0
    exchange = 0.0

    for i in range(self.k):
        if slater.occupation_pair(sd, i):
            one_electron += 2 * self.H[i, i]
            coulomb += self.G[i, i, i, i]
            for j in range( i + 1, self.k):
                if slater.occupation_pair(sd, j):
                    coulomb += 4 * self.G[i, j, i, j]
                    exchange -= 2 * self.G[i, j, j, i]
            for j in range(self.k):
                if not slater.occupation_pair(sd, j):
                    exc = slater.excite_pair(sd, i, j)
                    coulomb_tmp += self.G[i, i, j, j] * self.overlap_deriv(exc, c)

    one_electron *= d_olp
    exchange *= d_olp
    coulomb = coulomb * d_olp + coulomb_tmp

    return one_electron, coulomb, exchange


def jacobian(self, x):
    """
    Compute the Jacobian of the objective function.

    Parameters
    ----------
    x : 1-index np.ndarray
        The coefficient vector.

    """

    # Update the coefficient vector (and the `_C` matrix cache in APr2G)
    self.x[:] = x
    self._C[:] = permanent.borchardt_factor(self.C, self.p)
    jac = np.empty((len(self.pspace) + 2, x.size), dtype=x.dtype)

    # Intialize constant "variables"
    energy = sum(self.hamiltonian(self.ground))

    # Loop through all coefficients
    for c in range(x.size):

        # Intialize differentiated variables
        d_olp = self.overlap_deriv(self.ground, c)
        d_energy = sum(self.hamiltonian_deriv(self.ground, c))

        # Impose d<HF|Psi> == 0
        jac[-1, c] = d_olp

        # Impose dC[0, 0] == 0
        if c == 0:
            tmp = 1 / (self.x[self.k] + self.x[2 * self.k])
        elif c == self.k:
            tmp = self.x[0] / (1 + self.x[2 * self.k])
        elif c == 2 * self.k:
            tmp = self.x[0] / (1 + self.x[self.k])
        else:
            tmp = 0
        jac[-2, c] = tmp

        # Impose (for all SDs in `pspace`) d(<SD|H|Psi> - E<SD|H|Psi>) == 0
        for i, sd in enumerate(self.pspace):
            d_tmp = sum(self.hamiltonian_deriv(sd, c)) \
                - energy * self.overlap_deriv(sd, c) - d_energy * self.overlap(sd)
            jac[i, c] = d_tmp

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

    # Update the coefficient vector (and the `_C` matrix cache in APr2G)
    self.x[:] = x
    self._update_C()

    # Intialize needed variables
    olp = self.overlap(self.ground)
    energy = sum(self.hamiltonian(self.ground))
    obj = np.empty((len(self.pspace) + 2), dtype=x.dtype)

    # Impose <HF|Psi> == 1
    obj[-1] = olp - 1.0

    # Impose C[0, 0] == 1
    obj[-2] = self._C[0, 0] - 1.0

    # Impose (for all SDs in `pspace`) <SD|H|Psi> - E<SD|H|Psi> == 0
    for i, sd in enumerate(self.pspace):
        obj[i] = sum(self.hamiltonian(sd)) - energy * self.overlap(sd)

    return obj


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


def overlap_deriv(self, sd, c):
    """
    Compute the partial derivative of the overlap with respect to parameter `c`

    Parameters
    ----------
    sd : int
        The Slater Determinant against which to project.
    c : int

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
    if c not in occ and (c - self.k) not in occ and (c - 2 * self.k) not in occ:
        return 0
    else:
        cols = []
        cols.extend(occ)
        cols.extend([i + self.k for i in occ])
        cols.extend(list(range(2 * self.k, 2 * self.k + self.p)))
        nc = cols.index(c)
        return permanent.borchardt_deriv(self._C[:, occ], self.C[cols], nc)


def _update_C(self):
    """
    Compute the matrix corresponding to the shape of the
    Borchardt-decomposed coefficient array.

    """

    for i in range(self.p):
        for j in range(self.k):
            self._C[i, j] = self.x[j] / (self.x[1 * self.k + j] + self.x[2 * self.k + i])
