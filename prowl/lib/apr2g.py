from __future__ import absolute_import, division, print_function

import sys
import numpy as np
from ..utils import permanent, slater


def generate_guess(self):
    """
    Generate an appropriate random guess for the coefficient vector `x`.

    Returns
    -------
    x : 1-index np.ndarray

    """

    x = np.zeros((self.p + 2 * self.k), dtype=self.dtype)
    x += np.random.rand(x.size)
    x[self.p:(self.p + self.k)] *= -1

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
    self._C[:] = permanent.cauchy_factor(self.x, self.p)
    jac = np.empty((len(self.pspace) + 2, x.size), dtype=x.dtype)

    # Intialize constant "variables"
    energy = sum(self.hamiltonian(self.ground))

    # Loop through all coefficients
    for c in range(x.size):

        # Intialize differentiated variables
        d_olp = self.overlap_deriv(self.ground, c)
        d_energy = sum(self.hamiltonian_deriv(self.ground, c))

        # Impose <HF|Psi> == 1, C[0, 0] == 1
        jac[-1, c] = d_olp
        if c == 0:
            jac[-2, c] = -self.x[self.p + self.k] / ((self.x[0] - self.x[self.p]) ** 2)
        elif c == self.p:
            jac[-2, c] = self.x[self.p + self.k] / ((self.x[0] - self.x[self.p]) ** 2)
        elif c == (self.p + self.k):
            jac[-2, c] = 1 / (self.x[0] - self.x[self.p])
        else:
            jac[-2, c] = 0

        # Impose (for all SDs in `pspace`) d(<SD|H|Psi> - E<SD|H|Psi>) == 0
        for i, sd in enumerate(self.pspace):
            jac[i, c] = sum(self.hamiltonian_deriv(sd, c)) \
                - energy * self.overlap_deriv(sd, c) - d_energy * self.overlap(sd)

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
    self._C[:] = permanent.cauchy_factor(self.x, self.p)

    # Intialize needed variables
    olp = self.overlap(self.ground)
    energy = sum(self.hamiltonian(self.ground))
    obj = np.empty((len(self.pspace) + 2), dtype=x.dtype)

    # Impose <HF|Psi> == 1, C[0, 0] == 1
    obj[-1] = olp - 1.0
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
    return permanent.apr2g(self.x, self.p, occ)


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

    # Get occupied orbitals, i.e., columns
    occ = [i for i in range(self.k) if slater.occupation_pair(sd, i)]

    # Evaluate the overlap
    indices = list(range(self.p))
    indices.extend([self.p + i for i in occ])
    indices.extend([self.p + self.k + i for i in occ])
    if c not in indices:
        return 0
    return permanent.apr2g_deriv(self.x, self.p, occ, c)


def solve(self, **kwargs):
    """
    Optimize `self.objective(x)` to solve the coefficient vector.
    See `prowl.Base`.

    """

    # If APr2G has more parameters than APIG (for small `p`s and `k`s),
    # we need to restrict some of the parameters manually
    ubound = np.ones_like(self.x) * np.inf
    ubound[(self.p + self.k):] = 1
    lbound = -ubound
    if (self.p < 3) or (self.p > 2 and self.k < (self.p / (self.p - 2))):
        machine_eps = sys.float_info.epsilon
        extras = self.p + 2 * self.k - self.p * self.k
        for i in range(extras, self.p):
            ubound[i] = 1.0 + 10 * machine_eps
            lbound[i] = 1.0 - 10 * machine_eps
            self.x[i] = 1.0
    self.bounds = (lbound, ubound)

    return super(self.__class__, self).solve(**kwargs)
