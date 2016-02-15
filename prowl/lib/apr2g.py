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

    x = np.ones((self.p + 2 * self.k), dtype=self.dtype)

    # Lambdas in ascending order
    for i in range(self.p):
        x[i] = i + 1
        #x[i] = (i + 1) * self.p / 2

    # Epsilons in step with the lambdas
    x[self.p] = 0
    for j in range(1, self.p):
        x[self.p + j] = 0.5 * (x[j - 1] + x[j])
    for j in range(self.p, self.k):
        x[self.p + j] = self.p + j

    # Zetas in descending order
    for j in range(self.k):
        x[self.p + self.k + j] = 1.0

    # Add noise
    #x += (np.random.rand(x.size) - 0.5) * 2.0e-3
    if x.dtype == np.complex128:
        x += (np.random.rand(x.size) - 0.5) * 1.0e-2j

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
    zeta_indices = list(range(self.p + self.k, 2 * self.p + self.k))
    olp = self.overlap(self.ground)
    energy = sum(self.hamiltonian(self.ground))

    # Loop through all coefficients
    for c in range(x.size):

        # Intialize differentiated variables
        d_olp = self.overlap_deriv(self.ground, c)
        d_energy = sum(self.hamiltonian_deriv(self.ground, c))

        # Impose some normalization constraints (this is currently incorrect)
        if c < self.p:
            jac[-1, c] = -self.x[self.p + self.k + c] / ((self.x[c] - self.x[self.p + c]) ** 2)
            jac[-2, c] = -np.prod([self._C[i, i] for i in range(self.p)]) / (self.x[c] - self.x[self.p + c])
        elif self.p < c < 2 * self.p:
            jac[-1, c] = self.x[self.p + self.k + c] / ((self.x[c] - self.x[self.p + c]) ** 2)
            jac[-2, c] = np.prod([self._C[i, i] for i in range(self.p)]) / (self.x[c] - self.x[self.p + c])
        elif (self.p + self.k) < c < (2 * self.p + self.k):
            jac[-1, c] = 1 / (self.x[c] - self.x[self.p + c])
            jac[-2, c] = np.prod([self._C[i, i] for i in range(self.p) if i != c]) / (self.x[c] - self.x[self.p + c])
        else:
            jac[-1, c] = 0
            jac[-2, c] = 0
        jac[-1, c] *= self.p
        jac[-2, c] *= self.p

        # Impose (for all SDs in `pspace`) d(<SD|H|Psi> - E<SD|H|Psi>) == 0
        for i, sd in enumerate(self.pspace):
            sd_olp = self.overlap(sd)
            sd_d_olp = self.overlap_deriv(sd, c)
            jac[i, c] = (olp * sum(self.hamiltonian_deriv(sd, c)) - d_olp * sum(self.hamiltonian(sd))) / olp ** 2 \
                - (olp * sd_olp * d_energy + energy * olp * sd_d_olp - energy * sd_olp * d_olp) / olp ** 2

    # Fix NaNs (not sure if we can avoid this... but we'll try)
    for i in range(jac.shape[0]):
        for j in range(jac.shape[1]):
            if not np.isfinite(jac[i, j]):
                jac[i, j] = 0.0

    return jac * 0.5


def objective(self, x):
    """
    Compute the objective function for solving the coefficient vector.
    The function is of the form "<sd|H|Psi> - E<sd|Psi> == 0 for all sd in pspace".

    Parameters
    ----------
    x : 1-index np.ndarray
        The coefficient vector.

    Notes
    -----
    Dividing by the ground-state overlap is good.  Doing it in the normalization
    equations seems to be enough, but if that changes, we could divide the
    projected Schrodinger equations by the overlap.

    """

    # Update the coefficient vector (and the `_C` matrix cache in APr2G)
    self.x[:] = x
    self._C[:] = permanent.cauchy_factor(self.x, self.p)

    # Intialize needed variables
    olp = self.overlap(self.ground)
    energy = sum(self.hamiltonian(self.ground))
    obj = np.empty((len(self.pspace) + 2), dtype=x.dtype)

    # Impose some normalization constraints
    # NOTE: divide these by olp, it might help.
    obj[-1] = np.sum([self._C[i, i] for i in range(self.p)]) - self.p
    obj[-2] = np.prod([self._C[i, i] for i in range(self.p)]) - 1
    #obj[-2] = np.prod(self._C[:, :self.p]) - 1
    obj[-1] *= self.p
    obj[-2] *= self.p

    # Impose (for all SDs in `pspace`) <SD|H|Psi> - E<SD|H|Psi> == 0
    for i, sd in enumerate(self.pspace):
        obj[i] = (sum(self.hamiltonian(sd)) - energy * self.overlap(sd)) / olp

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
    return permanent.apr2g(self._C[:, occ])#self.x, self.p, occ)


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
    return permanent.apr2g_deriv(self._C, self.x, self.p, occ, c)


def solve(self, **kwargs):
    """
    Optimize `self.objective(x)` to solve the coefficient vector.
    See `prowl.Base`.

    """

    eps = 10 * sys.float_info.epsilon
    ubound = np.ones_like(self.x)
    ubound[:(self.p + self.k)] *= np.inf
    lbound = -ubound
    # NOTE: need to test bounds further and restrict for small systems

    return super(self.__class__, self).solve(**kwargs)
