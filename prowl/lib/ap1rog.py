from __future__ import absolute_import, division, print_function

from itertools import combinations
import numpy as np
from ..utils import slater
from ..utils import permanent


def generate_guess(self):
    """
    Generate an appropriate random guess for the coefficient vector `x`.

    Returns
    -------
    x : 1-index np.ndarray

    """

    params = self.p * (self.k - self.p)
    x = (2 / params) * (np.random.rand(params) - 0.5)

    return x


def generate_pspace(self):
    """
    Generate an appropriate projection space for solving the coefficient vector `x`.

    Returns
    -------
    pspace : list
        List of Python ints representing Slater determinants.

    """

    # Determine the ground state, occupied (HOMOs first!) and virtual indices
    ground = sum(2 ** i for i in range(self.n))
    pspace = [ground]
    occ = list(range(self.p - 1, -1, -1))
    vir = list(range(self.p, self.k))

    # Add all single (and double, if necessary) pair excitations
    for nexc in (1, 2):
        for i in combinations(occ, nexc):
            for j in combinations(vir, nexc):
                sd = ground
                for k, l in zip(i, j):
                    sd = slater.excite_pair(sd, k, l)
                pspace.append(sd)

        # If singles gave us enough determinants
        if len(pspace) > self.x.size:
            break

    # Return sorted `pspace`
    pspace.sort()
    return pspace


def generate_view(self):
    """
    Generate a view of `x` corresponding to the shape of the coefficient array.

    Returns
    -------
    view : 2-index np.ndarray

    """

    return self.x.reshape(self.p, self.k - self.p)


def jacobian(self, x):
    """
    Compute the Jacobian of the objective function.

    Parameters
    ----------
    x : 1-index np.ndarray
        The coefficient vector.

    """

    # Update the coefficient vector
    self.x[:] = x
    jac = np.empty((len(self.pspace), x.size), dtype=x.dtype)

    # Intialize constant "variables"
    energy = sum(self.hamiltonian(self.ground))

    # Loop through all coefficients
    c = 0

    for i in range(self.p):
        for j in range(self.k - self.p):

            # Intialize differentiated variables
            d_olp = self.overlap_deriv(self.ground, i, j)
            d_energy = sum(self.hamiltonian_deriv(self.ground, i, j))

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

    # Initialize needed variables
    energy = sum(self.hamiltonian(self.ground))
    obj = np.empty((len(self.pspace),), dtype=x.dtype)

    # Impose (for all SDs in `pspace`) <SD|H|Psi> - E<SD|Psi> == 0
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
    elif any(slater.occupation(sd, i * 2) != slater.occupation(sd, i * 2 + 1) for i in range(self.k)):
        return 0

    # Evaluate the overlap
    nexc = 0
    occ = []
    vir = []
    for i in range(self.p):
        if not slater.occupation_pair(sd, i):
            nexc += 1
            occ.append(i)
    for i in range(self.p, self.k):
        if slater.occupation_pair(sd, i):
            vir.append(i - self.p)

    # If `sd` is not excited
    if nexc == 0:
        return 1.0
    else:
        return permanent.dense(self.C[occ][:, vir])


def overlap_deriv(self, sd, x, y):
    """
    Compute the overlap of the wavefunction with `sd`.

    Parameters
    ----------
    sd : int
        The Slater Determinant against which to project.

    """

    # If the Slater determinant is bad
    if sd is None:
        return 0.0
    # If the SD and the wavefunction have a different number of electrons
    elif slater.number(sd) != self.n:
        return 0.0
    # If the SD is not a closed-shell singlet
    elif any(slater.occupation(sd, i * 2) != slater.occupation(sd, i * 2 + 1) for i in range(self.k)):
        return 0.0

    # Evaluate the overlap
    nexc = 0
    occ = []
    vir = []
    for i in range(self.p):
        if not slater.occupation_pair(sd, i):
            nexc += 1
            occ.append(i)
    for i in range(self.p, self.k):
        if slater.occupation_pair(sd, i):
            vir.append(i - self.p)

    if x in occ and y in vir:
        return permanent.dense_deriv(self.C[occ][:, vir], occ.index(x), vir.index(y))
    else:
        return 0.0
