from __future__ import absolute_import, division, print_function

from itertools import combinations
import numpy as np
from scipy.misc import comb
from ..utils import permanent
from ..utils import slater


def generate_guess(self):
    """
    Generate an appropriate random guess for the coefficient vector `x`.

    Returns
    -------
    x : 1-index np.ndarray

    """

    # `np.complex128` arrays are partitioned into columns like this:
    # [[Re][Im][Re][Im]...]; generate an array to handle this
    x = np.zeros((self.p, 2 * self.k))

    # Add the real identity
    x[:, 0::2] += np.eye(self.p, self.k)

    # Add real and imaginary random noise
    x[:, 0::2] += (0.2 / x.size) * (np.random.rand(self.p, self.k) - 0.5)
    x[:, 1::2] += (0.2 / x.size) * (np.random.rand(self.p, self.k) - 0.5)

    # Normalize
    x[:, 0::2] /= np.max(x[0::2])
    x[:, 1::2] /= np.max(x[1::2])

    # Make it a vector
    x = x.ravel()

    return x


def generate_pspace(self):
    """
    Generate an appropriate projection space for solving the coefficient vector `x`.

    Returns
    -------
    pspace : list
        List of Python ints representing Slater determinants.

    """

    # Determine the minimum length of `pspace`, and the ground state
    params = self.x.size
    ground = sum(2 ** i for i in range(self.n))
    pspace = [ground]

    # Get occupied (HOMOs first!) and virtual indices
    occ = list(range(self.p - 1, -1, -1))
    vir = list(range(self.p, self.k))

    # Add pair excitations
    nexc = 1
    nocc = len(occ) + 1
    nvir = len(vir) + 1
    while params > len(pspace) and nexc < nocc:
        # Determine the smallest usable set of frontier orbitals
        for i in range(2, nocc):
            if comb(i, 2, exact=True) ** 2 >= params - len(pspace):
                nfrontier = i
                break
        else:
            nfrontier = max(nocc, nvir)
        # Add excitations from all combinations of `nfrontier` HOMOs...
        for i in combinations(occ[:nfrontier], nexc):
            # ...to all combinations of `nfrontier` LUMOs
            for j in combinations(vir[:nfrontier], nexc):
                sd = ground
                for k, l in zip(i, j):
                    sd = slater.excite_pair(sd, k, l)
                pspace.append(sd)
        nexc += 1

    # Add single excitations (some betas to alphas) if necessary
    # (usually only for very small systems)
    if params > len(pspace):
        for i in occ:
            for j in vir:
                sd = slater.excite(ground, i * 2 + 1, j * 2)
                pspace.append(sd)

    # Return the sorted `pspace`
    pspace.sort()
    return pspace


def generate_view(self):
    """
    Generate a view of `x` corresponding to the shape of the coefficient array.

    Returns
    -------
    view : 2-index np.ndarray

    """

    return (self.x.reshape(self.p, 2 * self.k)).view(np.complex128)


def hamiltonian(self, sd):
    """
    Compute the Hamiltonian of the wavefunction projected against `sd`.

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

    olp = self.overlap(sd)

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
                    coulomb_tmp += self.G[i, i, j, j] * self.overlap(exc)

    one_electron *= olp
    exchange *= olp
    coulomb = coulomb * olp + coulomb_tmp

    return one_electron, coulomb, exchange


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
    elif any(slater.occupation(sd, 2 * i) != slater.occupation(sd, 2 * i + 1) for i in range(self.k)):
        return 0

    # Evaluate the overlap
    occ = [i for i in range(self.k) if slater.occupation_pair(sd, i)]
    return permanent.dense(self.C[:, occ])


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
    obj = np.empty(2 * len(self.pspace) + 2)

    # Impose <HF|Psi> == 1
    obj[0] = np.real(olp) - 1.0
    obj[1] = np.imag(olp)

    # Impose C[0, 0] == 1 + 0j
    obj[2] = self.x[0] - 1
    obj[3] = self.x[1]

    # Impose (for all SDs in `pspace`) <SD|H|Psi> - E<SD|H|Psi> == 0
    for i, sd in enumerate(self.pspace[1:]):
        tmp = sum(self.hamiltonian(sd)) - energy * self.overlap(sd)
        obj[2 * i + 4] = np.real(tmp)
        obj[2 * i + 5] = np.imag(tmp)

    print(np.sum(obj**2))
    return obj
