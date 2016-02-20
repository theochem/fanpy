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
    occ = list(range(self.n // 2 - 1, -1, -1))
    vir = list(range(self.n // 2, self.k))

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
    occ = []
    vir = []
    for i in range(self.p):
        if not slater.occupation_pair(sd, i):
            occ.append(i)
    for i in range(self.p, self.k):
        if slater.occupation_pair(sd, i):
            vir.append(i - self.p)

    if x in occ and y in vir:
        return permanent.dense_deriv(self.C[occ][:, vir], occ.index(x), vir.index(y))
    else:
        return 0.0


def truncate(self, val_threshold=1e-4):
    """ Truncates the CI expansion of APIG wavefunction such that the overlap between the
    truncated and the original wavefunction is close enough to 1

    Uses "inequality of Hadamard type for permanents"
    ..math::
        | A |^+ \leq \frac{N!} {N^{N/2}} \prod_{j=1}^N || \vec{A}^j ||
    where :math: `\vec{A}^j` is the jth column vector of A.

    Parameters
    ----------
    threshold : float
        Limit for the absolute difference between 1 and the overlap between the
        truncated and original wavefunction

    Returns
    -------
    slater_dets : list(M,)
        slater determinants (in integer form) in the truncated wavefunction
    error : float
        error associated with the truncation

    References
    ----------
     * http://arxiv.org/pdf/math/0508096v1.pdf

    """
    # assume wavefunction is converged
    # norms of each column
    # i'm guessing the generate_view gives the converged coefficients
    coeffs = np.hstack((np.identity(self.p), self.generate_view())) # this is the only part that is different
    col_norms = np.linalg.norm(coeffs, axis=0)
    # orbital indices sorted from largest to smallest norm
    orb_indices = np.argsort(col_norms)[::-1].tolist()
    # order column norms from largest to smallest norm
    col_norms = col_norms[orb_indices].tolist()

    # list of slater determinants
    sds = []

    # copied from combination code from itertools
    n = self.k
    r = self.p
    assert r <= n
    indices = range(r)

    upper_lim = np.prod([col_norms[j] for j in indices])
    assert upper_lim > val_threshold
    sds.append(slater.create_multiple_pairs(0, *(orb_indices[j] for j in indices)))
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return sds
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        upper_lim = np.prod([col_norms[j] for j in indices])
        if upper_lim > val_threshold:
            sds.append(slater.create_multiple_pairs(0, *(orb_indices[j] for j in indices)))
        else:
            #error += upper_lim of next one * num remaining in this sequence
            if i >= 1:
                indices[i-1] += 1
                for j in range(i, r):
                    indices[j] = indices[j-1] + 1
            else:
                return sds
