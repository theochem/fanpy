from __future__ import absolute_import, division, print_function

from itertools import combinations
import numpy as np
from scipy.misc import comb
from ..utils import slater



def __init__(self, n, H, G, dtype=None, pspace=None, x=None):
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
    dtype : {np.float64, np.complex128}
    pspace : list, optional
        List of Python ints representing Slater determinants that make up the
        N electron basis set
        If not specified, the Slater determinants for the CISD (all single and
        double excitations) are used
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
    self.pspace = self.generate_pspace() if pspace is None else pspace
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
    num_sds = len(self.pspace)
    # Generate an empty array of the appropriate shape
    x = np.zeros(num_sds, dtype=self.dtype)
    x[0] += 1.0

    # Add random noise
    x += (0.2 / x.size) * (np.random.rand(num_sds) - 0.5)
    if self.dtype == np.complex128:
        x += (0.2 / x.size) * (np.random.rand(num_sds) - 0.5) * 1j

    # Normalize
    x /= np.max(x)

    return x


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

    # Get spin indices
    ind_occ = [i for i in range(2 * self.k) if slater.occupation(sd, i)]
    ind_vir = [i for i in range(2 * self.k) if not slater.occupation(sd, i)]
    one_electron = 0.0
    coulomb = 0.0
    exchange = 0.0
    ind_first_occ = 0

    for i in ind_occ:
        ind_first_vir = 0
        # Add `i` to `ind_vir` because excitation to same orbital is possible
        tmp_ind_vir_1 = sorted(ind_vir + [i])

        for k in tmp_ind_vir_1:
            single_excitation = slater.excite(sd, i, k)
            if i % 2 == k % 2:
                one_electron += self.H[i // 2, k // 2] * self.overlap(single_excitation)
            # Avoid repetition by ensuring  `j > i` is satisfied
            for j in ind_occ[ind_first_occ + 1:]:
                # Add indices `i` and `j` to `ind_vir` because excitation to same
                # orbital is possible, and avoid repetition by ensuring `l > k` is
                # satisfied
                tmp_ind_vir_2 = sorted([j] + tmp_ind_vir_1[ind_first_vir + 1:])
                for l in tmp_ind_vir_2:
                    double_excitation = slater.excite(single_excitation, j, l)
                    olp = self.overlap(double_excitation)
                    if olp == 0:
                        continue
                    # In <ij|kl>, `i` and `k` must have the same spin, as must
                    # `j` and `l`
                    if i % 2 == k % 2 and j % 2 == l % 2:
                        coulomb += self.G[i // 2, j // 2, k // 2, l // 2] * olp
                    # In <ij|lk>, `i` and `l` must have the same spin, as must
                    # `j` and `k`
                    if i % 2 == l % 2 and j % 2 == k % 2:
                        exchange -= self.G[i // 2, j // 2, l // 2, k // 2] * olp

            ind_first_vir += 1
        ind_first_occ += 1

    return one_electron, coulomb, exchange


def hamiltonian_deriv(self, sd, index):
    """
    Compute the partial derivative of the Hamiltonian with respect to parameter `x`.

    Parameters
    ----------
    sd : int
        The Slater Determinant against which to project.
    index : int
        Index of the parameter with which we are derivatizing

    Returns
    -------
    energy : tuple
        Tuple of floats corresponding to the one electron, coulomb, and exchange
        energies.

    """

    # Get spin indices
    ind_occ = [i for i in range(2 * self.k) if slater.occupation(sd, i)]
    ind_vir = [i for i in range(2 * self.k) if not slater.occupation(sd, i)]
    one_electron = 0.0
    coulomb = 0.0
    exchange = 0.0
    ind_first_occ = 0

    for i in ind_occ:
        ind_first_vir = 0
        # Add `i` to `ind_vir` because excitation to same orbital is possible
        tmp_ind_vir_1 = sorted(ind_vir + [i])

        for k in tmp_ind_vir_1:
            single_excitation = slater.excite(sd, i, k)
            if i % 2 == k % 2:
                one_electron += self.H[i // 2, k // 2] * self.overlap_deriv(single_excitation, index)
            # Avoid repetition by ensuring  `j > i` is satisfied
            for j in ind_occ[ind_first_occ + 1:]:
                # Add indices `i` and `j` to `ind_vir` because excitation to same
                # orbital is possible, and avoid repetition by ensuring `l > k` is
                # satisfied
                tmp_ind_vir_2 = sorted([j] + tmp_ind_vir_1[ind_first_vir + 1:])
                for l in tmp_ind_vir_2:
                    double_excitation = slater.excite(single_excitation, j, l)
                    olp = self.overlap_deriv(double_excitation, index)
                    if olp == 0:
                        continue
                    # In <ij|kl>, `i` and `k` must have the same spin, as must
                    # `j` and `l`
                    if i % 2 == k % 2 and j % 2 == l % 2:
                        coulomb += self.G[i // 2, j // 2, k // 2, l // 2] * olp
                    # In <ij|lk>, `i` and `l` must have the same spin, as must
                    # `j` and `k`
                    if i % 2 == l % 2 and j % 2 == k % 2:
                        exchange -= self.G[i // 2, j // 2, l // 2, k // 2] * olp

            ind_first_vir += 1
        ind_first_occ += 1

    return one_electron, coulomb, exchange


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
    jac = np.empty((len(self.pspace) + 1, self.x.size), dtype=x.dtype)

    # Intialize unchanging variables
    energy = sum(self.hamiltonian(self.ground))

    # Loop through all coefficients
    c = 0
    for i in range(self.x.size):
        # Update changing variables
        d_olp = self.overlap_deriv(self.ground, i)
        d_energy = sum(self.hamiltonian_deriv(self.ground, i))

        # Impose d<HF|Psi> == 1 + 0j
        jac[-1, c] = d_olp

        # Impose (for all SDs in `pspace`) <SD|H|Psi> - E<SD|H|Psi> == 0
        for k, sd in enumerate(self.pspace):
            d_tmp = sum(self.hamiltonian_deriv(sd, i)) \
                - energy * self.overlap_deriv(sd, i) - d_energy * self.overlap(sd)
            jac[k, c] = d_tmp

        # Move to the next coefficient
        c += 1

    return jac


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
    obj = np.empty(len(self.pspace) + 1, dtype=x.dtype)

    # Impose <HF|Psi> == 1
    obj[-1] = olp - 1.0

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
    # else
    try:
        return self.x[self.pspace.index(sd)]
    except ValueError:
        return 0


def overlap_deriv(self, sd, i):
    """
    Compute the partial derivative of the overlap of the wavefunction with `sd`.

    Parameters
    ----------
    sd : int
        The Slater Determinant against which to project.

    """

    if 0 <= i < self.x.size and sd == self.pspace[i]:
        return 1
    else:
        return 0
    
