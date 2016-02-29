from __future__ import absolute_import, division, print_function

from itertools import permutations
import numpy as np
from ..utils import permanent
from ..utils import slater


def generate_guess(self):
    """
    Generate an appropriate random guess for the coefficient vector `x`.

    Returns
    -------
    x : 1-index np.ndarray

    """

    # Generate an empty array of the appropriate shape
    columns = 2 * self.k * self.seq - (self.seq ** 2 + self.seq) // 2
    x = np.zeros((self.p, columns), dtype=self.dtype)

    # Add random noise
    x[:, :] += (2.0e-1 / x.size) * (np.random.rand(self.p, columns) - 0.5)
    if self.dtype == np.complex128:
        x[:, :] += (2.0e-3 / x.size) * (np.random.rand(self.p, columns) - 0.5) * 1j

    # Normalize
    x[:, :] /= np.max(x)
    for i, j in enumerate(range(self.n // 2)):
        x[i, j] = 1.

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

    params = self.x.size
    ground = sum([2 ** i for i in range(self.n)])
    pspace = [ground]

    # Generate single, double, triple, etc., excitations
    for nexc in range(1, self.n + 1):
        for occ in permutations(list(range(self.n)), r=nexc): # HOMOs first
            for vir in permutations(list(range(self.n, 2 * self.k)), r=nexc):

                # Excite from the ground state
                new_sd = ground
                for i, a in zip(occ, vir):
                    new_sd = slater.excite(new_sd, i, a)

                # Check for sequential occupations
                sd = new_sd
                offset = 0
                columns = []
                for s in range(1, self.seq + 1):
                    for i in range(2 * self.k - s):
                        if slater.occupation(sd, i) and slater.occupation(sd, i + s):
                            columns.append(2 * self.k * (s - 1) + i + offset)
                            sd = slater.annihilate(sd, i)
                            sd = slater.annihilate(sd, i + s)
                    offset = -(s ** 2 + s) // 2

                # Add the SD to `pspace` if it satisfies the requirements
                if sd == 0 and new_sd is not None and new_sd not in pspace:
                    pspace.append(new_sd)

        # Break when we have enough SDs
        if len(pspace) >= params:
            break

    pspace.sort()
    return pspace


def generate_view(self):
    """
    Generate a view of `x` corresponding to the shape of the coefficient array.

    Returns
    -------
    view : 2-index np.ndarray

    """

    return self.x.reshape(self.p, self.x.size // self.p)


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


def hamiltonian_deriv(self, sd, x, y):
    """
    Compute the partial derivative of the Hamiltonian with respect to parameter `(x, y)`.

    Parameters
    ----------
    sd : int
        The Slater Determinant against which to project.
    x : int
    y : int

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
                one_electron += self.H[i // 2, k // 2] * self.overlap_deriv(single_excitation, x, y)
            # Avoid repetition by ensuring  `j > i` is satisfied
            for j in ind_occ[ind_first_occ + 1:]:
                # Add indices `i` and `j` to `ind_vir` because excitation to same
                # orbital is possible, and avoid repetition by ensuring `l > k` is
                # satisfied
                tmp_ind_vir_2 = sorted([j] + tmp_ind_vir_1[ind_first_vir + 1:])
                for l in tmp_ind_vir_2:
                    double_excitation = slater.excite(single_excitation, j, l)
                    olp = self.overlap_deriv(double_excitation, x, y)
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
    for i in range(self.C.shape[0]):
        for j in range(self.C.shape[1]):

            # Update changing variables
            d_olp = self.overlap_deriv(self.ground, i, j)
            d_energy = sum(self.hamiltonian_deriv(self.ground, i, j))

            # Impose d<HF|Psi> == 1 + 0j
            jac[-1, c] = d_olp

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

    # Impose C[0, 0] == 1 + 0j
    #obj[-2] = self.x[0] - 1.0

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

    # Evaluate the overlap
    columns = []
    offset = 0
    for s in range(1, self.seq + 1):
        for i in range(2 * self.k - s):
            if slater.occupation(sd, i) and slater.occupation(sd, i + s):
                columns.append(2 * self.k * (s - 1) + i + offset)
                sd = slater.annihilate(sd, i)
                sd = slater.annihilate(sd, i + s)
        offset = -(s ** 2 + s) // 2
    if sd != 0:
        return 0
    else:
        return permanent.dense(self.C[:, columns])


def overlap_deriv(self, sd, x, y):
    """
    Compute the partial derivative of the overlap of the wavefunction with `sd`.

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

    # Evaluate the overlap
    columns = []
    offset = 0
    for s in range(1, self.seq + 1):
        for i in range(2 * self.k - s):
            if slater.occupation(sd, i) and slater.occupation(sd, i + s):
                columns.append(2 * self.k * (s - 1) + i + offset)
                sd = slater.annihilate(sd, i)
                sd = slater.annihilate(sd, i + s)
        offset = -(s ** 2 + s) // 2
    if sd != 0:
        return 0
    elif y not in columns:
        return 0
    else:
        return permanent.dense_deriv(self.C[:, columns], x, columns.index(y))
