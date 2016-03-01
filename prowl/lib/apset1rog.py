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

    # Generate empty arrays (C and B) of the appropriate shape
    b = np.zeros(self.k ** 2, dtype=self.dtype)
    c = np.zeros(self.p * (2 * self.k - self.p), dtype=self.dtype)

    for vec in (b, c):
        # Add random noise
        vec[:] += (2.0e-1 / vec.size) * (np.random.rand(vec.size) - 0.5)
        if self.dtype == np.complex128:
            vec[:] += (2.0e-3 / vec.size) * (np.random.rand(vec.size) - 0.5) * 1j
        # Normalize
        #vec[:] /= np.max(vec)

    # Construct x
    x = np.empty(b.size + c.size, dtype=self.dtype)
    x[:b.size] = b
    x[b.size:] = c

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

    # Generate single and double excitations
    for nexc in (1, 2):
        perms = permutations(list(range(self.p)), r=nexc)
        for occ in perms:
            for i in occ:
                tmp0 = slater.annihilate_pair(ground, i)
                for a in range(self.p, self.k):
                    tmp1 = slater.create(tmp0, 2 * a)
                    for b in range(self.p, self.k):
                        sd = slater.create(tmp1, 2 * b + 1)
    
                        # Add the SD to `pspace` if it satisfies the requirements
                        if sd is not None and sd not in pspace:
                            pspace.append(sd)

                            # Break when we have enough SDs
                            if len(pspace) >= params:
                                pass
                                #pspace.sort()
                                #return pspace
    pspace.sort()
    return pspace


def generate_view(self):
    """
    Generate a view of `x` corresponding to the shape of the coefficient array.

    Returns
    -------
    b : 2-index np.ndarray
    c : 2-index np.ndarray

    """

    split = self.k ** 2
    b = self.x[:split].reshape(self.k, self.k)
    c = self.x[split:].reshape(self.p, 2 * self.k - self.p)
    return b, c


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
    obj = np.empty(len(self.pspace) + 0, dtype=x.dtype)

    # Impose <HF|Psi> == 1
    #obj[-1] = olp - 1.0

    # Impose C[0, 0] == 1 + 0j
    #obj[-2] = self.C[0][0, 0] - 1.0

    # Impose (for all SDs in `pspace`) <SD|H|Psi> - E<SD|H|Psi> == 0
    for i, sd in enumerate(self.pspace):
        obj[i] = sum(self.hamiltonian(sd)) - energy * self.overlap(sd)

    print(np.sum(obj ** 2))
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
    # If the 1-ro form is broken
    #if any([slater.occupation(sd, 2 * i) != slater.occupation(sd, 2 * i + 1) for i in range(self.p)]):
        #return 0
    # If the electrons are not split evenly between the two sets
    else:
        pass
        a_count = 0
        for i in range(0, 2 * self.k + 1, 2):
            if slater.occupation(sd, i):
                a_count += 1
        b_count = 0
        for j in range(1, 2 * self.k + 1, 2):
            if slater.occupation(sd, j):
                b_count += 1
        if a_count != b_count:
            return 0

    # Evaluate the overlap
    b_set0 = []
    b_set1 = []
    c_occ = []
    # If the SD is the ground state
    if sd == self.ground:
        olp_perm = 1
        olp_det = np.linalg.det(self.C[0][:self.p, :self.p])
    # Determine which base coefficients contribute to the overlap
    else:
        b_set0 = [i for i in range(self.k) if slater.occupation(sd, 2 * i)]
        b_set1 = [i for i in range(self.k) if slater.occupation(sd, 2 * i + 1)]
        # Loop through occupieds
        for i in range(self.k):
            i_occ = slater.occupation(sd, 2 * i)
            j_occ = slater.occupation(sd, 2 * i + 1)
            if i_occ and j_occ:
                c_occ.append(i)
            elif i_occ != j_occ:
                c_occ.append(i + self.k)
        aug_C = np.empty((self.p, 2 * self.k), dtype=self.dtype)
        aug_C[:, :self.p] = np.eye(self.p)
        aug_C[:, self.p:] = self.C[1]
        # Compute the permanent with the appropriate rows and columns
        #print(b_set0, b_set1, c_occ)
        olp_det = np.linalg.det(self.C[0][b_set0][:, b_set1])
        olp_perm = permanent.dense(aug_C[:, c_occ])
    # Return
    return olp_det * olp_perm

    # Evaluate the overlap
    #print("SD: {}".format(bin(sd)))
    #return b_det * c_perm
    #print( self.C[0][b_set0][:, b_set1].shape, self.C[1][c_occ][:, c_vir].shape )
    #return np.linalg.det(self.C[0][b_set0][:, b_set1]) * permanent.dense(self.C[1][c_occ][:, c_vir])


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
