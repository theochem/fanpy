from __future__ import absolute_import, division, print_function

from itertools import combinations, product

import numpy as np
from scipy.linalg import eigh

from .wavefunction import Wavefunction
from .math import binomial


class FullCI(Wavefunction):
    """ FCI Wavefunction class

    Contains the necessary information to projectively (and variationally) solve the wavefunction

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : np.ndarray(K,K)
        One electron integrals for the spatial orbitals
    Ha : np.ndarray(K,K)
        One electron integrals for the alpha spin orbitals
    Hb : np.ndarray(K,K)
        One electron integrals for the beta spin orbitals
    G : np.ndarray(K,K,K,K)
        Two electron integrals for the spatial orbitals
    Ga : np.ndarray(K,K,K,K)
        Two electron integrals for the alpha spin orbitals
    Gb : np.ndarray(K,K,K,K)
        Two electron integrals for the beta spin orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nspatial : int
        Number of spatial orbitals
    nspin : int
        Number of spin orbitals (alpha and beta)
    nelec : int
        Number of electrons
    npair : int
        Number of electron pairs
        Assumes that the number of electrons is even
    nparticle : int
        Number of quasiparticles (electrons)
    ngeminal : int
        Number of geminals
    nproj : int
        Dimension of the projection space
    nparam : int
        Number of parameters
    x : np.ndarray(K)
        Guess for the parameters
        Iteratively updated during convergence
        Initial guess before convergence
        Coefficients after convergence
    C : np.ndarray(K)
        Coefficient of the Slater determinants

    Private
    -------
    _nproj_default : int
        Default dimension of projection space
    _methods : dict
        Default dimension of projection space
    _energy : float
        Electronic energy
    """
    # FIXME: turn C into property and have a better attribute name

    #
    # Default attribute values
    #

    @property
    def _nproj_default(self):

        return binomial(self.nspin, self.nelec)

    @property
    def _methods(self):

        return {"default": self._solve_eigh}

    #
    # Special methods
    #

    def __init__(
        self,
        # Mandatory arguments
        nelec=None,
        H=None,
        G=None,
        # Arguments handled by base Wavefunction class
        dtype=None,
        nuc_nuc=None,
        nparticle=None,
        odd_nelec=None,
        nproj=None,
        naproj=None,
        nrproj=None,
        # Arguments handled by FullCI class
        x=None,
        civec=None,
    ):

        super(FullCI, self).__init__(
            nelec=nelec,
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
            nparticle=nparticle,
            odd_nelec=odd_nelec,
            nproj=nproj,
            naproj=naproj,
            nrproj=nrproj,
        )

        self.assign_x(x=x)
        self.assign_civec(civec=civec)
        self.assign_Hvec()
        self.assign_Gvec()
        self.assign_cache()

    def __call__(self,  method="default", **kwargs):
        """ Optimize coefficients

        Unless _methods is changed, only _solve_eigh will be available
        """
        # FIXME: move to wavefunction?

        methods = self._methods
        if method in methods:
            method = methods[method.lower()]
            return method(**kwargs)
        else:
            raise ValueError("method must be one of {0}".format(methods.keys()))

    #
    # Solver methods
    #

    def _solve_eigh(self, **kwargs):
        """ Solves for the ground state using eigenvalue decomposition
        """

        ci_matrix = self.compute_ci_matrix()
        result = eigh(ci_matrix, **kwargs)

        self.C[:] = result[1][:, 0]
        self._energy[:] = result[0][0]

        del ci_matrix
        return result

    #
    # Assignment methods
    #

    def assign_x(self, x=None):

        nparam = self._nproj_default

        if x is not None:
            if not isinstance(x, np.ndarray):
                raise TypeError("x must be of type {0}".format(np.ndarray))
            elif x.shape != (nparam + 1,):
                raise ValueError("x must be of length nparam + 1 ({0})".format(nparam + 1))
            elif x.dtype not in (float, complex, np.float64, np.complex128):
                raise TypeError("x's dtype must be one of {0}".format((float, complex, np.float64, np.complex128)))

        else:
            x = np.empty(nparam + 1, dtype=self.dtype)
            x[:-1] = 2.0 * (np.random.random(nparam) - 0.5) / (nparam ** 2)
            if x.dtype == np.complex128:
                x[:-1] = 2.0j * (np.random.random(nparam) - 0.5) / (nparam ** 2)

        C = x[:-1]
        C[0] = 1.0 # Intermediate normalization
        energy = x[-1, ...] # Use the Ellipsis to obtain a view

        self.nparam = nparam
        self.x = x
        self.C = C
        self._energy = energy

    def assign_civec(self, civec=None):

        if civec is not None:
            if not isinstance(civec, np.ndarray):
                raise TypeError("civec must be of type {0}".format(np.ndarray))
            elif civec.shape != (self.nproj, self.nspin):
                raise ValueError("civec must be of shape (nproj, nspin) ({0})".format((self.nproj, self.nspin)))

        else:
            civec = self.compute_civec()

        self.civec = civec
        self.civeca = civec[:, 0::2]
        self.civecb = civec[:, 1::2]

    def assign_Hvec(self):

        self.Hvec, self.Hvec_values, self.Hvec_shape = self.compute_Hvec()
        self.Hveca = self.Hvec[0::2]
        self.Hvecb = self.Hvec[1::2]

    def assign_Gvec(self):

        self.Gvec, self.Gvec_values, self.Gvec_shape = self.compute_Gvec()
        self.Gveca = self.Gvec[0::2]
        self.Gvecb = self.Gvec[1::2]

    def assign_cache(self):

        self.civec_index, cache, _ = self.compute_vec_index(self.civec)
        self.Hvec_index, _, _ = self.compute_vec_index(self.Hvec, cache=cache, length=self.Hvec_shape)
        self.Gvec_index, _, unique = self.compute_vec_index(self.Gvec, cache=cache, length=self.Gvec_shape)
        del cache
        #self.cache, self.cache_status, self.dcache, self.dcache_status = self.compute_cache(unique)

    #
    # Computation methods
    #

    def compute_civec(self):
        """ Generates Slater determinants

        Number of Slater determinants generated is determined strictly by the size of the
        projection space (self.nproj). First row corresponds to the ground state SD, then
        the next few rows are the first excited, then the second excited, etc

        Returns
        -------
        civec : np.ndarray(nproj, nspin)
            Boolean array that describes the occupations of the Slater determinants
            Each row is a Slater determinant
            Each column is the index of the spin orbital

        """

        nproj = self.nproj
        nspin = self.nspin
        nelec = self.nelec

        # ASSUME: certain structure for civec
        # spin orbitals are shuffled (alpha1, beta1, alph2, beta2, etc)
        # spin orbitals are ordered by energy
        civec = np.zeros((nproj, nspin), dtype=bool)
        civec[:, :nelec] = True

        count = 1
        for nexc in range(1, self.nelec + 1):
            occ_combinations = combinations(reversed(range(nelec)), nexc)
            vir_combinations = combinations(range(nelec, nspin), nexc)
            for occ, vir in product(occ_cobinations, vir_combinations):
                civec[count, occ] = False
                civec[count, vir] = True
                count += 1
                if count == nproj:
                    return civec
        else:
            return civec

    def compute_Hvec(self):

        civec = self.civec
        Ha = self.Ha
        Hb = self.Hb
        nproj = self.nproj
        nspin = self.nspin
        nelec = self.nelec
        orb = np.arange(nspin)
        maxcount = 0
        shape = []

        Hvec = np.empty((nproj, np.max((nelec * (nspin - nelec), 2 * nspin)), nspin), dtype=bool)
        Hvec_values = np.empty(Hvec.shape[:2], dtype=self.dtype)

        for sd in range(nproj):
            count = 0
            occ = orb[civec[sd, :]]

            for i in occ:
                vir = orb[np.bitwise_not(civec[sd, :])].tolist()
                vir.append(i)
                vir.sort()

                for k in vir:
                    Hvec[sd, count, :] = civec[sd, :]
                    Hvec[sd, count, (i, k)] = (False, True)
                    if np.sum(Hvec[sd, count, :]) != nelec:
                        continue

                    if i % 2 == 0:
                        H = Ha
                    else:
                        H = Hb

                    if i % 2 == k % 2:
                        Hvec_values[sd, count] = H[i // 2, k // 2]
                        count += 1

            shape.append(count)

        Hvec_shape = np.array(shape)
        Hvec = Hvec[:, :np.max(Hvec_shape), :]
        Hvec_values = Hvec_values[:, :Hvec.shape[1]]

        return Hvec, Hvec_values, Hvec_shape

    def compute_Gvec(self):

        civec = self.civec
        Ga = self.Ga
        Gb = self.Gb
        nspatial = self.nspatial
        orb = np.arange(self.nspin)
        maxcount = 0
        shape = []

        Gvec = np.empty((self.nproj, 10 * self.nproj ** 2, self.nspin), dtype=bool)
        Gvec_values = np.zeros(Gvec.shape[:2], dtype=self.dtype)

        for sd in range(self.nproj):
            count = 0
            occ = orb[civec[sd, :]]

            for ic, i in enumerate(occ):
                vir0 = orb[np.bitwise_not(civec[sd, :])].tolist()
                vir0.append(i)
                vir0.sort()

                for kc, k in enumerate(vir0):

                    for j in occ[ic:]:
                        vir1 = list(vir0[kc:])
                        vir1.append(j)
                        vir1.sort()

                        for l in vir1:
                            Gvec[sd, count, :] = civec[sd, :]
                            Gvec[sd, count, [i, k]] = (False, True)
                            if np.sum(Gvec[sd, count, :]) != self.nelec:
                                continue
                            Gvec[sd, count, [j, l]] = (False, True)
                            if np.sum(Gvec[sd, count, :]) != self.nelec:
                                continue

                            if i % 2 == 0:
                                Gik = Ga
                            else:
                                Gik = Gb

                            if j % 2 == 0:
                                Gjk = Ga
                            else:
                                Gjk = Gb

                            flag = False
                            if i % 2 == k % 2 and j % 2 == l % 2:
                                Gvec_values[sd, count] += Gik[i // 2, j // 2, k // 2, l // 2]
                                flag = True
                            if i % 2 == l % 2 and j % 2 == k % 2:
                                Gvec_values[sd, count] -= Gjk[i // 2, j // 2, l // 2, k // 2]
                                flag = True
                            if flag:
                                count += 1

            shape.append(count)

        Gvec_shape = np.array(shape)
        Gvec = Gvec[:, :np.max(Gvec_shape), :]
        Gvec_values = Gvec_values[:, :Gvec.shape[1]]

        return Gvec, Gvec_values, Gvec_shape

    def compute_vec_index(self, vec, cache=None, length=None):

        if cache is None:
            cache = {}
            unique = 0
        else:
            unique = np.max(cache.values())

        if length is None:
            if vec.ndim == 3:
                length = vec.shape[:-2]
            elif vec.ndim == 2:
                length = vec.shape[0]

        if vec.ndim == 3:
            index = np.empty(vec.shape[:-1], dtype=int)
            for i in range(vec.shape[0]):
                last = length[i]
                for j in range(vec.shape[1]):
                    if j == last:
                        break
                    key = tuple(vec[i, j, :])
                    if key not in cache:
                        cache[key] = unique
                        index[i, j] = unique
                        unique += 1
                    else:
                        index[i, j] = cache[key]

        elif vec.ndim == 2:
            index = np.empty(vec.shape[0], dtype=int)
            for i in range(vec.shape[0]):
                if i == length:
                    break
                key = tuple(vec[i, :])
                if key not in cache:
                    cache[key] = unique
                    index[i] = unique
                    unique += 1
                else:
                    index[i] = cache[key]

        return index, cache, unique

    def compute_cache(self, unique):

        cache = np.empty(unique, dtype=self.dtype)
        cache_status = np.empty(unique, dtype=bool)
        dcache = np.empty((unique, self.nparam), dtype=self.dtype)
        dcache_status = np.empty((unique, self.nparam), dtype=bool)

        return cache, cache_status, dcache, dcache_status

    def compute_ci_matrix(self):

        # TODO: Make an _unrestricted (and maybe a DOCI-specific) version


        nproj = self.nproj
        nspatial = self.nspatial
        H = self.H
        G = self.G
        civec = self.civec
        orb = np.arange(self.nspin)

        ci_matrix = np.zeros((nproj, nproj), dtype=self.dtype)

        # Loop only over upper triangular
        for sd0 in range(nproj):
            for sd1 in range(nproj):

                diff = np.bitwise_xor(civec[sd0, :], civec[sd1, :])
                diff_sd0 = np.bitwise_and(civec[sd0, :], diff)
                diff_sd1 = np.bitwise_and(civec[sd1, :], diff)

                diff = orb[diff]
                diff_sd0 = orb[diff_sd0]
                diff_sd1 = orb[diff_sd1]

                if diff.size > 4 or diff.size % 2 or diff_sd0.size != diff_sd1.size:
                    continue

                if diff.size == 4:
                    i, j = diff_sd0.tolist()
                    k, l = diff_sd1.tolist()
                    if i % 2 == k % 2 and j % 2 == l % 2:
                        I, J, K, L = i // 2, j // 2, k // 2, l // 2
                        ci_matrix[sd0, sd1] += G[I, J, K, L]
                    if i % 2 == l % 2 and j % 2 == k % 2:
                        I, J, K, L = i // 2, j // 2, k // 2, l // 2
                        ci_matrix[sd0, sd1] -= G[I, J, L, K]

                elif diff.size == 2:
                    i, k = np.asscalar(diff_sd0), np.asscalar(diff_sd1)
                    if i % nspatial == k % nspatial:
                        I, K = i // 2, k // 2
                        ci_matrix[sd0, sd1] += H[I, K]
                        for j in orb[civec[sd0, :]]:
                            if j != i:
                                if i % 2 == k % 2:
                                    J = j // 2
                                    ci_matrix[sd0, sd1] += G[I, J, K, J]
                                    if i % 2 == j % 2:
                                        ci_matrix[sd0, sd1] -= G[I, J, J, K]

                else:
                    occ = orb[civec[sd0, :]]
                    for ic, i in enumerate(occ):
                        I = i // 2
                        ci_matrix[sd0, sd1] += self.H[I, I]
                        for j in occ[(ic + 1):]:
                            J = j // 2
                            ci_matrix[sd0, sd1] += self.G[I, J, I, J]
                            if i % 2 == j % 2:
                                ci_matrix[sd0, sd1] -= self.G[I, J, J, I]

        # Make it Hermitian
        ci_matrix[:, :] = np.triu(ci_matrix) + np.tril(ci_matrix, -1)

        return ci_matrix

    def compute_projection(self, sd=0, vec=0, deriv=None):

        if vec == 0:
            index = self.civec_index[sd]
        elif vec == 1:
            index = self.Hvec_index[sd]
        elif vec == 2:
            index = self.Gvec_index[sd]

        if deriv is None:
            return self.C[index]
        elif deriv == index:
            return 1.0
        else:
            return 0.0

    def compute_energy(self, sd=None, nuc_nuc=True, deriv=None):

        nuc_nuc = self.nuc_nuc if nuc_nuc else 0.0

        if sd is None:
            return np.asscalar(self._energy) + nuc_nuc

        Hvec_shape = self.Hvec_shape[sd]
        Hvec_values = self.Hvec_values[sd]
        Hvec_projections = np.empty_like(Hvec_values)

        Gvec_shape = self.Gvec_shape[sd]
        Gvec_values = self.Gvec_values[sd]
        Gvec_projections = np.empty_like(Gvec_values)

        for i in range(Hvec_shape):
            Hvec_projections[i] = self.compute_projection((sd, i), vec=1, deriv=deriv)

        for i in range(Gvec_shape):
            Gvec_projections[i] = self.compute_projection((sd, i), vec=2, deriv=deriv)

        return np.dot(Hvec_projections, Hvec_values) \
             + np.dot(Gvec_projections, Gvec_values) \
             + nuc_nuc
