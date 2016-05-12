from __future__ import absolute_import, division, print_function

from itertools import combinations, product

import numpy as np
import gmpy2
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
        del ci_matrix

        self.C[...] = result[1][:, 0]
        self._energy[...] = result[0][0]

        return result

    #
    # Assignment methods
    #

    def assign_x(self, x=None):

        nparam = self._nproj_default + 1

        if x is not None:
            if not isinstance(x, np.ndarray):
                raise TypeError("x must be of type {0}".format(np.ndarray))
            elif x.shape != (nparam,):
                raise ValueError("x must be of length nparam ({0})".format(nparam))
            elif x.dtype not in (float, complex, np.float64, np.complex128):
                raise TypeError("x's dtype must be one of {0}".format((float, complex, np.float64, np.complex128)))

        else:
            x = np.empty(nparam, dtype=self.dtype)
            x[:-1] = 2.0 * (np.random.random(nparam - 1) - 0.5) / (nparam ** 2)
            if x.dtype == np.complex128:
                x[:-1] = 2.0j * (np.random.random(nparam - 1) - 0.5) / (nparam ** 2)

        C = x[:-1]
        C[0] = 1.0 # Intermediate normalization
        energy = x[-1, ...] # Use the Ellipsis to obtain a view

        self.nparam = nparam
        self.x = x
        self.C = C
        self._energy = energy

    def assign_civec(self, civec=None):

        if civec is not None:
            if not isinstance(civec, list):
                raise TypeError("civec must be of type {0}".format(list))
        else:
            civec = self.compute_civec()

        self.civec = civec
        self.cache = {}
        self.d_cache = {}

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

        nspin = self.nspin
        nelec = self.nelec

        civec = []

        # ASSUME: certain structure for civec
        # spin orbitals are shuffled (alpha1, beta1, alph2, beta2, etc)
        # spin orbitals are ordered by energy
        ground = gmpy2.mpz(0)
        for i in range(nelec):
            # Add electrons
            ground |= 1 << i
        civec.append(ground)

        count = 1
        for nexc in range(1, nelec + 1):
            occ_combinations = combinations(reversed(range(nelec)), nexc)
            vir_combinations = combinations(range(nelec, nspin), nexc)
            for occ, vir in product(occ_combinations, vir_combinations):
                sd = ground
                for i in occ:
                    # Remove electrons
                    sd &= ~(1 << i)
                for a in vir:
                    # Add electrons
                    sd |= 1 << a
                civec.append(sd)
                count += 1
                if count == self.nproj:
                    return civec
        else:
            return civec

    def compute_ci_matrix(self):

        # TODO: Make an _unrestricted (and maybe a DOCI-specific) version


        nproj = self.nproj
        nspatial = self.nspatial
        H = self.H
        G = self.G
        civec = self.civec

        ci_matrix = np.zeros((nproj, nproj), dtype=self.dtype)

        # Loop only over upper triangular
        for nsd0, sd0 in enumerate(self.civec):
            #for nsd1, sd1 in enumerate(self.civec[(nsd0):]):
            for nsd1, sd1 in enumerate(self.civec):

                mask = sd0 ^ sd1
                mask_sd0 = sd0 & mask
                mask_sd1 = sd1 & mask

                diff_sd0 = []
                bit = -1
                while True:
                    bit = gmpy2.bit_scan1(mask_sd0, bit + 1)
                    if bit is None: break
                    else: diff_sd0.append(bit)
                diff_sd1 = []
                bit = -1
                while True:
                    bit = gmpy2.bit_scan1(mask_sd1, bit + 1)
                    if bit is None: break
                    else: diff_sd1.append(bit)
                   
                if gmpy2.popcount(mask_sd0) != gmpy2.popcount(mask_sd1):
                    continue
                else:
                    diffcount = gmpy2.popcount(mask)

                if diffcount == 4:
                    i, j = diff_sd0
                    k, l = diff_sd1
                    if i % 2 == k % 2 and j % 2 == l % 2:
                        I, J, K, L = i // 2, j // 2, k // 2, l // 2
                        ci_matrix[nsd0, nsd1] += G[I, J, K, L]
                    if i % 2 == l % 2 and j % 2 == k % 2:
                        I, J, K, L = i // 2, j // 2, k // 2, l // 2
                        ci_matrix[nsd0, nsd1] -= G[I, J, L, K]

                elif diffcount == 2:
                    occ = []
                    bit = -1
                    while True:
                        bit = gmpy2.bit_scan1(mask_sd0, bit + 1)
                        if bit is None: break
                        else: occ.append(bit)
                    i, k = diff_sd0[0], diff_sd1[0]
                    if i % nspatial == k % nspatial:
                        I, K = i // 2, k // 2
                        ci_matrix[nsd0, nsd1] += H[I, K]
                        for j in occ:
                            if j != i:
                                if i % 2 == k % 2:
                                    J = j // 2
                                    ci_matrix[nsd0, nsd1] += G[I, J, K, J]
                                    if i % 2 == j % 2:
                                        ci_matrix[nsd0, nsd1] -= G[I, J, J, K]

                elif diffcount == 0:
                    occ = []
                    bit = -1
                    while True:
                        bit = gmpy2.bit_scan1(mask_sd0, bit + 1)
                        if bit is None: break
                        else: occ.append(bit)
                    for ic, i in enumerate(occ):
                        I = i // 2
                        ci_matrix[nsd0, nsd1] += H[I, I]
                        for j in occ[(ic + 1):]:
                            J = j // 2
                            ci_matrix[nsd0, nsd1] += G[I, J, I, J]
                            if i % 2 == j % 2:
                                ci_matrix[nsd0, nsd1] -= G[I, J, J, I]

        # Make it Hermitian
        ci_matrix[:, :] = np.triu(ci_matrix) + np.tril(ci_matrix, -1)

        return ci_matrix

    def compute_projection(self, sd, deriv=None):

        if deriv is None:
            return self.cache.get(sd, self.C[self.civec.index(sd)])
        else:
            return self.d_cache.get(sd, self._compute_projection_deriv(sd, deriv))
                    
    def _compute_projection_deriv(self, sd, deriv):

        return 1.0 if deriv == self.civec.index(sd) else 0.0

    def compute_energy(self, sd=None, nuc_nuc=True, deriv=None):

        nuc_nuc = self.nuc_nuc if nuc_nuc else 0.0

        if sd is None:
            return np.asscalar(self._energy) + nuc_nuc

        else:
            # TODO: ADD HAMILTONIANS
            raise NotImplementedError

