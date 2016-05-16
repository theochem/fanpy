from __future__ import absolute_import, division, print_function

from itertools import combinations, product

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

from .wavefunction import Wavefunction
from .math import binomial
from . import slater


class CIWavefunction(Wavefunction):
    """ Wavefunction expressed as a linear combination of Slater determinants

    Contains the necessary information to variationally solve the CI wavefunction

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

    Private
    -------
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
        # Arguments handled by FullCI class
        civec=None,
    ):

        super(CIWavefunction, self).__init__(
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
            nparticle=nparticle,
            odd_nelec=odd_nelec,
        )

        self.assign_civec(civec=civec)

    #
    # Solver methods
    #

    def _solve_eigh(self, which='LM', **kwargs):
        """ Solves for the ground state using eigenvalue decomposition
        """

        ci_matrix = self.compute_ci_matrix()
        result = eigsh(ci_matrix, 1, which=which, **kwargs)
        del ci_matrix

        self.C[...] = result[1][:, 0]
        self._energy[...] = result[0][0]

        return result

    #
    # Computation methods
    #

    # FIXME: turn abstract
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
        #FIXME: code repeated in proj_wavefunction.py

        nspin = self.nspin
        nelec = self.nelec

        civec = []

        # ASSUME: certain structure for civec
        # spin orbitals are shuffled (alpha1, beta1, alph2, beta2, etc)
        # spin orbitals are ordered by energy
        ground = slater.ground(nelec, nspin)
        civec.append(ground)
        # FIXME: need to reorder occ_indices to prioritize certain excitations
        occ_indices = slater.occ_indices(ground)
        vir_indices = slater.vir_indices(ground, nspin)

        count = 1
        for nexc in range(1, nelec + 1):
            occ_combinations = combinations(occ_indices, nexc)
            vir_combinations = combinations(vir_indices, nexc)
            for occ, vir in product(occ_combinations, vir_combinations):
                sd = slater.annihilate(ground, *occ)
                sd = slater.create(ground, *vir)
                civec.append(sd)
                count += 1
                if count == self.nproj:
                    return civec
        else:
            return civec

    # FIXME: turn abstract
    def compute_ci_matrix(self):
        """ Returns Hamiltonian matrix in the Slater determinant basis

        ..math::
            H_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

        Returns
        -------
        matrix : np.ndarray(K, K)
        """

        # TODO: Make an _unrestricted (and maybe a DOCI-specific) version


        nproj = self.nproj
        nspatial = self.nspatial
        H = self.H
        G = self.G
        civec = self.civec

        def is_alpha(i):
            """ Checks if index `i` belongs to an alpha spin orbital

            Parameter
            ---------
            i : int
                Index of the spin orbital in the Slater determinant

            """
            if i < nspatial:
                return True
            else:
                return False

        def spatial_index(i):
            """ Returns the index of the spatial orbital that corresponds to the
            spin orbital `i`

            Parameter
            ---------
            i : int
                Index of the spin orbital in the Slater determinant
            """
            if is_alpha(i):
                return i
            else:
                return i-nspatial


        ci_matrix = np.zeros((nproj, nproj), dtype=self.dtype)

        # Loop only over upper triangular
        for nsd0, sd0 in enumerate(self.civec):
            for nsd1, sd1 in enumerate(self.civec[nsd0:]):
            #for nsd1, sd1 in enumerate(self.civec):
                diff_sd0, diff_sd1 = slater.diff(sd0, sd1)
                shared_indices = slater.occ_indices(slater.shared(sd0, sd1))

                if len(diff_sd0) != len(diff_sd1):
                    continue
                else:
                    diff_order = len(diff_sd0)

                # two sd's are different by double excitation
                if diff_order == 2:
                    i, j = diff_sd0
                    k, l = diff_sd1
                    I = spatial_index(i)
                    J = spatial_index(j)
                    K = spatial_index(k)
                    L = spatial_index(l)
                    if is_alpha(i) == is_alpha(k) and is_alpha(j) == is_alpha(l):
                        ci_matrix[nsd0, nsd1] += G[I, J, K, L]
                    if i % 2 == l % 2 and j % 2 == k % 2:
                        ci_matrix[nsd0, nsd1] -= G[I, J, L, K]

                # two sd's are different by single excitation
                elif diff_order == 1:
                    i, = diff_sd0
                    k, = diff_sd1
                    I = spatial_index(i)
                    K = spatial_index(k)
                    # if the spins match
                    if i % 2 == k % 2:
                        ci_matrix[nsd0, nsd1] += H[I, K]
                        for j in shared_indices:
                            if j != i:
                                if i % 2 == k % 2:
                                    J = spatial_index(j)
                                    ci_matrix[nsd0, nsd1] += G[I, J, K, J]
                                    if i % 2 == j % 2:
                                        ci_matrix[nsd0, nsd1] -= G[I, J, J, K]

                # two sd's are the same
                elif diff_order == 0:
                    for ic, i in enumerate(shared_indices):
                        I = spatial_index(i)
                        ci_matrix[nsd0, nsd1] += H[I, I]
                        for j in shared_indices[ic+1:]:
                            J = spatial_index(j)
                            ci_matrix[nsd0, nsd1] += G[I, J, I, J]
                            if i % 2 == j % 2:
                                ci_matrix[nsd0, nsd1] -= G[I, J, J, I]

        # Make it Hermitian
        ci_matrix[:, :] = np.triu(ci_matrix) + np.tril(ci_matrix, -1)

        return ci_matrix
