from __future__ import absolute_import, division, print_function

import numpy as np
from gmpy2 import mpz

from .proj_wavefunction import ProjectionWavefunction
from .. import slater
from ..sd_list import doci_sd_list
from .proj_hamiltonian import doci_hamiltonian
from ..math_tools import permanent_borchardt, adjugate
from .ap1rog import AP1roG


class APr2G(ProjectionWavefunction):
    """ Antisymmetric Product of rank-2 Geminals

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : tuple of np.ndarray(K,K)
        One electron integrals for the spatial orbitals
    G : tuple of np.ndarray(K,K,K,K)
        Two electron integrals for the spatial orbitals
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
    _nci : int
        Number of Slater determinants

    Methods
    -------
    """


    @property
    def template_params(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to this template
        The parameters are assumed to be ordered as [lambda_1, ... , lambda_p, epsilon_1, ... , epsilon_k, xi_1, ... , xi_k]

        Returns
        -------
        template_params : np.ndarray(K, )

        """
        ap1rog = AP1roG(nelec=self.nelec, H=self.H, G=self.G, nuc_nuc=self.nuc_nuc)
        ap1rog()
        gem_coeffs = np.hstack((np.eye(self.npair), np.reshape(ap1rog.params, (self.npair, self.nspatial - self.npair))))
        params = self._convert_from_ap1rog(self.npair, self.nspatial, gem_coeffs)
        return params


    @staticmethod
    def _convert_from_ap1rog(npair, nspatial, matrix):
        """
        Initialize an APr2G coefficient vector from the AP1roG matrix.
        """

        # Construct a least-squares augmented matrix
        A = np.zeros((matrix.size + 1, npair + 2 * nspatial), dtype=matrix.dtype)
        lambdas = A[:, :npair]
        epsilons = A[:, npair:npair + nspatial]
        xis = A[:, npair + nspatial:npair + 2 * nspatial]
        for i in range(npair):
            j = i * nspatial
            lambdas[j:j + nspatial, i] = matrix[i, :]
            epsilons[j:j + nspatial, :] = np.diag(-matrix[i, :])
            xis[j:j + nspatial, :] = -np.eye(nspatial)
        A[-1, npair + nspatial] = -1000
        b = np.zeros(matrix.size + 1, dtype=matrix.dtype)
        b[-1] = 1000

        # Solve the least-squares system
        sol = np.linalg.lstsq(A, b)[0]
        x = np.zeros(sol.size, dtype=sol.dtype)
        x[:] = sol
        return x

    def compute_pspace(self, num_sd):
        """ Generates Slater determinants to project onto

        # FIXME: wording
        Since APIG wavefunction only consists of Slater determinants with orbitals that are
        paired (alpha and beta orbitals corresponding to the same spatial orbital are occupied),
        the Slater determinants used correspond to those in DOCI wavefunction

        Parameters
        ----------
        num_sd : int
            Number of Slater determinants to generate

        Returns
        -------
        pspace : list of gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        """
        return doci_sd_list(self, num_sd)

    def compute_overlap(self, sd, deriv=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        overlap : float
        """
        # caching is done wrt mpz objects, so you should convert sd to mpz first
        sd = mpz(sd)
        # get indices of the occupied orbitals
        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
        occ_alpha_indices = slater.occ_indices(alpha_sd)
        occ_beta_indices = slater.occ_indices(beta_sd)
        if occ_alpha_indices != occ_beta_indices:
            raise ValueError('Given Slater determinant, {0}, does not belong'
                             ' to the DOCI Slater determinants'.format(bin(sd)))

        # build geminal coefficient matrix: parameters are assumed to be ordered as [lambda, epsilon, xi]
        params = self.params[:-1]

        val = 0.0
        # if no derivatization
        if deriv is None:
            val = permanent_borchardt(params, self.npair, self.npair)
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.params.size - 1:
            # Determine the indices of the parameters which will correspond to occupied columns
            # Check that parameter to derivatize is present in the matrix
            indices = list(range(self.npair))
            indices.extend([self.npair + c for c in occ_alpha_indices])
            indices.extend([self.npair + self.nspatial + c for c in occ_alpha_indices])
            if deriv not in indices:
                val = 0.0
            else:
                # create new list leaving out parameters corresponding to unoccupied columns
                deriv = indices.index(deriv)
                new_params = params[indices]

                # create geminal coefficient matrix and element-wise squared of the matrix
                lambda_matrix = np.array([params[:self.npair], ] * self.nspatial).transpose()
                epsilon_matrix = np.array([params[self.npair:self.npair + self.nspatial], ] * self.npair)
                xi_matrix = np.array([params[self.npair + self.nspatial:], ] * self.npair)
                gem_coeffs = (xi_matrix / (lambda_matrix - epsilon_matrix))[:, occ_alpha_indices]
                ew_square = gem_coeffs.copy()
                ew_square **= 2

                # compute derivative of the geminal coefficient matrix
                d_gemcoeffs = np.zeros_like(gem_coeffs)
                d_ewsquare = np.zeros_like(ew_square)

                # if deriving w.r.t. lambda_i: only changes row i
                if 0 <= deriv < self.npair:
                    num = - new_params[2 * self.npair:]
                    denom = np.array([new_params[deriv], ] * self.npair) - new_params[self.npair:2 * self.npair]
                    d_gemcoeffs[deriv, :] = num / denom ** 2
                    d_ewsquare[deriv, :] = -2 * num ** 2 / denom ** 3
                # if deriving w.r.t. epsilon_i: only changes column i
                elif self.npair <= deriv < 2 * self.npair:
                    num = np.array([new_params[deriv+self.npair], ] * self.npair)
                    denom = new_params[:self.npair] - np.array([new_params[deriv], ] * self.npair)
                    d_gemcoeffs[:, deriv - self.npair] = (num / denom ** 2).transpose()
                    d_ewsquare[:, deriv - self.npair] = (2 * num ** 2 / denom ** 3).transpose()
                # if deriving w.r.t xi_i: only changes column i
                else:
                    num = np.array([new_params[deriv], ] * self.npair)
                    denom = new_params[:self.npair] - np.array([new_params[deriv - self.npair], ] * self.npair)
                    d_gemcoeffs[:, deriv - 2 * self.npair] = (1 / denom).transpose()
                    d_ewsquare[:, deriv - 2 * self.npair] = (2 * num / denom ** 2).transpose()

                # compute determinants and adjugate matrices
                det_gemcoeffs = np.linalg.det(gem_coeffs)
                det_ewsquare = np.linalg.det(ew_square)
                adj_gemcoeffs = adjugate(gem_coeffs)
                adj_ewsquare = adjugate(ew_square)

                val = np.trace(adj_ewsquare.dot(d_ewsquare)) / det_ewsquare \
                        - det_ewsquare / det_gemcoeffs ** 2 * np.trace(adj_gemcoeffs.dot(d_gemcoeffs))

            # construct new gmpy2.mpz to describe the slater determinant and
            # derivation index
            self.d_cache[(sd, deriv)] = val
        return val

    def compute_hamiltonian(self, sd, deriv=None):
        """ Computes the hamiltonian of the wavefunction with respect to a Slater
        determinant

        ..math::
            \big< \Phi_i \big| H \big| \Psi \big>

        Since only Slater determinants from DOCI will be used, we can use the DOCI
        Hamiltonian

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        float
        """
        return sum(doci_hamiltonian(self, sd, self.orb_type, deriv=deriv))

    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in
        ProjectionWavefunction.compute_norm

        Some of the cache are emptied because the parameters are rewritten
        """
        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.npair, self.nspatial)
        # normalize the geminals
        norm = np.sum(gem_coeffs**2, axis=1)
        gem_coeffs *= np.abs(norm[:, np.newaxis])**(-0.5)
        # flip the negative norms
        gem_coeffs[norm < 0, :] *= -1
        # normalize the wavefunction
        norm = self.compute_norm()
        gem_coeffs *= norm**(-0.5/self.npair)
        # set attributes
        self.params = np.hstack((gem_coeffs.flatten(), self.params[-1]))
        # FIXME: need smarter caching (just delete the ones affected)
        for sd in self.ref_sd:
            del self.cache[sd]
            # This requires d_cache to be a dictionary of dictionary
            # self.d_cache[sd] = {}
            for i in (j for j in self.d_cache.keys() if j[0]==sd):
                del self.cache[i]
