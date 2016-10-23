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
    def bounds(self):
        """ Boundaries for the parameters

        Used to set bounds on the optimizer

        Returns
        -------
        bounds : iterable of 2-tuples
            Each 2-tuple correspond to the min and the max value for the parameter
            with the same index.
        """
        # FIXME: apply better bounds
        # remove bounds on parameters (for now)
        # bounds = [(-np.inf, np.inf) for i in self.nparam]
        low_bounds = [-np.inf for i in range(self.nparam)]
        upp_bounds = [np.inf for i in range(self.nparam)]
        return (tuple(low_bounds), tuple(upp_bounds))

    def assign_params(self, params=None, ap1rog_params=None):
        """ Assigns the parameters to the wavefunction

        Parameters
        ----------
        params : np.ndarray(K,)
            Parameters of the wavefunction
        ap1rog_coeffs : np.ndarray(P*K+1, )
            Geminal coefficients and the energy of AP1roG wavefunction
        """
        super(APr2G, self).assign_params(params=params)
        assert not ((params is not None) and (ap1rog_params is not None)),\
            'Cannot give both params and ap1rog_params'
        if ap1rog_params is not None:
            gem_coeffs = np.hstack((np.identity(self.npair), ap1rog_params[:-1].reshape(self.npair, self.nspatial-self.npair)))
            gem_params = self._convert_from_ap1rog(self.npair, self.nspatial, gem_coeffs)
            self.params = np.hstack((gem_params, ap1rog_params[-1]))


    @property
    def template_coeffs(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to this template
        The parameters are assumed to be ordered as [lambda_1, ... , lambda_p, epsilon_1, ... , epsilon_k, xi_1, ... , xi_k]

        Returns
        -------
        template_coeffs : np.ndarray(K, )

        """
        params = np.zeros(self.npair + self.nspatial*2)
        # set epsilons to 1
        params[self.npair:self.npair+self.nspatial] = 1.0
        # set zetas such that first npair columns are ones
        params[self.npair+self.nspatial:self.npair+2*self.nspatial] = 1.0
        return params


    # FIXME: this needs to be checked
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
        occ_indices = occ_alpha_indices

        # get the appropriate parameters: parameters are assumed to be ordered as [lambda, epsilon, xi]
        lambdas = self.params[:self.npair]
        epsilons = self.params[self.npair:self.npair+self.nspatial][occ_indices,]
        zetas = self.params[self.npair+self.nspatial:self.npair+self.nspatial*2][occ_indices,]

        gem_coeffs = zetas / (lambdas[:, np.newaxis] - epsilons)
        ew_square = gem_coeffs**2

        val = 0.0
        # if no derivatization
        if deriv is None:
            val = permanent_borchardt(lambdas, epsilons, zetas)
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.params.size - 1:
            # FIXME: move to math_tools.permanent_borchardt?
            # compute derivative of the geminal coefficient matrix
            d_gemcoeffs = np.zeros((self.npair, self.npair))
            d_ewsquare = np.zeros((self.npair, self.npair))

            # if derivatizing with respect to row elements (lambdas)
            if 0 <= deriv < self.npair:
                d_gemcoeffs[deriv, :] = -zetas / (lambdas[deriv] - epsilons) ** 2
                d_ewsquare[deriv, :] = -2*zetas**2 / (lambdas[deriv] - epsilons)**3
            # if derivatizing with respect to column elements (epsilons, zetas)
            elif self.npair <= deriv < self.npair + 2*self.nspatial:
                # convert deriv index to the column index
                deriv = (deriv - self.npair) % self.nspatial
                if deriv not in occ_indices:
                    return 0.0
                # if derivatizing with respect to epsilons
                elif deriv < self.npair + self.nspatial:
                    # convert deriv index to index within occ_indices
                    deriv = occ_indices.index(deriv)
                    # calculate derivative
                    d_gemcoeffs[:, deriv] = zetas[deriv] / (lambdas - epsilons[deriv])**2
                    d_ewsquare[:, deriv] = 2*zetas[deriv]**2 / (lambdas - epsilons[deriv])**3
                # if derivatizing with respect to zetas
                elif deriv >= self.npair + self.nspatial:
                    # convert deriv index to index within occ_indices
                    deriv = occ_indices.index(deriv)
                    # calculate derivative
                    d_gemcoeffs[:, deriv] = 1.0 / (lambdas - epsilons[deriv])
                    d_ewsquare[:, deriv] = 2*zetas[deriv] / (lambdas - epsilons[deriv])**2
            else:
                return 0.0

            # compute determinants and adjugate matrices
            det_gemcoeffs = np.linalg.det(gem_coeffs)
            det_ewsquare = np.linalg.det(ew_square)
            adj_gemcoeffs = adjugate(gem_coeffs)
            adj_ewsquare = adjugate(ew_square)

            val = (np.trace(adj_ewsquare.dot(d_ewsquare)) / det_gemcoeffs
                    - det_ewsquare / (det_gemcoeffs ** 2) * np.trace(adj_gemcoeffs.dot(d_gemcoeffs)))

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

    # FIXME: remove
    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in
        ProjectionWavefunction.compute_norm

        Some of the cache are emptied because the parameters are rewritten
        """
        # normalize the wavefunction
        norm = self.compute_norm()
        # set attributes
        self.params[self.npair+self.nspatial:self.npair+self.nspatial*2]*= norm**(-0.5/self.npair)
        for sd in self.default_ref_sds:
            del self.cache[sd]
            # This requires d_cache to be a dictionary of dictionary
            # self.d_cache[sd] = {}
            for i in (j for j in self.d_cache.keys() if j[0]==sd):
                del self.cache[i]
