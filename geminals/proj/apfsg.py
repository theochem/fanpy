from __future__ import absolute_import, division, print_function

import numpy as np
from gmpy2 import mpz, popcount

from .proj_wavefunction import ProjectionWavefunction
from .. import slater
from ..sd_list import ci_sd_list
from ..math_tools import permanent_ryser
from .proj_hamiltonian import hamiltonian


class APfsG(ProjectionWavefunction):
    """ Antisymmetric Product of Factorized Set Geminals

    ..math::
        \big| \Psi_{\mathrm{APFSG}} \big>
        &= \prod_{q=1}^R T_q^\dagger \big| \theta \big>\\
        &= \prod_{q=1}^R \left(c_{q;0} + \sum_{i=1}^{B} c^{(CS)}_{q;i}a_{2i-1}^\dagger a_{2i}^\dagger + \sum_{j \in S_1} \sum_{k \in S_2}a^{(os)}_{q;i}b^{(os)}_{jk}a_j^\dagger a_k^\dagger (1 - \hat{n_{j'}}) (1-\hat{n_{k'}}) \right) \big| \theta \big>
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
    orb_type : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    params : np.ndarray(K)
        Guess for the parameters
        Iteratively updated during convergence
        Initial guess before convergence
        Coefficients after convergence
    cache : dict of mpz to float
        Cache of the Slater determinant to the overlap of the wavefunction with this
        Slater determinant
    d_cache : dict of (mpz, int) to float
        Cache of the Slater determinant to the derivative(with respect to some index)
        of the overlap of the wavefunction with this Slater determinan

    Properties
    ----------
    _methods : dict
        Default dimension of projection space
    _energy : float
        Electronic energy
    _nci : int
        Number of Slater determinants
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals
    npair : int
        Number of electron pairs (rounded down)
    nparam : int
        Number of parameters used to define the wavefunction
    nproj : int
        Number of Slater determinants to project against
    ref_sd : int or list of int
        Reference Slater determinants with respect to which the norm and the energy
        are calculated
        Integer that describes the occupation of a Slater determinant as a bitstring
        Or list of integers
    template_params : np.ndarray(K)
        Default numpy array of parameters.
        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

    Methods
    -------
    __init__(nelec=None, H=None, G=None, dtype=None, nuc_nuc=None, orb_type=None)
        Initializes wavefunction
    __call__(method="default", **kwargs)
        Solves the wavefunction
    assign_dtype(dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_integrals(H, G, orb_type=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
        (and the wavefunction)
    assign_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_nelec(nelec)
        Assigns the number of electrons
    _solve_least_squares(**kwargs)
        Solves the system of nonliear equations (and the wavefunction) using
        least squares method
    assign_params(params=None)
        Assigns the parameters used to describe the wavefunction.
        Adds random noise from the template if necessary
    assign_pspace(pspace=None)
        Assigns projection space
    overlap(sd, deriv=None)
        Retrieves overlap from the cache if available, otherwise compute overlap
    compute_norm(sd=None, deriv=None)
        Computes the norm of the wavefunction
    compute_energy(include_nuc=False, sd=None, deriv=None)
        Computes the energy of the wavefunction
    objective(x)
        The objective (system of nonlinear equations) associated with the projected
        Schrodinger equation
    jacobian(x)
        The Jacobian of the objective
    compute_pspace
        Generates a tuple of Slater determinants onto which the wavefunction is projected
    compute_overlap
        Computes the overlap of the wavefunction with one or more Slater determinants
    compute_hamiltonian
        Computes the hamiltonian of the wavefunction with respect to one or more Slater
        determinants
        By default, the energy is determined with respect to ref_sd
    normalize
        Normalizes the wavefunction (different definitions available)
        By default, the norm should the projection against the ref_sd squared
    """

    @property
    def template_params(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to this template

        Returns
        -------
        template_params : np.ndarray(K, )
        """
        init_pair = np.eye(self.npair, self.nspatial, dtype=self.dtype)
        init_unpair = np.eye(self.npair, self.nspatial, dtype=self.dtype)
        init_spin = np.eye(self.nspatial,
                           self.nspatial, dtype=self.dtype).flatten()
        init_spatial = np.hstack((init_pair, init_unpair)).flatten()
        return np.hstack((init_spatial, init_spin))

    def compute_pspace(self, num_sd):
        """ Generates Slater determinants to project onto

        # to do: add some more detailed doc

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
        number_sd = 2 * self.npair * self.nspatial + self.nspatial ** 2
        return ci_sd_list(self, number_sd)

    def compute_overlap(self, sd, deriv=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.
        ..math::
            \big< \Phi_q^i \big| \Psi_{\mathrm{APFSG}} \big>
            &= \big< \Phi_q^i \big| \prod_{q=1}^Q T_q^\dagger \big| \theta \big>\\
            &=\begin{vmatrix}
            c^{(cs)}_{1;i_1} & c^{(cs)}_{1;i_2} & \dots & c^{(cs)}_{1;i_{N_{cs}}} & a^{(os)}_{1;i_1} & a^{(os)}_{1;i_2} & \dots & a^{(os)}_{1;i_{N_{cs}}}\\
            c^{(cs)}_{2;i_1} & c^{(cs)}_{2;i_2} & \dots & c^{(cs)}_{2;i_{N_{cs}}} & a^{(os)}_{2;i_1} & a^{(os)}_{2;i_2} & \dots & a^{(os)}_{2;i_{N_{cs}}}\\
            \vdots & \vdots  &  & \vdots  & \vdots  & \vdots  &  &\vdots \\
            c^{(cs)}_{Q;i_1} & c^{(cs)}_{Q;i_2} & \dots & c^{(cs)}_{Q;i_{N_{cs}}} & a^{(os)}_{Q;i_1} & a^{(os)}_{Q;i_2} & \dots & a^{(os)}_{Q;i_{N_{cs}}}
            \end{vmatrix}^+
            \begin{vmatrix}
            b^{(os)}_{j_1k_1} & b^{(os)}_{j_1k_2} & \dots & b^{(os)}_{j_1k_{N_{os}}} \\
            b^{(os)}_{j_2k_1} & b^{(os)}_{j_2k_2} & \dots & b^{(os)}_{j_2k_{N_{os}}} \\
            \vdots & \vdots & & \vdots \\
            b^{(os)}_{j_{N_{cs}}k_1} & b^{(os)}_{j_{N_{cs}}k_2} & \dots & b^{(os)}_{j_{N_{cs}}k_{N_{os}}}
            \end{vmatrix}


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
        sd = mpz(sd)
        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
        if popcount(alpha_sd) != popcount(beta_sd):
            return 0.
        occ_alpha_indices = slater.occ_indices(alpha_sd)
        occ_beta_indices = slater.occ_indices(beta_sd)
        pair_sd = alpha_sd & beta_sd
        unpair_sd = alpha_sd ^ beta_sd
        pair_list = slater.occ_indices(pair_sd)
        unpair_list = slater.occ_indices(unpair_sd)
        # build the occupations of bosons and fermions
        boson_list = list(pair_list) + [i + len(pair_list) for i in unpair_list]
        # build geminal and sd coefficient
        num_boson_part = self.npair*self.nspatial*2
        boson_part = self.params[:num_boson_part].reshape(self.npair, 2*self.nspatial)
        fermion_part = self.params[num_boson_part: -1].reshape(self.nspatial, self.nspatial)

        val = 0
        if deriv is None:
            val = permanent_ryser(boson_part[:, boson_list])
            val *= np.linalg.det(fermion_part[occ_alpha_indices, :][:, occ_beta_indices])
            self.cache[sd] = val
        else:
            # if the parameter to derivatize is in the permanent
            if deriv < num_boson_part:
                row_to_remove = deriv // boson_part.shape[1]
                col_to_remove = deriv % boson_part.shape[1]
                if col_to_remove in boson_list:
                    row_inds = [i for i in range(boson_part.shape[0]) if i != row_to_remove]
                    col_inds = [i for i in boson_list if i != col_to_remove]
                    if len(row_inds) == len(col_inds) == 0:
                        val = 1
                    else:
                        val = permanent_ryser(boson_part[row_inds, :][:, col_inds])
                    val *= np.linalg.det(fermion_part)
                    self.d_cache[(sd, deriv)] = val
            # if the parameter to derivatize is in the determinant
            elif deriv < self.params.size - 1:
                row_to_remove = deriv - boson_part.size // fermion_part.shape[1]
                col_to_remove = deriv - boson_part.size % fermion_part.shape[1]
                if row_to_remove in occ_alpha_indices and col_to_remove in occ_beta_indices:
                    row_inds = [i for i in occ_alpha_indices if i != row_to_remove]
                    col_inds = [i for i in occ_beta_indices if i != col_to_remove]
                    if len(row_inds) == len(col_inds) == 0:
                        val = 1
                    else:
                        val = np.linalg.det(fermion_part[row_inds, :][:, col_inds])
                    val *= permanent_ryser(boson_part)
                    self.d_cache[(sd, deriv)] = val
        return val

    def compute_hamiltonian(self, sd, deriv=None):
        """ Computes the hamiltonian of the wavefunction with respect to a Slater
        determinant

        ..math::
            \big< \Phi_i \big| H \big| \Psi_{mathrm{APFSG}} \big>

        Since only Slater determinants from CI will be used, we can use the CI
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
        return sum(hamiltonian(self, sd, self.orb_type, deriv=deriv))

    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in
        ProjectionWavefunction.compute_norm

        Some of the cache are emptied because the parameters are rewritten

        APFSG is normalized by contruction.
        """
        pass
