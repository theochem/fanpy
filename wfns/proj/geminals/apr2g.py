""" APr2G wavefunction
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .apig import APIG
from ... import slater
from ...math_tools import permanent_borchardt, adjugate


class APr2G(APIG):
    """ Antisymmetric Product of rank-2 Geminals

    ..math::
        \ket{\Psi_{\mathrm{APr2G}}}
        &= \prod_{p=1}^P G_p^\dagger \ket{\theta}\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} |C(\mathbf{m})|^+
        \ket{\mathbf{m}}
    where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a
    Slater determinant (DOCI).

    Class Variables
    ---------------
    _nconstraints : int
        Number of constraints
    _seniority : int, None
        Seniority of the wavefunction
        None means that all seniority is allowed
    _spin : float, None
        Spin of the wavefunction
        :math:`\frac{1}{2}(N_\alpha - N_\beta)` (Note that spin can be negative)
        None means that all spins are allowed

    Attributes
    ----------
    nelec : int
        Number of electrons
    one_int : 1- or 2-tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        1-tuple for spatial (restricted) and generalized orbitals
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
    two_int : 1- or 3-tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        In physicist's notation
        1-tuple for spatial (restricted) and generalized orbitals
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components)
    dtype : {np.float64, np.complex128}
        Numpy data type
    nuc_nuc : float
        Nuclear-nuclear repulsion energy
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    pspace : tuple of gmpy2.mpz
        Slater determinants onto which the wavefunction is projected
    ref_sds : tuple of gmpy2.mpz
        Slater determinants that will be used as a reference for the wavefunction (e.g. for
        initial guess, energy calculation, normalization, etc)
    params : np.ndarray
        Parameters of the wavefunction (including energy)
    cache : dict of sd to float
        Cache of the overlaps that are calculated for each Slater determinant encountered
    d_cache : dict of gmpy2.mpz to float
        Cache of the derivative of overlaps that are calculated for each Slater determinant and
        derivative index encountered
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j
        to the column index of the geminal coefficient matrix
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j

    Properties
    ----------
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals
    nparams : int
        Number of parameters
    nproj : int
        Number of Slater determinants
    npair : int
        Number of electron pairs
    ngem : int
        Number of geminals
    template_coeffs : np.ndarray
        Initial guess coefficient matrix for the given reference Slater determinants
    template_orbpairs : tuple
        List of orbital pairs that will be used to construct the geminals

    Method
    ------
    __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_integrals(self, one_int, two_int, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
    assign_pspace(self, pspace=None)
        Assigns the tuple of Slater determinants onto which the wavefunction is projected
        Default uses `generate_pspace`
    generate_pspace(self)
        Generates the default tuple of Slater determinants with the appropriate spin and seniority
        in increasing excitation order.

        number (42)
    assign_ref_sds(self, ref_sds=None)
        Assigns the reference Slater determinants from which the initial guess, energy, and norm are
        calculated
        Default is the first Slater determinant of projection space
    assign_params(self, params=None)
        Assigns the parameters of the wavefunction (including energy)
        Default contains coefficients from abstract property, `template_coeffs`, and the energy of
        the reference Slater determinants with the coefficients from `template_coeffs`
    assign_orbpairs(self, orbpairs=None)
        Assigns the orbital pairs that will be used to construct geminals
    get_overlap(self, sd, deriv=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    compute_norm(self, ref_sds=None, deriv=None)
        Calculates the norm from given Slater determinants
        Default `ref_sds` is the `ref_sds` given by the intialization
        Default is no derivatization
    compute_hamitonian(self, slater_d, deriv=None)
        Calculates the expectation value of the Hamiltonian projected onto the given Slater
        determinant, `slater_d`
        By default no derivatization
    compute_energy(self, include_nuc=False, ref_sds=None, deriv=None)
        Calculates the energy projected onto given Slater determinants
        Default `ref_sds` is the `ref_sds` given by the intialization
        By default, electronic energy, no derivatization
    objective(self, x, weigh_constraints=True)
        Objective of the equations that will need to be solved (to solve the Projected Schrodinger
        equation)
    jacobian(self, x, weigh_constraints=True)
        Jacobian of the objective
    compute_overlap(self, sd, deriv=None)
        Calculates the overlap between the wavefunction and a Slater determinant
        Function in FancyCI
    """
    # FIXME: add constraints to parameters
    #        zetas should be less than 1
    #        lambda - epsilon should be greater than 1
    #        lambda should be around 1 (epsilons should be less than 0)
    @property
    def template_coeffs(self):
        """ Default parameters for the given reference Slater determinantes

        Ordered as [lambda_1, ... , lambda_p, epsilon_1, ... , epsilon_k, zeta_1, ... , zeta_k]

        Note
        ----
        Returned value must be a numpy array in the desired shape
        """
        # FIXME: not a very good guess
        params = np.zeros(self.npair + self.nspatial*2)
        # set lambdas
        params[:self.npair] = np.arange(1, 0.8, -0.2/self.npair)
        # set epsilons to orbital energies
        lowest_orb_energy = self.one_int[0][0, 0]
        params[self.npair:self.npair + self.nspatial] = np.arange(lowest_orb_energy, 0,
                                                                  -lowest_orb_energy/self.nspatial)
        # set first npair zetas to 1
        params[self.npair + self.nspatial:2*self.npair + self.nspatial] = 1.0
        # set rest to 0.01
        params[2*self.npair + self.nspatial:self.npair + 2*self.nspatial] = 0.01
        return params


    def assign_params(self, params=None):
        """ Assigns the parameters of the wavefunction

        Parameters
        ----------
        params : np.ndarray, None
            Parameters of the wavefunction
            Last parameter is the energy
            Default is the `template_coeffs` for the coefficient and energy of the reference
            Slater determinants
            If energy is given as zero, the energy of the reference Slater determinants are used
        add_noise : bool
            Flag to add noise to the given parameters

        Raises
        ------
        TypeError
            If `params` is not a numpy array
            If `params` does not have data type of `float`, `complex`, `np.float64` and
            `np.complex128`
            If `params` has data type of `float or `np.float64` and wavefunction does not have data
            type of `np.float64`
        ValueError
            If `params` is None (default) and `ref_sds` has more than one Slater determinants
            If `params` is not a one dimensional numpy array with appropriate dimensions
            If geminal coefficient matrix has a zero in the denominator
        """
        super(self.__class__, self).assign_params(params=params)
        # check for zeros in denominator
        lambdas = self.params[:self.npair]
        epsilons = self.params[self.npair:self.npair+self.nspatial]
        if np.any(np.abs(lambdas[:, np.newaxis] - epsilons) < 1e-9):
            raise ValueError('Geminal coefficient matrix has a division by zero')


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

        Raises
        ------
        ValueError
            If `sd` does not have same number of electrons as ground state HF
            If `sd` does not have seniority zero
        """
        # caching is done wrt mpz objects, so you should convert sd to mpz first
        sd = slater.internal_sd(sd)
        # get indices of the occupied orbitals
        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
        occ_indices = slater.occ_indices(alpha_sd)
        if len(occ_indices) != self.npair:
            raise ValueError('Given Slater determinant, {0}, does not have the same number of'
                             ' electrons as ground state HF wavefunction'.format(bin(sd)))
        if occ_indices != slater.occ_indices(beta_sd):
            raise ValueError('Given Slater determinant, {0}, does not belong'
                             ' to the DOCI Slater determinants'.format(bin(sd)))

        # get the appropriate parameters: parameters are assumed to be ordered as [lambda, epsilon,
        # zeta]
        lambdas = self.params[:self.npair]
        epsilons = self.params[self.npair:self.npair+self.nspatial]
        zetas = self.params[self.npair+self.nspatial:self.npair+self.nspatial*2]

        gem_coeffs = zetas / (lambdas[:, np.newaxis] - epsilons)

        val = 0.0
        # if no derivatization
        if deriv is None:
            val = permanent_borchardt(lambdas, epsilons[occ_indices,], zetas[occ_indices,])
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.params.size - 1:
            # FIXME: move to math_tools.permanent_borchardt?
            # dumb way for debugging purposes
            if deriv < self.npair:
                row_to_remove = deriv
                for col_to_remove in occ_indices:
                    row_inds = [i for i in range(int(self.npair)) if i != row_to_remove]
                    col_inds = [i for i in occ_indices if i != col_to_remove]
                    submatrix = gem_coeffs[row_inds, :][:, col_inds]
                    if len(row_inds) == 0 and len(col_inds) == 0:
                        val += (-zetas[col_to_remove]/(lambdas[row_to_remove]
                                                       - epsilons[col_to_remove])**2)
                    else:
                        val += (np.linalg.det(submatrix**2)/np.linalg.det(submatrix)
                                * (-zetas[col_to_remove]/(lambdas[row_to_remove]
                                                          - epsilons[col_to_remove])**2))
            elif self.npair <= deriv < self.npair + 2*self.nspatial:
                col_to_remove = (deriv - self.npair) % self.nspatial
                if col_to_remove in occ_indices:
                    for row_to_remove in range(int(self.npair)):
                        row_inds = [i for i in range(int(self.npair)) if i != row_to_remove]
                        col_inds = [i for i in occ_indices if i != col_to_remove]
                        submatrix = gem_coeffs[row_inds, :][:, col_inds]
                        if deriv < self.npair + self.nspatial:
                            if len(row_inds) == 0 and len(col_inds) == 0:
                                val += (zetas[col_to_remove]/(lambdas[row_to_remove]
                                                              - epsilons[col_to_remove])**2)
                            else:
                                val += (np.linalg.det(submatrix**2) / np.linalg.det(submatrix)
                                        * (zetas[col_to_remove]/(lambdas[row_to_remove]
                                                                 - epsilons[col_to_remove])**2))
                        else:
                            if len(row_inds) == 0 and len(col_inds) == 0:
                                val += 1.0/(lambdas[row_to_remove] - epsilons[col_to_remove])
                            else:
                                val += (np.linalg.det(submatrix**2)/np.linalg.det(submatrix)
                                        * (1.0/(lambdas[row_to_remove] - epsilons[col_to_remove])))
            self.d_cache[(sd, deriv)] = val

            # FIXME: needs to be debugged
            # # compute derivative of the geminal coefficient matrix
            # d_gemcoeffs = np.zeros((self.npair, self.npair))
            # d_ewsquare = np.zeros((self.npair, self.npair))

            # # if derivatizing with respect to row elements (lambdas)
            # if 0 <= deriv < self.npair:
            #     d_gemcoeffs[deriv, :] = -zetas / (lambdas[deriv] - epsilons) ** 2
            #     d_ewsquare[deriv, :] = -2*zetas**2 / (lambdas[deriv] - epsilons)**3
            # # if derivatizing with respect to column elements (epsilons, zetas)
            # elif self.npair <= deriv < self.npair + 2*self.nspatial:
            #     # convert deriv index to the column index
            #     deriv = (deriv - self.npair) % self.nspatial
            #     if deriv not in occ_indices:
            #         return 0.0
            #     # if derivatizing with respect to epsilons
            #     elif deriv < self.npair + self.nspatial:
            #         # convert deriv index to index within occ_indices
            #         deriv = occ_indices.index(deriv)
            #         # calculate derivative
            #         d_gemcoeffs[:, deriv] = zetas[deriv] / (lambdas - epsilons[deriv])**2
            #         d_ewsquare[:, deriv] = 2*zetas[deriv]**2 / (lambdas - epsilons[deriv])**3
            #     # if derivatizing with respect to zetas
            #     elif deriv >= self.npair + self.nspatial:
            #         # convert deriv index to index within occ_indices
            #         deriv = occ_indices.index(deriv)
            #         # calculate derivative
            #         d_gemcoeffs[:, deriv] = 1.0 / (lambdas - epsilons[deriv])
            #         d_ewsquare[:, deriv] = 2*zetas[deriv] / (lambdas - epsilons[deriv])**2
            # else:
            #     return 0.0

            # # compute determinants and adjugate matrices
            # det_gemcoeffs = np.linalg.det(gem_coeffs)
            # det_ewsquare = np.linalg.det(gem_coeffs**2)
            # adj_gemcoeffs = adjugate(gem_coeffs)
            # adj_ewsquare = adjugate(gem_coeffs**2)

            # val = (np.trace(adj_ewsquare.dot(d_ewsquare)) / det_gemcoeffs
            #       - det_ewsquare / (det_gemcoeffs * 2) * np.trace(adj_gemcoeffs.dot(d_gemcoeffs)))

            self.d_cache[(sd, deriv)] = val
        return val


    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in compute_norm

        Raises
        ------
        ValueError
            If the norm is zero
            If the norm is negative
        """
        # compute norm
        norm = self.compute_norm(ref_sds=self.ref_sds)
        # check norm
        if abs(norm) < 1e-9:
            raise ValueError('Norm of the wavefunction is zero. Cannot normalize')
        if norm < 0:
            raise ValueError('Norm of the wavefunction is negative. Cannot normalize')
        # set attributes
        self.params[self.npair+self.nspatial:self.npair+2*self.nspatial] *= norm**(-0.5/self.ngem)
        self.cache = {sd : val * norm**(-0.5) for sd, val in self.cache.iteritems()}
        self.d_cache = {d_sd : val * norm**(-0.5) for d_sd, val in self.cache.iteritems()}
