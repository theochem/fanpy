""" Antisymmeterized Product of Geminals
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .geminal import Geminal
from ..proj_wavefunction import ProjectedWavefunction
from ... import slater
from ...graphs import generate_complete_pmatch
from ...math_tools import permanent_ryser

class APG(Geminal):
    """ Antisymmeterized Product of Geminals Wavefunction

    Here, the geminal wavefunction is built with the perfect matchings of a graph whose edges
    are the orbital pairs that are used to construct the wavefunction. For example, the APG
    wavefunction is analogous to all perfect matchings of a complete graph. The APsetG wavefunction
    is analogous to all perfect matchings of a complete bipartite graph. Though this structure is
    very flexible, the cost of the wavefunction is restricted by the cost of permanent evaluation.
    For now, the perfect matching representation of a geminal wavefunction is restricted to the
    evaluation of permanents, so the numerically efficient approximations will also be restricted to
    those of the permanent (i.e. borchardt theorem and reference Slater determinant).

    If you want to use another function to evaluate the overlap, such as a pfaffian, you will be
    better off constructing a new child of Geminal class. Otherwise, a child of APG class should be
    sufficient after overwriting template_orbpairs and assign_pmatch_generator.

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
    template_coeffs : np.ndarray
        Initial guess coefficient matrix for the given reference Slater determinants


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
        The number of Slater determinants is truncated by the number of parameters plus a magic
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
    """
    @property
    def template_orbpairs(self):
        """ List of orbital pairs that will be used to construct the geminals
        """
        return tuple((i, j) for i in range(self.nspin) for j in range(i+1, self.nspin))


    @property
    def template_coeffs(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_coeffs : np.ndarray(K, )

        Note
        ----
        Assumes that C_{p;ij} = C_{p;ji}

        """
        gem_coeffs = np.zeros((self.ngem, self.norbpair), dtype=self.dtype)
        # FIXME: only applies to reference Slater determinant of ground state HF
        col_inds = [self.dict_orbpair_ind[(i, i+self.nspatial)] for i in range(self.npair)]
        gem_coeffs[:, col_inds] += np.eye(self.ngem, self.npair)
        return gem_coeffs


    @property
    def norbpair(self):
        """ Number of orbital pairs used to construct the geminals
        """
        return len(self.dict_orbpair_ind)


    def __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None, pspace=None,
                 ref_sds=None, params=None, ngem=None, orbpairs=None, pmatch_generator=None):
        """ Initializes a wavefunction

        Parameters
        ----------
        nelec : int
            Number of electrons

        one_int : np.ndarray(K,K), 1- or 2-tuple np.ndarray(K,K)
            One electron integrals
            For spatial and generalized orbitals, np.ndarray or 1-tuple of np.ndarray
            For unretricted spin orbitals, 2-tuple of np.ndarray

        two_int : np.ndarray(K,K,K,K), 1- or 3-tuple np.ndarray(K,K,K,K)
            For spatial and generalized orbitals, np.ndarray or 1-tuple of np.ndarray
            For unrestricted orbitals, 3-tuple of np.ndarray

        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type
            Default is `np.float64`

        nuc_nuc : {float, None}
            Nuclear nuclear repulsion value
            Default is `0.0`

        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbital used in obtaining the one-electron and two-electron integrals
            Default is `'restricted'`

        pspace : list/tuple of int/long/gmpy2.mpz, None
            Slater determinants onto which the wavefunction is projected
            Default uses `generate_pspace`

        ref_sds : int/long/gmpy2.mpz, list/tuple of int/long/gmpy2.mpz, None
            Slater determinants that will be used as a reference for the wavefunction (e.g. for
            initial guess, energy calculation, normalization, etc)
            Default uses first Slater determinant of `pspace`

        params : np.ndarray, None
            Parameters of the wavefunction (including energy)
            Default uses `template_coeffs` and energy of the reference Slater determinants

        ngem : int, None
            Number of geminals

        orbpairs : tuple/list of 2-tuple of ints
            List of orbital pairs that are allowed to contribute to geminals

        pmatch_generator : function
            Function that returns the perfect matchings available for a given Slater determinant
            Input is list/tuple of orbitals indices that are occupied in the Slater determinant
            Generates a `npair`-tuple of 2-tuple where each 2-tuple is an orbital pair.
            For example, {0b1111:(((0,2),(1,3)), ((0,1),(2,3)))} would associate the
            Slater determinant `0b1111` with pairing schemes ((0,2), (1,3)), where
            `0`th and `2`nd orbitals are paired with `1`st and `3`rd orbitals, respectively,
            and ((0,1),(2,3)) where `0`th and `1`st orbitals are paired with `2`nd
            and `3`rd orbitals, respectively.
        """
        # NOTE: modified from Geminal.__init__ because APG.pspace depends on assign_orbpairs and
        #       APG.assign_params depends on assign_pmatch_generator
        super(ProjectedWavefunction, self).__init__(nelec, one_int, two_int, dtype=dtype,
                                                    nuc_nuc=nuc_nuc, orbtype=orbtype)
        self.cache = {}
        self.d_cache = {}
        self.assign_ngem(ngem=ngem)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_pspace(pspace=pspace)
        self.assign_ref_sds(ref_sds=ref_sds)
        self.assign_pmatch_generator(pmatch_generator=pmatch_generator)
        self.assign_params(params=params)


    def assign_pmatch_generator(self, pmatch_generator=None):
        """ Assigns the function that is used to generate the perfect matchings of a Slater
        determinant

        Parameters
        ----------
        pmatch_generator : function
            Function that returns the perfect matchings available for a given Slater determinant
            Input is list/tuple of orbitals indices that are occupied in the Slater determinant
            Generates a `npair`-tuple of 2-tuple where each 2-tuple is an orbital pair.
            For example, {0b1111:(((0,2),(1,3)), ((0,1),(2,3)))} would associate the
            Slater determinant `0b1111` with pairing schemes ((0,2), (1,3)), where
            `0`th and `2`nd orbitals are paired with `1`st and `3`rd orbitals, respectively,
            and ((0,1),(2,3)) where `0`th and `1`st orbitals are paired with `2`nd
            and `3`rd orbitals, respectively.

        Raises
        ------
        TypeError
            If pmatch_generator is not a function
        ValueError
            If pairing scheme contains a pair with the wrong number (not 2) of orbital indices
            If pairing scheme contains more than `npair` orbital pairs
            If pairing scheme contains orbital indices that are not in the given Slater determinant
        """
        if pmatch_generator is None:
            pmatch_generator = generate_complete_pmatch
        if not callable(pmatch_generator):
            raise TypeError('Given pmatch_generator is not a function')
        # quite expensive
        for sd in self.pspace:
            occ_indices = slater.occ_indices(sd)
            for scheme in pmatch_generator(occ_indices):
                if not all(len(pair) == 2 for pair in scheme):
                    raise ValueError('All pairing in the scheme must be in 2')
                if len(scheme) != self.npair:
                    raise ValueError('There is at least one redundant orbital pair')
                if set(occ_indices) != set(orb_ind for pair in scheme for orb_ind in pair):
                    raise ValueError('Pairing scheme contains orbitals that is not contained in'
                                     ' the provided Slater determinant')
        self.pmatch_generator = pmatch_generator


    def compute_overlap(self, sd, deriv=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.
        ..math::
            \big< \Phi_k \big| \Psi_{\mathrm{APIG}} \big>
            &= \big< \Phi_k \big| \prod_{p=1}^P two_int_p^\dagger \big| \theta \big>\\
            &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}}
               | C(\mathbf{m}) |^+ \big< \Phi_k \big| \mathbf{m} \big>
            &= | C(\Phi_k) |^+
        where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a
        Slater determinant (DOCI).

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        pairing_scheme : generator of list of list of 2 ints
            Contains all of available pairing schemes for the given Slater determinant
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        overlap : float

        Raises
        ------
        ValueError
            If the number of electrons of the wavefunction and the SD does not match
        """
        sd = slater.internal_sd(sd)
        # get indices of the occupied orbitals
        occ_indices = slater.occ_indices(sd)
        if len(occ_indices) != self.nelec:
            raise ValueError('Given Slater determinant, {0}, does not have the same number of'
                             ' electrons as ground state HF wavefunction'.format(bin(sd)))
        # get the pairing schemes
        pairing_schemes = self.pmatch_generator(occ_indices)
        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.ngem, self.norbpair)

        val = 0.0
        # if no derivatization
        if deriv is None:
            for scheme in pairing_schemes:
                # FIXME: it will be better to get the sign as we generate the pairing scheme
                sign = (-1)**slater.find_num_trans([i for pair in scheme for i in pair],
                                                   occ_indices, is_creator=True)
                indices = [self.dict_orbpair_ind[pair] for pair in scheme]
                matrix = gem_coeffs[:, indices]
                val += sign * permanent_ryser(matrix)
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.params.size - 1:
            row_to_remove = deriv // self.norbpair
            col_to_remove = deriv % self.norbpair
            orbs_to_remove = self.dict_ind_orbpair[col_to_remove]
            # if the column corresponds to orbitals that are occupied in the Slater determinant
            if orbs_to_remove[0] in occ_indices and orbs_to_remove[1] in occ_indices:
                for scheme in pairing_schemes:
                    # if the column/orbital pair is used to describe the Slater determinant
                    if orbs_to_remove in scheme:
                        # FIXME: it will be better to get the sign as we generate the pairing scheme
                        sign = (-1)**slater.find_num_trans([i for pair in scheme for i in pair],
                                                           occ_indices, is_creator=True)
                        row_inds = [i for i in range(self.npair) if i != row_to_remove]
                        col_inds = [self.dict_orbpair_ind[i] for i in scheme
                                    if self.dict_orbpair_ind[i] != col_to_remove]
                        if len(row_inds) == 0 and len(col_inds) == 0:
                            val += sign * 1
                        else:
                            val += sign * permanent_ryser(gem_coeffs[row_inds][:, col_inds])
                self.d_cache[(sd, deriv)] = val
        return val


    # FIXME: REPEATED CODE IN A LOT OF GEMINAL CODE
    def normalize(self):
        """ Normalizes the wavefunction such that the norm with respect to `ref_sds` is 1

        Raises
        ------
        ValueError
            If the norm is zero
            If the norm is negative
        """
        norm = self.compute_norm(ref_sds=self.ref_sds)
        if abs(norm) < 1e-9:
            raise ValueError('Norm of the wavefunction is zero. Cannot normalize')
        if norm < 0:
            raise ValueError('Norm of the wavefunction is negative. Cannot normalize')
        self.params[:-1] *= norm**(-0.5/self.ngem)
        self.cache = {sd : val * norm**(-0.5) for sd, val in self.cache.iteritems()}
        self.d_cache = {d_sd : val * norm**(-0.5) for d_sd, val in self.cache.iteritems()}
