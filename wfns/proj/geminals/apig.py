""" APIG wavefunction
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .geminal import Geminal
from ... import slater
from ...math_tools import permanent_ryser


class APIG(Geminal):
    """ Antisymmetric Product of Interacting Geminals

    ..math::
        \ket{\Psi_{\mathrm{APIG}}}
        &= \prod_{p=1}^P two_int_p^\dagger \ket{\theta}\\
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
        Function in FancyCI
    """
    _seniority = 0
    _spin = 0

    @property
    def template_orbpairs(self):
        """ Default orbital pairs used to construct the wavefunction
        """
        return tuple((i, i+self.nspatial) for i in range(self.nspatial))


    @property
    def template_coeffs(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_coeffs : np.ndarray
        """
        # FIXME: only applies to reference Slater determinant of ground state HF
        # FIXME: should find perfect matching of the template_orbpairs to construct the refernece SD
        # TODO: raise error if none exist
        # NOTE: not too easy b/c ref_sds requires pspace which requires template_coeffs (cyclic
        #       dependence)
        return np.eye(self.ngem, self.nspatial, dtype=self.dtype)


    def assign_orbpairs(self, orbpairs=None):
        """ Assigns the orbital pairs that will be used to construct geminals

        Parameters
        ----------
        orbpairs : tuple/list of 2-tuple of ints
            Each 2-tuple is an orbital pair that is allowed to contribute to geminals

        Raises
        ------
        TypeError
            If `orbpairs` is not a tuple/list of 2-tuples
        ValueError
            If an orbital pair has the same integer
            If any two orbital pair shares an orbital
            If any orbital is not included in any orbital pair
        """
        super(APIG, self).assign_orbpairs(orbpairs=orbpairs)
        all_orbs = [j for i in self.dict_orbpair_ind.iterkeys() for j in i]
        if len(all_orbs) != len(set(all_orbs)):
            raise ValueError('At least two orbital pairs share an orbital')
        elif len(set(all_orbs)) != self.nspin:
            raise ValueError('Not all of the orbitals are included in orbital pairs')


    def compute_overlap(self, sd, deriv=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.
        ..math::
            \braket{\Phi_k | \Psi_{\mathrm{APIG}}}
            &= \braket{\Phi_k | \prod_{p=1}^P two_int_p^\dagger | \theta }\\
            &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} |C(\mathbf{m})|^+
               \braket{\Phi_k | \mathbf{m}}
            &= | C(\Phi_k) |^+
        where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a Slater determinant
        (DOCI).

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant as a bitstring
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

        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)

        val = 0.0
        # if no derivatization
        if deriv is None:
            val = permanent_ryser(gem_coeffs[:, occ_indices])
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.params.size - 1:
            row_to_remove = deriv // self.nspatial
            col_to_remove = deriv % self.nspatial
            if col_to_remove in occ_indices:
                row_inds = [i for i in range(int(self.npair)) if i != row_to_remove]
                col_inds = [i for i in occ_indices if i != col_to_remove]
                if len(row_inds) == 0 and len(col_inds) == 0:
                    val = 1.0
                else:
                    val = permanent_ryser(gem_coeffs[row_inds, :][:, col_inds])
                # construct new gmpy2.mpz to describe the slater determinant and
                # derivation index
                self.d_cache[(sd, deriv)] = val
        return val


    # FIXME; REPEATED CODE IN A LOT OF GEMINAL CODE
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


    def to_apr2g(self, rmsd=0.01):
        """ Converts APIG wavefunction to APr2G wavefunction

        Using least squares, the APIG geminal coefficients are converted to the APr2G variant, i.e.
        find the coefficients :math:`\{\lambda_j\}`, :math:`\{\epsilon_i\}`, and :math:`\{\zeta_i\}`
        such that following equation is best satisfied
        ..math::
            C_{ij} &= \frac{\zeta_i}{\epsilon_i + \lambda_j}\\
            0 &= \zeta_i - C_{ij} \epsilon_i - C_{ij} \lambda_j\\

        The least square has the form of :math:`Ax=b`. Given that the :math:`b=0`
        and the unknowns are
        ..math::
            x = \begin{bmatrix}
            \lambda_1 \\ \vdots\\ \lambda_K\\
            \zeta_1 \\ \vdots\\ \zeta_P\\
            \epsilon_1 \\ \vdots\\ \epsilon_P\\
            \end{bmatrix},
        then A must be
        ..math::
            A = \begin{bmatrix}
            -C_{11} & 0 & \dots & 0 & -C_{11} & 0 & \dots & 0 &  & 1 & 0 & \dots & 0\\
            -C_{12} & 0 & \dots & 0 & 0 & -C_{12} & \dots & 0 &  & 0 & 1 & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
            & \vdots & \vdots\\
            -C_{1K} & 0 & \dots & 0 & 0 & 0 & \dots & -C_{1K} &  & 0 & 0 & \dots & 1\\
            0 & -C_{21} & \dots & 0 & -C_{21} & 0 & \dots & 0 &  & 1 & 0 & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
            & \vdots & \vdots\\
            0 & -C_{2K} & \dots & 0 & 0 & 0 & \dots & -C_{2K} &  & 0 & 0 & \dots & 1\\
            0 & 0 & \dots & -C_{PK} & -C_{P1} & 0 & \dots & 0 &  & 1 & 0 & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
            & \vdots & \vdots\\
            0 & 0 & \dots & -C_{PK} & 0 & 0 & \dots & -C_{PK} &  & 0 & 0 & \dots & 1\\
            \end{bmatrix}


        Parameters
        ----------
        rmsd : float
            Root mean square deviation allowed for the generated APr2G coefficient matrix (compared
            to the APIG coefficient matrix)

        Returns
        -------
        apr2g : APr2G instance
            APr2G wavefunction that best corresponds to the given APIG wavefunction

        Raises
        ------
        ValueError
            If generate APr2G coefficient matrix has a root mean square deviation with the APIG
            coefficient matrix that is greater than the threshold value

        Example
        -------
        Assuming we have a system with 2 electron pairs and 4 spatial orbitals,
        we have
        ..math::
            C = \begin{bmatrix}
            C_{11} & \dots & C_{1K}\\
            C_{21} & \dots & C_{2K}
            \end{bmatrix}
        ..math::
            A = \begin{bmatrix}
            C_{11} & 0 & -C_{11} & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
            C_{12} & 0 & 0 & -C_{12} & 0 & 0 & 0 & 1 & 0 & 0\\
            C_{13} & 0 & 0 & 0 & -C_{13} & 0 & 0 & 0 & 1 & 0\\
            C_{14} & 0 & 0 & 0 & 0 & -C_{14} & 0 & 0 & 0 & 1\\
            0 & C_{21} & -C_{21} & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
            0 & C_{22} & 0 & -C_{22} & 0 & 0 & 0 & 1 & 0 & 0\\
            0 & C_{23} & 0 & 0 & -C_{23} & 0 & 0 & 0 & 1 & 0\\
            0 & C_{24} & 0 & 0 & 0 & -C_{24} & 0 & 0 & 0 & 1\\
            \end{bmatrix}

        ..math::
            x = \begin{bmatrix}
            \lambda_ 1& \lambda_2
            \epsilon_1 \\ \epsilon_2\ \\ \epsilon_3 \\ \epsilon_4\\
            \zeta_1 \\ \zeta_2\\
            \end{bmatrix}
        """
        # import APr2G (because nasty cyclic import business)
        from .apr2g import APr2G

        # make coefficients
        apig_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)
        # assign least squares matrix by reference
        matrix = np.zeros((apig_coeffs.size, self.npair + 2*self.nspatial), dtype=self.dtype)
        # set up submatrices that references a specific part of the matrix
        lambdas = matrix[:, :self.npair]
        epsilons = matrix[:, self.npair:self.npair + self.nspatial]
        zetas = matrix[:, self.npair + self.nspatial:self.npair + 2*self.nspatial]
        for i in range(self.npair):
            lambdas[i*self.nspatial:(i + 1)*self.nspatial, i] = -apig_coeffs[i, :]
            epsilons[i*self.nspatial:(i + 1)*self.nspatial, :] = np.diag(apig_coeffs[i, :])
            zetas[i*self.nspatial:(i + 1)*self.nspatial, :] = np.identity(self.nspatial)

        # Turn system of equations heterogeneous
        # Select indices that will be assigned values
        indices = np.zeros(self.npair + 2*self.nspatial, dtype=bool)
        vals = np.array([])

        # assign epsilons
        #  with orbital energy
        # indices[self.npair:self.npair + self.nspatial] = True
        # vals = np.diag(self.one_int[0])
        # indices[self.npair] = True
        # vals = np.hstack((vals, np.diag(self.one_int[0])[0]))

        # assign zetas
        # FIXME: assign weights based on orbital coupling
        # indices[self.npair + self.nspatial:2*self.npair + self.nspatial] = True
        # vals = np.hstack((vals, np.ones(self.npair)))
        indices[self.npair + self.nspatial] = True
        vals = np.hstack((vals, 1))

        # find ordinate
        ordinate = -matrix[:, indices].dot(vals)

        # Solve the least-squares system
        sol = np.zeros(indices.size)
        sol[indices] = vals
        sol[-indices] = np.linalg.lstsq(matrix[:, -indices], ordinate)[0].astype(self.dtype)

        # Check
        lambdas = sol[:self.npair][:, np.newaxis]
        epsilons = sol[self.npair:self.npair+self.nspatial]
        zetas = sol[self.npair+self.nspatial:]
        apr2g_coeffs = zetas / (lambdas - epsilons)
        deviation = (np.sum((apig_coeffs - apr2g_coeffs)**2)/apig_coeffs.size)**(0.5)
        if deviation > rmsd or np.isnan(deviation):
            raise ValueError('APr2G coefficient matrix has RMSD of {0} with the APIG coefficient'
                             ' matrix'.format(deviation))
        # Add energy
        params = np.hstack((sol, self.get_energy()))
        # make wavefunction
        return APr2G(self.nelec, self.one_int, self.two_int, dtype=self.dtype, nuc_nuc=self.nuc_nuc,
                     orbtype=self.orbtype, pspace=self.pspace, ref_sds=self.ref_sds,
                     params=params, ngem=self.ngem, orbpairs=self.dict_orbpair_ind.keys())


    def to_apg(self):
        """ Converts APIG wavefunction to APG wavefunction

        Returns
        -------
        apg : APG instance
            APG wavefunction that corresponds to the given APIG wavefunction
        """
        # import APG (because nasty cyclic import business)
        from .apg import APG

        # make apig coefficients
        apig_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)
        # make apg coefficients
        apg = APG(self.nelec, self.one_int, self.two_int, dtype=self.dtype, nuc_nuc=self.nuc_nuc,
                  orbtype=self.orbtype, ref_sds=self.ref_sds, ngem=self.ngem)
        apg_coeffs = apg.params[:-1].reshape(apg.template_coeffs.shape)
        # assign apg coefficients
        for orbpair, ind in self.dict_orbpair_ind.items():
            apg_coeffs[:, apg.dict_orbpair_ind[orbpair]] = apig_coeffs[:, ind]
        apg.assign_params(np.hstack((apg_coeffs.flat, self.get_energy())))

        # make wavefunction
        return apg
