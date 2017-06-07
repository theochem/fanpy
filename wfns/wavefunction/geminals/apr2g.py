"""Antisymmeterized Product of Rank-2 Geminals (APr2G) Wavefunction."""
from __future__ import absolute_import, division, print_function
import functools
import numpy as np
from .apig import APIG
from .base_geminal import BaseGeminal
from ...backend import slater, math_tools

__all__ = []


class APr2G(APIG):
    """Antisymmeterized Product of Rank-2 Geminals (APr2G) Wavefunction.

    Attributes
    ----------
    nelec : int
        Number of electrons
    nspin : int
        Number of spin orbitals (alpha and beta)
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction
    params : np.ndarray
        Parameters of the wavefunction
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j
        to the column index of the geminal coefficient matrix
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j

    Properties
    ----------
    npairs : int
        Number of electorn pairs
    nspatial : int
        Number of spatial orbitals
    ngem : int
        Number of geminals
    spin : float, None
        Spin of the wavefunction
        :math:`\frac{1}{2}(N_\alpha - N_\beta)` (Note that spin can be negative)
        None means that all spins are allowed
    seniority : int, None
        Seniority (number of unpaired electrons) of the wavefunction
        None means that all seniority is allowed
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters
    template_params : np.ndarray
        Template for the initial guess of geminal coefficient matrix
        Depends on the attributes given

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_nspin(self, nspin)
        Assigns the number of spin orbitals
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_params(self, params)
        Assigns the parameters of the wavefunction
    assign_ref_sds(self, ref_sds=None)
        Assigns the reference Slater determinants from which the initial guess, energy, and norm are
        calculated
        Default is the first Slater determinant of projection space
    assign_orbpairs(self, orbpairs=None)
        Assigns the orbital pairs that will be used to construct geminals
    compute_permanent(self, orbpairs, deriv_row_col=None)
        Compute the permanent that corresponds to the given orbital pairs
    get_overlap(self, sd, deriv_ind=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    generate_possible_orbpairs(self, occ_indices)
        Yields the possible orbital pairs that can construct the given Slater determinant.
    """
    # FIXME: add constraints to parameters
    #        zetas should be less than 1
    #        lambda - epsilon should be greater than 1
    #        lambda should be around 1 (epsilons should be less than 0)
    @property
    def template_params(self):
        """Return the template of the parameters in a APr2G wavefunction.

        Ordered as [lambda_1, ... , lambda_p, epsilon_1, ... , epsilon_k, zeta_1, ... , zeta_k]

        Note
        ----
        Requires calculation. May be slow.
        """
        apig_template = super().template_params
        apig_template += 0.0001*np.random.rand(*apig_template.shape)
        return self.params_from_apig(apig_template, rmsd=0.01)

    @property
    def lambdas(self):
        """Return the :math:`lambda` part of the APr2G parameters."""
        return self.params[:self.ngem]

    @property
    def epsilons(self):
        """Return the :math:`epsilons` part of the APr2G parameters."""
        return self.params[self.ngem:self.ngem+self.norbpair]

    @property
    def zetas(self):
        """Return the :math:`zetas` part of the APr2G parameters."""
        return self.params[self.ngem+self.norbpair:]

    @property
    def apig_params(self):
        """Return corresponding APIG parameters.

        Returns
        -------
        apig_params : np.ndarray(P, K)
        """
        return self.zetas / (self.lambdas[:, np.newaxis] - self.epsilons)

    def assign_params(self, params=None):
        """ Assigns the parameters of the wavefunction

        Parameters
        ----------
        params : np.ndarray, None
            Parameters of the wavefunction
        add_noise : bool
            Flag to add noise to the given parameters

        Raises
        ------
        TypeError
            If `params` is not a numpy array
            If `params` does not have data type of `float`, `complex`, `np.float64` and
            `np.complex128`
            If `params` has complex data type and wavefunction has float data type
        ValueError
            If `params` does not have the same shape as the template_params
            If APr2G parameters result in geminal coefficient matrix with a zero denominator

        Note
        ----
        Depends on dtype, template_params, and nparams
        """
        if isinstance(params, BaseGeminal):
            raise NotImplementedError('APr2G wavefunction cannot assign parameters using a '
                                      'BaseGeminal instance.')
        super().assign_params(params=params)
        # check for zeros in denominator
        if np.any(np.abs(self.lambdas[:, np.newaxis] - self.epsilons) < 1e-9):
            raise ValueError('Corresponding geminal coefficient matrix has a division by zero')

    def compute_permanent(self, col_inds, deriv=None):
        """Compute the permanent that corresponds to the given orbital pairs

        Parameters
        ----------
        col_inds : np.ndarray
            Indices of the columns of geminal coefficient matrices that will be used.
        deriv : int, None
            Indices of the element (in APr2G parameters) with respect to which the permanent is
            derivatized
            Default is no derivatization

        Returns
        -------
        permanent :float

        Raises
        ------
        ValueError
            If index with respect to which the permanent is derivatized is invalid
        """
        row_inds = np.arange(self.ngem)
        col_inds = np.array(col_inds)

        if deriv is None:
            return math_tools.permanent_borchardt(self.lambdas[row_inds],
                                                  self.epsilons[col_inds], self.zetas[col_inds])
        # if differentiating along row (lambda)
        # FIXME: not the best way of evaluating
        elif 0 <= deriv < self.npair:
            row_to_remove = deriv
            row_inds_trunc = row_inds[row_inds != row_to_remove]
            val = 0.0
            for col_to_remove in col_inds:
                col_inds_trunc = col_inds[col_inds != col_to_remove]
                # this will never happen (but just in case)
                if row_inds_trunc.size == row_inds.size or col_inds_trunc.size == col_inds.size:
                    continue
                # derivative of matrix element c_ij wrt lambda_i
                der_cij_rowi = (-self.zetas[col_to_remove] /
                                (self.lambdas[row_to_remove] - self.epsilons[col_to_remove])**2)
                if row_inds_trunc.size == col_inds_trunc.size == 0:
                    val += der_cij_rowi
                else:
                    val += (der_cij_rowi
                            * math_tools.permanent_borchardt(self.lambdas[row_inds_trunc],
                                                             self.epsilons[col_inds_trunc],
                                                             self.zetas[col_inds_trunc]))
            return val
        # if differentiating along column (epsilon or zeta)
        elif self.ngem <= deriv < self.ngem + 2*self.norbpair:
            col_to_remove = (deriv - self.ngem) % self.norbpair
            col_inds_trunc = col_inds[col_inds != col_to_remove]
            if col_inds_trunc.size == col_inds.size:
                return 0.0

            val = 0.0
            for row_to_remove in row_inds:
                row_inds_trunc = row_inds[row_inds != row_to_remove]
                # differentiating wrt column
                if self.ngem <= deriv < self.ngem + self.norbpair:
                    # derivative of matrix element c_ij wrt epsilon_j
                    der_cij_colj = (self.zetas[col_to_remove] /
                                    (self.lambdas[row_to_remove] - self.epsilons[col_to_remove])**2)
                else:
                    # derivative of matrix element c_ij wrt zeta_j
                    der_cij_colj = 1.0/(self.lambdas[row_to_remove] - self.epsilons[col_to_remove])

                if row_inds_trunc.size == col_inds_trunc.size == 0:
                    val += der_cij_colj
                else:
                    val += (der_cij_colj
                            * math_tools.permanent_borchardt(self.lambdas[row_inds_trunc],
                                                             self.epsilons[col_inds_trunc],
                                                             self.zetas[col_inds_trunc]))
            return val
        else:
            raise ValueError('Invalid derivatization index.')

    # FIXME: mostly replicates BaseGeminal.get_overlap
    def get_overlap(self, sd, deriv=None):
        """Compute the overlap between the geminal wavefunction and a Slater determinant.

        The results are cached in self._cache_fns

        .. math::

            \big| \Psi \big>
            &= \prod_{p=1}^{N_{gem}} \sum_{pq} C_{pq} a^\dagger_p a^\dagger_q \big| \theta \big>\\
            &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} |C(\mathbf{m})|^+
            \big| \mathbf{m} \big>

        where :math:`N_{gem}` is the number of geminals, :math:`\mathbf{m}` is a Slater determinant.

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant as a bitstring
        deriv : None, int
            Index of the paramater with respect to which the overlap is derivatized
            Default is no derivatization

        Returns
        -------
        overlap : float

        Note
        ----
        Bit of performance is lost in exchange for generalizability. Hopefully it is still readable.
        """
        if not slater.is_internal_sd(sd):
            sd = slater.internal_sd(sd)

        if deriv is None:
            return super().get_overlap(sd)
        elif isinstance(deriv, int):
            if deriv >= self.nparams:
                return 0.0
            # if differentiating along column/epsilon/zeta
            if self.ngem <= deriv < self.ngem + 2*self.norbpair:
                col_removed = (deriv - self.ngem) % self.norbpair
                orb_1, orb_2 = self.dict_ind_orbpair[col_removed]
                # if differentiating along column that is not used by the Slater determinant
                if not (slater.occ(sd, orb_1) and slater.occ(sd, orb_2)):
                    return 0.0

            # if cached function has not been created yet
            if 'overlap derivative' not in self._cache_fns:
                # assign memory allocated to cache
                if self.memory == np.inf:
                    memory = None
                else:
                    memory = int((self.memory - 5*8*self.params.size)
                                 / (self.params.size + 1) * self.params.size)

                # create function that will be cached
                @functools.lru_cache(maxsize=memory, typed=False)
                def _olp_deriv(sd, deriv):
                    occ_indices = slater.occ_indices(sd)

                    val = 0.0
                    for orbpairs in self.generate_possible_orbpairs(occ_indices):
                        col_inds = np.array([self.dict_orbpair_ind[orbp] for orbp in orbpairs])
                        val += self.compute_permanent(col_inds, deriv=deriv)
                    return val

                # store the cached function
                self._cache_fns['overlap derivative'] = _olp_deriv
            # if cached function already exists
            else:
                # reload cached function
                _olp_deriv = self._cache_fns['overlap derivative']

            return _olp_deriv(sd, deriv)

    @staticmethod
    def params_from_apig(apig_params, rmsd=0.1, method='least squares'):
        """Convert APIG parameters to APr2G wavefunction.

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
        apig_params : np.ndarray(P,K)
            APIG geminal coefficient matrix
            Number of rows is the number of geminals
            Number of columns is the number of orbital pairs
        rmsd : float
            Root mean square deviation allowed for the generated APr2G coefficient matrix (compared
            to the APIG coefficient matrix)
            Default is 0.1
        method : {'least squares', 'svd'}
            Method by which the APr2G parameters are obtained
            Default is 'least squares'

        Returns
        -------
        apr2g_params : APr2G instance
            APr2G parameters that best corresponds to the given APIG parameters

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

        Note
        ----
        This does not always work. You will likely need to tinker with some of the parameters inside
        this function.
        """
        ngem, norbpair = apig_params.shape
        # assign least squares matrix by reference
        matrix = np.zeros((apig_params.size, ngem + 2*norbpair), dtype=apig_params.dtype)
        # set up submatrices that references a specific part of the matrix
        # NOTE: these values are broadcasted to `matrix`
        lambdas = matrix[:, :ngem]
        epsilons = matrix[:, ngem:ngem + norbpair]
        zetas = matrix[:, ngem + norbpair:ngem + 2*norbpair]
        for i in range(ngem):
            lambdas[i*norbpair:(i + 1)*norbpair, i] = -apig_params[i, :]
            epsilons[i*norbpair:(i + 1)*norbpair, :] = np.diag(apig_params[i, :])
            zetas[i*norbpair:(i + 1)*norbpair, :] = np.identity(norbpair)

        # solve by Least Squares
        if method == 'least squares':
            # Turn system of equations heterogeneous
            indices = np.zeros(ngem + 2*norbpair, dtype=bool)
            vals = np.array([])

            # assign lambdas
            # indices[:ngem] = True
            # vals = np.hstack((vals, [1]*ngem))

            # assign epsilons
            # indices[ngem] = True
            # indices[ngem+norbpair-1] = True
            # vals = np.hstack((vals, [-10, -1]))

            # assign zetas
            indices[ngem + norbpair] = True
            vals = np.hstack((vals, [1]))
            # indices[ngem + norbpair:ngem + 2*norbpair] = True
            # vals = np.hstack((vals, np.ones(norbpair)))

            ordinate = -matrix[:, indices].dot(vals)

            # Solve the least-squares system
            apr2g_params = np.zeros(indices.size)
            apr2g_params[indices] = vals
            apr2g_params[-indices] = np.linalg.lstsq(matrix[:, -indices], ordinate)[0]
        # solve by SVD
        elif method == 'svd':
            u, s, vT = np.linalg.svd(matrix, full_matrices=False)
            # find null vectors
            indices = np.abs(s) < 1
            # guess solution
            b = np.vstack([np.random.rand(ngem, 1),
                           np.sort(np.random.rand(norbpair, 1)) - 1,
                           np.ones((norbpair, 1))])
            # linearly combine right null vectors
            lin_comb = np.linalg.lstsq(vT[indices].T, b)[0]
            apr2g_params = vT[indices].T.dot(lin_comb).flatten()

        # Check
        lambdas = apr2g_params[:ngem, np.newaxis]
        epsilons = apr2g_params[ngem:ngem+norbpair]
        zetas = apr2g_params[ngem+norbpair:]
        apr2g_coeffs = zetas / (lambdas - epsilons)
        deviation = (np.sum((apig_params - apr2g_coeffs)**2)/apig_params.size)**(0.5)
        if np.isnan(deviation) or deviation > rmsd:
            raise ValueError('APr2G coefficient matrix has RMSD of {0} with the APIG coefficient'
                             ' matrix'.format(deviation))

        return apr2g_params
