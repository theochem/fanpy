"""Antisymmeterized Product of Rank-2 Geminals (APr2G) Wavefunction."""
from __future__ import absolute_import, division, print_function
import functools
import numpy as np
from wfns.wavefunction.geminals.apig import APIG
from wfns.wavefunction.geminals.base_geminal import BaseGeminal
from wfns.backend import slater, math_tools
from pydocstring.wrapper import docstring_class

__all__ = []


@docstring_class(indent_level=1)
class APr2G(APIG):
    r"""Antisymmeterized Product of Rank-2 Geminals (APr2G) Wavefunction.

    APIG wavefunction where the geminal coefficient is parameterized as a rank-2 Cauchy matrix.

    .. math::

        C_{pi} = \frac{\zeta_i}{\epsilon_i - \lambda_p}

    Then, the evaluation of the permanent is equilalent to product of determinants.

    .. math::

        |C|^+ = \frac{|C \circ C|^-}{|C|^-}

    """
    # FIXME: add constraints to parameters
    #        zetas should be less than 1
    #        lambda - epsilon should be greater than 1
    #        lambda should be around 1 (epsilons should be less than 0)
    @property
    def template_params(self):
        """

        Parameters are ordered as follows: lambda_1, ..., lambda_p, epsilon_1, ..., epsilon_k,
        zeta_1, ..., zeta_k.

        Notes
        -----
        Requires calculation. May be slow.

        """
        apig_template = super().template_params
        apig_template += 0.0001*np.random.rand(*apig_template.shape)
        return self.params_from_apig(apig_template, rmsd=0.01)

    @property
    def lambdas(self):
        r"""Return the :math:`lambda` part of the APr2G parameters.

        Returns
        -------
        lambdas : np.ndarray(ngem)
            The :math:`lambda` part of the APr2G parameters.

        """
        return self.params[:self.ngem]

    @property
    def epsilons(self):
        r"""Return the :math:`epsilons` part of the APr2G parameters.

        Returns
        -------
        epsilons : np.ndarray(nspatial)
            The :math:`epsilon` part of the APr2G parameters.

        """
        return self.params[self.ngem:self.ngem+self.norbpair]

    @property
    def zetas(self):
        r"""Return the :math:`zetas` part of the APr2G parameters.

        Returns
        -------
        zetas : np.ndarray(nspatial)
            The :math:`zeta` part of the APr2G parameters.

        """
        return self.params[self.ngem+self.norbpair:]

    @property
    def apig_params(self):
        """Return corresponding APIG parameters.

        Returns
        -------
        apig_params : np.ndarray(ngem, nspatial)
            Geminal coefficient matrix.

        """
        return self.zetas / (self.lambdas[:, np.newaxis] - self.epsilons)

    def assign_params(self, params=None):
        """

        Raises
        ------
        TypeError
            If `params` is not a numpy array.
            If `params` does not have data type of `float`, `complex`, `np.float64` and
            `np.complex128`.
            If `params` has complex data type and wavefunction has float data type.
        ValueError
            If `params` does not have the same shape as the template_params.
            If APr2G parameters create a zero in the denominator.
        NotImplementedError
            If BaseGeminal instance is given as the parameter.

        """
        if isinstance(params, BaseGeminal):
            raise NotImplementedError('APr2G wavefunction cannot assign parameters using a '
                                      'BaseGeminal instance.')
        super().assign_params(params=params)
        # check for zeros in denominator
        if np.any(np.abs(self.lambdas[:, np.newaxis] - self.epsilons) < 1e-9):
            raise ValueError('Corresponding geminal coefficient matrix has a division by zero')

    def compute_permanent(self, col_inds, deriv=None):
        """

        Parameters
        ----------
        col_inds : np.ndarray
            Indices of the columns of geminal coefficient matrices that will be used.
        deriv : {int, None}
            Indices of the element (in APr2G parameters) with respect to which the permanent is
            derivatized.
            Default is no derivatization.

        Returns
        -------
        permanent : float
            Permanent of the matrix that corresponds to the given column indices.

        Raises
        ------
        ValueError
            If index with respect to which the permanent is derivatized is invalid.

        """
        row_inds = np.arange(self.ngem)
        col_inds = np.array(col_inds)

        if deriv is None:
            return math_tools.permanent_borchardt(self.lambdas[row_inds],
                                                  self.epsilons[col_inds],
                                                  self.zetas[col_inds])
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
                    for orbpairs, sign in self.generate_possible_orbpairs(occ_indices):
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
        r"""Convert APIG parameters to APr2G wavefunction.

        Using least squares, the APIG geminal coefficients are converted to the APr2G variant, i.e.
        find the coefficients :math:`\{\lambda_j\}`, :math:`\{\epsilon_i\}`, and :math:`\{\zeta_i\}`
        such that following equation is best satisfied:

        .. math::
            C_{ij} &= \frac{\zeta_i}{\epsilon_i + \lambda_j}\\
            0 &= \zeta_i - C_{ij} \epsilon_i - C_{ij} \lambda_j\\

        The least square has the form of :math:`Ax=b`. Given that the :math:`b=0` and the unknowns
        are

        .. math::
            x = \begin{bmatrix}
            \lambda_1 \\ \vdots\\ \lambda_K\\
            \zeta_1 \\ \vdots\\ \zeta_P\\
            \epsilon_1 \\ \vdots\\ \epsilon_P\\
            \end{bmatrix},

        Then, A must be

        .. math::
            A =
            \left[
            \begin{array}{@{}*{12}{c}@{}}
                -C_{11} & 0 & \dots & 0 & -C_{11} & 0 & \dots & 0 & 1 & 0 & \dots & 0\\
                -C_{12} & 0 & \dots & 0 & 0 & -C_{12} & \dots & 0 & 0 & 1 & \dots & 0\\
                \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
                & \vdots & \vdots & \vdots\\
                -C_{1K} & 0 & \dots & 0 & 0 & 0 & \dots & -C_{1K} & 0 & 0 & \dots & 1\\
                0 & -C_{21} & \dots & 0 & -C_{21} & 0 & \dots & 0 & 1 & 0 & \dots & 0\\
                \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
                & \vdots & \vdots & \vdots\\
                0 & -C_{2K} & \dots & 0 & 0 & 0 & \dots & -C_{2K} & 0 & 0 & \dots & 1\\
                0 & 0 & \dots & -C_{PK} & -C_{P1} & 0 & \dots & 0 & 1 & 0 & \dots & 0\\
                \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
                & \vdots & \vdots & \vdots\\
                0 & 0 & \dots & -C_{PK} & 0 & 0 & \dots & -C_{PK} & 0 & 0 & \dots & 1\\
            \end{array}
            \right]

        Parameters
        ----------
        apig_params : np.ndarray(P,K)
            Geminal coefficient matrix.
            Number of rows is the number of geminals.
            Number of columns is the number of orbital pairs.
        rmsd : float
            Root mean square deviation allowed for the generated APr2G coefficient matrix (compared
            to the APIG coefficient matrix).
            Default is `0.1`.
        method : {'least squares', 'svd'}
            Method by which the APr2G parameters are obtained.
            Default is 'least squares'.

        Returns
        -------
        apr2g_params : APr2G instance
            APr2G parameters that "best" corresponds to the given APIG parameters.

        Raises
        ------
        ValueError
            If generate APr2G coefficient matrix has a root mean square deviation with the APIG
            coefficient matrix that is greater than the threshold value.

        Notes
        -----
        This does not always work. You will likely need to tinker with some of the parameters inside
        this function.

        Examples
        --------
        Assuming we have a system with 2 electron pairs and 4 spatial orbitals, we have

        .. math::
            C = \begin{bmatrix}
            C_{11} & \dots & C_{1K}\\
            C_{21} & \dots & C_{2K}
            \end{bmatrix}

        .. math::
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

        .. math::
            x = \begin{bmatrix}
            \lambda_ 1& \lambda_2
            \epsilon_1 \\ \epsilon_2\ \\ \epsilon_3 \\ \epsilon_4\\
            \zeta_1 \\ \zeta_2\\
            \end{bmatrix}

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
            not_indices = np.logical_not(indices)
            apr2g_params[not_indices] = np.linalg.lstsq(matrix[:, not_indices], ordinate)[0]
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
