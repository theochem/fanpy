"""
APr2G geminal wavefunction class.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from geminals.apig import APIG
from geminals.slater_det import excite_pairs, is_occupied, is_pair_occupied


class APr2G(APIG):
    """
    A restricted antisymmetrized product of rank-two geminals (R(APr2G)) implementation.

    See APIG class documentation.

    """

    #
    # Class-wide (behaviour-changing) attributes and properties
    #

    _exclude_ground = False
    _is_complex = True
    _normalize = True

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        assert len(value) == value.shape[0] == 2 * self.norbs + self.npairs, \
            "The `params` (zeta|epsilon|lambda)^T should be of length (2K + P)."
        self._params = value

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        if value is None:
            return
        assert len(value) == value.shape[0] == 2 * self.norbs + self.npairs, \
            "APr2G coefficients can only be constructed from the parameters."
        self._coeffs = self._construct_coeffs(value)
        self._coeffs_optimized = True

    #
    # Methods
    #

    def _generate_x0(self, model_coeffs=None, energy_first=1, energy_last=100):
        """ Generates an initial guess

        If model_coeffs is given, we try to find a set of coefficients such that
        is as closely satisfied as possible, using least squares method
        ..math::
            C_{ij} &= \frac{\zeta_j}{\epsilon_j + \lambda_i}\\
            0 &= \zeta_j - C_{ij} \epsilon_j - C_{ij} \lambda_i\\
        This is a set of linear equations. These equations will be solved by least
        squares method.

        The least square has the form of :math:`Ax=b`. Given that the :math:`b=0`
        and the unknowns are
        ..math::
            x = \begin{bmatrix}
            \zeta_1 \\ \vdots\\ \zeta_K\\
            \epsilon_1 \\ \vdots\\ \epsilon_K\\
            \lambda_1 \\ \vdots\\ \lambda_P\\
            \end{bmatrix},
        then A must be
        ..math::
            A = \begin{bmatrix}
            1 & 0 & \dots & 0 & -C_{11} & 0 & \dots & 0 & -C_{11} & 0 & \dots & 0\\
            0 & 1 & \dots & 0 & 0 & -C_{12} & \dots & 0 & -C_{12} & 0 & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
            0 & 0 & \dots & 1 & 0 & 0 & \dots & -C_{1K} & -C_{1K} & 0 & \dots & 0\\
            1 & 0 & \dots & 0 & -C_{21} & 0 & \dots & 0 & 0 & -C_{21} & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
            0 & 0 & \dots & 1 & 0 & 0 & \dots & -C_{2K} & 0 & -C_{2K} & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
            1 & 0 & \dots & 0 & -C_{P1} & 0 & \dots & 0 & 0 & 0 & \dots & -C_{PK}\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
            0 & 0 & \dots & 1 & 0 & 0 & \dots & -C_{PK} & 0 & 0 & \dots & -C_{PK}\\
            \end{bmatrix}

        Parameter
        ---------
        model_coeffs : {None, np.ndarray(P,K)}
            Coefficients after which the initial guess is modeled

        Example
        -------
        Assuming we have a system with 2 electron pairs and 4 spatial orbitals,
        we have
        ..math::
            C = \begin{bmatrix}
            C_{11} & \dots & C_{1K}\\
            \vdots & \vdots & \vdots\\
            C_{P1} & \dots & C_{PK}
            \end{bmatrix}

        ..math::
            A = \begin{bmatrix}
            1 & 0 & 0 & 0 & -C_{11} & 0 & 0 & 0 & -C_{11} & 0\\
            0 & 1 & 0 & 0 & 0 & -C_{12} & 0 & 0 & -C_{12} & 0\\
            0 & 0 & 1 & 0 & 0 & 0 & -C_{13} & 0 & -C_{13} & 0\\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & -C_{14} & -C_{14} & 0\\
            1 & 0 & 0 & 0 & -C_{21} & 0 & 0 & 0 & 0 & -C_{21} \\
            0 & 1 & 0 & 0 & 0 & -C_{22} & 0 & 0 & 0 & -C_{22} \\
            0 & 0 & 1 & 0 & 0 & 0 & -C_{23} & 0 & 0 & -C_{23} \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & -C_{24} & 0 & -C_{24} \\
            \end{bmatrix}

        ..math::
            x = \begin{bmatrix}
            \zeta_1 \\ \zeta_2\\ \zeta_3 \\ \zeta_4\\
            \epsilon_1 \\ \epsilon_2\\ \epsilon_3\\ \epsilon_4\\
            \lambda_1 \\ \lambda_2 \\
            \end{bmatrix}
        """

        # This is NOT a good guess, it's a placeholder!
        if model_coeffs is None:
            x0 = 2.0*(np.random.rand(2*(2 * self.norbs + self.npairs)) - 0.5)
        else:
            # Here, we try to solve Ax = 0 where
            # A
            A = np.empty((0, 2*model_coeffs.shape[1] + model_coeffs.shape[0]))
            for row_ind, row in enumerate(model_coeffs):
                temp = np.empty((model_coeffs.shape[1], 0))
                # coefficients for zetas
                temp = np.hstack((temp, np.identity(row.size)))
                # coefficients for epsilons
                temp = np.hstack((temp, -row*np.identity(row.size)))
                # coefficients for lambdas
                temp_zeros = np.zeros(model_coeffs.T.shape)
                temp_zeros[:, row_ind] = row
                temp = np.hstack((temp, temp_zeros))
                A = np.vstack((A, temp))
            # b
            b = np.zeros((A.shape[0], 1))
            # Select method here
            method = 'set variables'
            if method == 'svd':
                # Ax = b, solve for x by svd
                u, s, vT = np.linalg.svd(A)
                # Select here
                n = 5
                assert np.all(s[-n:] < 0.1)
                b = np.vstack([np.ones((model_coeffs.shape[1], 1)),
                               np.sort(np.random.rand(model_coeffs.shape[1], 1)*100),
                               np.random.rand(model_coeffs.shape[0], 1)])
                lin_comb = np.linalg.lstsq(vT[-n:-1].T, b)[0]
                print(vT[-n:].T)
                x0 = vT[-n:-1].T.dot(lin_comb)
            elif method == 'set variables':
                # Turn equation heterogeneous (Set one variables)
                # Select indices here
                indices = list(range(model_coeffs.shape[0]))
                indices += [model_coeffs.shape[1], model_coeffs.shape[1]*2-1]
                row_indices = np.zeros(np.prod(model_coeffs.shape)).astype(bool)
                for i in indices:
                    row_indices[i::model_coeffs.shape[1]] = True
                var_indices = np.zeros(model_coeffs.shape[1]*2 + model_coeffs.shape[0]).astype(bool)
                var_indices[indices] = True
                # Select values here
                var_values = np.array([1]*model_coeffs.shape[0] + [energy_first, energy_last])
                b[row_indices, 0] -= np.sum(A[:, var_indices]*var_values, axis=1)[row_indices]
                x0 = np.zeros(A.shape[1])
                x0[var_indices] = var_values
                x0[-var_indices] = np.linalg.lstsq(A[:, -var_indices], b)[0]
                x0 = x0.reshape((A.shape[1], 1))
            else:
                x0 = np.linalg.lstsq(A, b)[0]
            diff = (np.sum(((model_coeffs - self._construct_coeffs(x0))/model_coeffs.size)**2))**0.5
            print('Difference with the initial coefficient: {0}'.format(diff))
        lambdas = x0[model_coeffs.shape[1]*2:]
        epsilons = x0[model_coeffs.shape[1]:model_coeffs.shape[1]+model_coeffs.shape[0]]
        assert np.all(epsilons-lambdas > 1e-9)
        return x0

    def _construct_coeffs(self, x0):
        """
        See APIG._construct_coeffs().

        """

        coeffs = np.zeros((self.npairs, self.norbs))
        for i in range(self.npairs):
            for j in range(self.norbs):
                coeffs[i, j] = x0[j] / (x0[self.norbs + j] - x0[2 * self.norbs + i])
        return coeffs

    def generate_pspace(self):
        """
        See APIG.generate_pspace().

        Notes
        -----
        If K > P + 2, then AP1roG has more free parameters than APr2G, and AP1roG's
        generate_pspace() can also be used to AP2rG.  However, if K <= P + 2, then APr2G
        has more free parameters than AP1roG, and APIG's generate_pspace() must be used.

        """

        if True: #self.norbs <= self.npairs + 2:
            # Use APIG's generate_pspace()
            self._is_complex = True
            return super(APr2G, self).generate_pspace()

        ground = self.ground
        pspace = [ground]

        # Return a tuple of all unique pair excitations
        for unoccup in range(self.npairs, self.norbs):
            for i in range(self.npairs):
                pspace.append(excite_pairs(ground, i, unoccup))
        return tuple(set(pspace))

    def overlap(self, phi, coeffs=None):
        """
        Compute the overlap of the APr2G wavefunction with a Slater determinant (or its
        derivative wrt one of the Borchardt parameters), using Borchardt's
        theorem and Jacobi's formula

        Parameters
        ----------
        phi : int
            The Slater determinant (the int's binary representation) whose overlap with
            the APr2G wavefunction is to be computed.

        coeffs : 1-index np.ndarray, optional
            In APr2G (and *only* in APr2G), the keyword argument `coeffs` represents the
            parameters vector (zeta|epsilon|lambda)^T.  Defaults to the APr2G instance's
            optimized parameters.

        Returns
        -------
        overlap : float

        Raises
        ------
        AssertionError
            If `coeffs` is not given, but the coefficients have not yet been optimized.

        """

        if coeffs is None:
            assert self._coeffs_optimized, \
                "The geminal coefficeint matrix has not yet been optimized."
            coeffs = self.params

        coeffs = self._params

        # If the Slater determinant is bad
        if phi == 0:
            return 0
        elif phi not in self.pspace:
            return 0

        # If the Slater det. and geminal wavefcuntion have a different number of electrons
        elif bin(phi).count("1") != self.nelec:
            return 0

        # If the Slater determinant is non-singlet
        elif any(is_occupied(phi, i * 2) != is_occupied(phi, i * 2 + 1) for i in range(self.norbs)):
            return 0

        ind_occ = [i for i in range(self.norbs) if is_pair_occupied(phi, i)]

        # Debug
        assert len(ind_occ) == self.npairs, \
            "A Slater determinant with the wrong number of electrons made it through."

        # Construct the square coefficient matrix, its Hadamard square, and their
        # determinants
        matrix = self._construct_coeffs(coeffs)[:, ind_occ]
        det_matrix = np.linalg.det(matrix)
        hadamard = matrix**2
        det_hadamard = np.linalg.det(hadamard)

        # If taking the derivative
        if self._overlap_derivative:

            # Evaluate the inverses of the matrix and its Hadamard square
            inv_matrix = np.linalg.inv(matrix)
            inv_hadamard = np.linalg.inv(hadamard)

            # Take the derivative of `matrix` and `hadamard` wrt a parameter in
            # (\zeta_j, \epsilon_j, \lambda_i) using Jacobi's formula
            deriv_matrix = np.zeros(self.npairs, self.npairs)
            deriv_hadamard = np.zeros(self.npairs, self.npairs)

            # d(X)/d(\zeta_j)
            if self._overlap_indices < self.norbs:
                # Turn the _`overlap_indices` into the correct `j`
                j_deriv = self._overlap_indices
                for i in range(self.npairs):
                    for j in range(self.npairs):
                        zeta_j = coeffs[ind_occ[j]]
                        epsilon_j = coeffs[self.norbs + ind_occ[j]]
                        lambda_i = coeffs[2 * self.norbs + ind_occ[i]]
                        # \zeta_j appears in the matrix element
                        if j == j_deriv:
                            deriv_matrix[i, j] = 1 / (epsilon_j - lambda_i)
                            deriv_hadamard[i, j] = 2 * zeta_j / ((epsilon_j - lambda_i)**2)
                        # \zeta_j does not appear in the matrix element
                        else:
                            deriv_matrix[i, j] = 0.
                            deriv_hadamard[i, j] = 0.

            # d(X)/d(\epsilon_j)
            elif self._overlap_indices < 2 * self.norbs:
                # Turn the _`overlap_indices` into the correct `j`
                j_deriv = self._overlap_indices - self.norbs
                for i in range(self.npairs):
                    for j in range(self.npairs):
                        zeta_j = coeffs[ind_occ[j]]
                        epsilon_j = coeffs[self.norbs + ind_occ[j]]
                        lambda_i = coeffs[2 * self.norbs + ind_occ[i]]
                        # \epsilon_j appears in the matrix element
                        if j == j_deriv:
                            deriv_matrix[i, j] = -zeta_j / ((epsilon_j - lambda_i)**2)
                            deriv_hadamard[i, j] = -2 * (zeta_j**2) / ((epsilon_j - lambda_i)**3)
                        # \epsilon_j does not appear in the matrix element
                        else:
                            deriv_matrix[i, j] = 0.
                            deriv_hadamard[i, j] = 0.

            # d(X)/d(\lambda_i)
            else:
                i_deriv = self._overlap_indices - 2 * self.norbs
                # Turn the _`overlap_indices` into the correct `i`
                for i in range(self.npairs):
                    for j in range(self.npairs):
                        zeta_j = coeffs[ind_occ[j]]
                        epsilon_j = coeffs[self.norbs + ind_occ[j]]
                        lambda_i = coeffs[2 * self.norbs + ind_occ[i]]
                        # \lambda_i appears in the matrix element
                        if i == i_deriv:
                            deriv_matrix[i, j] = zeta_j / ((epsilon_j - lambda_i)**2)
                            deriv_hadamard[i, j] = 2 * (zeta_j**2) / ((epsilon_j - lambda_i)**3)
                        # \lambda_i does not appear in the matrix element
                        else:
                            deriv_matrix[i, j] = 0.
                            deriv_hadamard[i, j] = 0.

            # Evaluate d(permanent)/dq
            overlap = np.trace(inv_hadamard.dot(deriv_hadamard))
            overlap -= np.trace(deriv_matrix.dot(inv_matrix))
            overlap *= det_hadamard / det_matrix

        # If not taking the derivative
        else:
            # Evaluate the permanent by Borchardt's theorem
            overlap = det_hadamard / det_matrix

        return overlap

    def nonlin(self, x0, pspace):
        """
        Construct the nonlinear system of equation which will be used to optimize the
        geminal coefficients.

        Parameters
        ----------
        x0 : 1-index np.ndarray
            A guess at the optimal geminal coefficients.

        pspace : iterable of ints
            Iterable of integers that, in binary, describes which spin orbitals are used
            to build the Slater determinant.

        Returns
        -------
        objective : 1-index np.ndarray
            The objective function of the nonlinear system.

        """
        # Handle differences in indexing between geminal methods
        coeffs = self._construct_coeffs(x0)
        self._params = x0
        energy = sum(self.compute_energy(self.ground, coeffs))
        objective = np.zeros(x0.size)
        objective[0] = np.real(self.overlap(self.ground, coeffs) - 1)
        objective[1] = np.imag(self.overlap(self.ground, coeffs))
        d_index = 0
        o_index = 2
        while o_index < objective.size:
            # Complex
            comp_eqn = (energy*self.overlap(pspace[d_index], coeffs)
                        -sum(self.compute_energy(pspace[d_index], coeffs)))
            # Real part
            objective[o_index] = np.real(comp_eqn)
            # Imaginary part
            objective[o_index + 1] = np.imag(comp_eqn)
            d_index += 1
            o_index += 2
        return objective

# vim: set textwidth=90 :
