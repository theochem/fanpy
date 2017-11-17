"""Wavefunction with nonorthonormal orbitals."""
import itertools as it
import functools
import numpy as np
from wfns.backend import slater
from wfns.wfn.composite.base_one import BaseCompositeOneWavefunction


# FIXME: needs refactoring
class NonorthWavefunction(BaseCompositeOneWavefunction):
    r"""Wavefunction with nonorthonormal orbitals expressed with respect to orthonormal orbitals.

    A parameterized multideterminantal wavefunction can be written as

    .. math::
        \ket{\Psi} = \sum_{\mathbf{m}} f(\mathbf{m}) \ket{\mathbf{m}}

    where :math:`\ket{\mathbf{m}}` is a Slater determinant. If the Slater determinants are
    constructed from nonorthonormal orbitals, then each Slater determinant can be expressed as a
    linear combination of Slater determinants constructed from orthonormal orbitals.

    .. math::
        \ket{\Psi}
        &= \sum_{\mathbf{n}} f(\mathbf{n}) \ket{\mathbf{n}}\\
        &= \sum_{\mathbf{n}} f(\mathbf{n}) \sum_{\mathbf{m}}
        |C(\mathbf{n}, \mathbf{m})|^- \ket{\mathbf{m}}\\
        &= \sum_{\mathbf{n}} \sum_{\mathbf{m}}
        f(\mathbf{n}) |C(\mathbf{n}, \mathbf{m})|^- \ket{\mathbf{m}}

    where :math:`\ket{\mathbf{m}}` and :math:`\ket{\mathbf{n}}` are Slater determinants constructed
    from orthonormal and nonorthonormal orbitals. The nonorthonormal orbitals are constructed by
    linearly transforming the orbitals of :math:`\ket{\mathbf{m}}` with :math:`C`. The
    :math:`C(\mathbf{n}, \mathbf{m})` is a submatrix of :math:`C` where rows are selected according
    to :math:`\ket{\mathbf{n}}` and columns to :math:`\ket{\mathbf{m}}`.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    params : tuple of np.ndarray
        Orbital transformation matrices.
        If one transformation matrix is given, then the transformation coresponds to those of
        restricted orbitals, where the spatial orbitals are transformed or to those of generalized
        orbitals, where spin orbitals are transformed.
        If two transformation matrices are given, then the transformation corresponds to those of
        unrestricted orbitals, where the spatial orbitals are transformed.
    wfn : BaseWavefunction
        Wavefunction whose orbitals are rotated.
    jacobi_indices : 2-tuple of ints
        Orbitals that are rotated.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    param_shape : tuple of int
        Shape of the parameters.
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    template_params : np.ndarray
        Default parameters of the wavefunction.
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Orbital type.

    Methods
    -------
    __init__(self, nelec, nspin, wfn, dtype=None, memory=None, params=None, orbtype=None,
             jacobi_indices=None):
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_dtype(self, dtype)
        Assign the data type of the parameters.
    assign_memory(self, memory=None):
        Assign memory available for the wavefunction.
    assign_params(self, params)
        Assign the orbital transformation matrix.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.

    """
    @property
    def spin(self):
        """Return the spin of the wavefunction.

        If the orbitals are restricted or unrestricted, the spin should be same as the original.
        Otherwise, the orbitals may mix regardless of the spin, the spin of the wavefunction is hard
        to determine.

        Returns
        -------
        spin : float
            Spin of the (composite) wavefunction if the orbitals are restricted or unrestricted.
            None if all spins are allowed.

        """
        if self.orbtype in ['restricted', 'unrestricted']:
            return self.wfn.spin
        else:
            return None

    @property
    def seniority(self):
        """Return the seniority of the wavefunction.

        If the orbitals are restricted or unrestricted, the seniority should be same as the
        original. Otherwise, the orbitals may mix regardless of the seniority, the seniority of the
        wavefunction is hard to determine.

        Returns
        -------
        seniority : int
            Seniority of the (composite) wavefunction if the orbitals are restricted or
            unrestricted.
            None if all seniority are allowed.

        """
        if self.orbtype in ['restricted', 'unrestricted']:
            return self.wfn.seniority
        else:
            return None

    @property
    def template_params(self):
        """Return the template of the parameters of the given wavefunction.

        The orbital transformation matrix is the parameter of this (composite) wavefunction.

        Returns
        -------
        template_params : np.ndarray
            Rotation matrix.

        """
        return (np.eye(self.nspatial, self.wfn.nspatial, dtype=self.dtype), )

    @property
    def nparams(self):
        """Return the number of wavefunction parameters.

        Returns
        -------
        nparams : tuple of int
            Number of elements in each transformation matrix.

        """
        return tuple(i.size for i in self.params)

    @property
    def param_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        param_shape : tuple of 2-tuple of ints
            Shape of each transformation matrix.

        """
        return tuple(i.shape for i in self.params)

    @property
    def orbtype(self):
        """Return the orbital type.

        Returns
        -------
        orbtype : str
            'restricted' if only one transformation matrix is given and the number of rows
            corresponds to the number of spatial orbitals.
            'unrestricted' if two transformation matrices are given and the number of rows
            corresponds to the number of spatial orbitals.
            'restricted' if only one transformation matrix is given and the number of rows
            corresponds to the number of spin orbitals.

        Raises
        ------
        NotImplementedError
            If unsupported transformation matrices are given.

        """
        if len(self.params) == 1 and self.params[0].shape[0] == self.nspatial:
            return 'restricted'
        elif (len(self.params) == 2 and self.params[0].shape[0] == self.nspatial and
              self.params[1].shape[0] == self.nspatial):
            return 'unrestricted'
        elif len(self.params) == 1 and self.params[0].shape[0] == self.nspin:
            return 'generalized'
        else:
            raise NotImplementedError('Unsupported orbital type.')

    def assign_params(self, params=None):
        """Assign the orbital transformation matrix.

        Parameters
        ----------
        params : {np.ndarray, tuple of np.ndarray}
            Transformation matrices.
            If one transformation matrix is given, then the transformation coresponds to those of
            restricted orbitals, where the spatial orbitals are transformed or to those of
            generalized orbitals, where spin orbitals are transformed.
            If two transformation matrices are given, then the transformation corresponds to those
            of unrestricted orbitals, where the spatial orbitals are transformed.
            Default is the template parameters.

        Raises
        ------
        TypeError
            If transformation matrix is not a numpy array or 1- or 2-tuple/list of numpy arrays.
            If transformation matrix is not a two dimension numpy array.
            If transformation matrix is not a two dimension numpy array.
        ValueError
            If transformation matrix does not have the right shape.
        """
        if params is None:
            params = self.template_params

        if isinstance(params, np.ndarray):
            params = (params, )
        elif not (isinstance(params, (tuple, list)) and len(params) in [1, 2]):
            raise TypeError('Transformation matrix must be a two dimensional numpy array or a '
                            '1- or 2-tuple/list of two dimensional numpy arrays. Only one numpy '
                            'arrays indicate that the orbitals are restricted or generalized and '
                            'two numpy arrays indicate that the orbitals are unrestricted.')
        params = tuple(params)

        for i in params:
            if not (isinstance(i, np.ndarray) and len(i.shape) == 2):
                raise TypeError('Transformation matrix must be a two-dimensional numpy array.')
            elif i.dtype != self.dtype:
                raise TypeError('Transformation matrix must have the same data type as the given '
                                'wavefunction.')

            if (len(params) == 1 and
                    not ((i.shape[0] == self.nspatial and i.shape[1] == self.wfn.nspatial) or
                         (i.shape[0] == self.nspin and i.shape[1] == self.wfn.nspin))):
                raise ValueError('Given the type of transformation, the numpy matrix has the '
                                 'wrong shape. If only one numpy array is given, the '
                                 'orbitals are transformed either from orthonormal spatial '
                                 'orbitals to nonorthonormal spatial orbitals or from '
                                 'orthonormal spin orbitals to nonorthonormal spin orbitals.')

            elif (len(params) == 2 and
                  not (i.shape[0] == self.nspatial and i.shape[1] == self.wfn.nspatial)):
                raise ValueError('Given the type of transformation, the numpy matrix has the '
                                 'wrong shape. If two numpy arrays are given, the orbitals are '
                                 'transformed from orthonormal spatial orbitals to nonorthonormal '
                                 'spatial orbitals.')

        self.params = params

    def load_cache(self):
        """Load the functions whose values will be cached.

        To minimize the cache size, the input is made as small as possible. Therefore, the cached
        function is not a method of an instance (because the instance is an input) and the smallest
        representation of the Slater determinant (an integer) is used as the only input. However,
        the functions must access other properties/methods of the instance, so they are defined
        within this method so that the instance is available within the namespace w/o use of
        `global` or `local`.

        Since the bitstring is used to represent the Slater determinant, they need to be processed,
        which may result in repeated processing depending on when the cached function is accessed.

        It is assumed that the cached functions will not be used to calculate redundant results. All
        simplifications that can be made is assumed to have already been made. For example, it is
        assumed that the overlap derivatized with respect to a parameter that is not associated with
        the given Slater determinant will never need to be evaluated because these conditions are
        caught before calling the cached functions.

        Notes
        -----
        Needs to access `memory` and `params`.

        """
        # assign memory allocated to cache
        if self.memory == np.inf:
            memory = None
        else:
            memory = int((self.memory - 5*8*sum(self.nparams)) / (sum(self.nparams) + 1))

        # create function that will be cached
        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp(sd):
            """Calculate the overlap with the Slater determinant.

            Parameters
            ----------
            sd : gmpy2.mpz
                Occupation vector of a Slater determinant given as a bitstring.

            Returns
            -------
            olp : float
                Overlap of the wavefunction with the Slater determinant.

            """
            output = 0.0
            if self.orbtype == 'generalized':
                row_inds = slater.occ_indices(sd)
                all_col_inds = range(self.wfn.nspin)
                for col_inds in it.combinations(all_col_inds, self.nelec):
                    nonorth_sd = slater.create(0, *col_inds)
                    wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                    nonorth_coeff = np.linalg.det(self.params[0][row_inds, :][:, col_inds])
                    output += wfn_coeff * nonorth_coeff
            else:
                alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
                alpha_row_inds = slater.occ_indices(alpha_sd)
                beta_row_inds = slater.occ_indices(beta_sd)
                all_col_inds = range(self.wfn.nspatial)
                for alpha_col_inds in it.combinations(all_col_inds, len(alpha_row_inds)):
                    if len(alpha_col_inds) == 0:
                        alpha_coeff = 1.0
                    else:
                        alpha_coeff = np.linalg.det(self.params[0]
                                                    [alpha_row_inds, :][:, alpha_col_inds])
                    for beta_col_inds in it.combinations(all_col_inds, len(beta_row_inds)):
                        # FIXME: change i+nspatial to slater.to_beta
                        nonorth_sd = slater.create(0, *alpha_col_inds,
                                                   *[i + self.nspatial for i in beta_col_inds])
                        wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                        if self.orbtype == 'restricted':
                            i = 0
                        elif self.orbtype == 'unrestricted':
                            i = 1

                        if len(beta_col_inds) == 0:
                            beta_coeff = 1.0
                        else:
                            beta_coeff = np.linalg.det(self.params[i]
                                                       [beta_row_inds, :][:, beta_col_inds])
                        output += wfn_coeff * alpha_coeff * beta_coeff
            return output

        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp_deriv(sd, deriv):
            """Calculate the derivative of the overlap with the Slater determinant.

            Parameters
            ----------
            sd : gmpy2.mpz
                Occupation vector of a Slater determinant given as a bitstring.
            deriv : int
                Index of the parameter with respect to which the overlap is derivatized.

            Returns
            -------
            olp_deriv : float
                Derivative of the overlap of the wavefunction with the Slater determinant.

            """
            # lots of repetition b/c slight variations with different orbital types
            transform_ind = deriv // self.nparams[0]
            row_removed = ((deriv % self.nparams[0]) // self.param_shape[transform_ind][1])
            col_removed = ((deriv % self.nparams[0]) % self.param_shape[transform_ind][1])

            output = 0.0
            if self.orbtype == 'generalized':
                row_inds = slater.occ_indices(slater.annihilate(sd, row_removed))
                row_sign = (-1) ** np.sum(np.array(row_inds) < row_removed)
                all_col_inds = (i for i in range(self.wfn.nspin) if i != col_removed)
                for col_inds in it.combinations(all_col_inds, len(row_inds)):
                    nonorth_sd = slater.create(0, col_removed, *col_inds)
                    wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                    col_sign = (-1) ** np.sum(np.array(col_inds) < col_removed)
                    nonorth_coeff = np.linalg.det(self.params[0][row_inds, :][:, col_inds])
                    output += wfn_coeff * row_sign * col_sign * nonorth_coeff
                return output

            alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
            if not (slater.occ(alpha_sd, row_removed) or slater.occ(beta_sd, row_removed)):
                return 0.0
            # FIXME/TODO: need to add signature for derivative
            if (self.orbtype == 'restricted' and slater.occ(alpha_sd, row_removed)
                    and slater.occ(beta_sd, row_removed)):
                # if both alpha and beta Slater determinants contain the orbital
                alpha_row_inds = slater.occ_indices(slater.annihilate(alpha_sd, row_removed))
                beta_row_inds = slater.occ_indices(slater.annihilate(beta_sd, row_removed))

                all_col_inds = [i for i in range(self.wfn.nspatial) if i != col_removed]

                # alpha block includes derivatized column
                for alpha_col_inds in it.combinations(all_col_inds, len(alpha_row_inds)):
                    if len(alpha_col_inds) == 0:
                        der_alpha_coeff = 1.0
                    else:
                        der_alpha_coeff = np.linalg.det(self.params[0]
                                                        [alpha_row_inds, :]
                                                        [:, alpha_col_inds])
                    alpha_coeff = np.linalg.det(self.params[0]
                                                [alpha_row_inds+(row_removed, ), :]
                                                [:, alpha_col_inds+(col_removed, )])
                    # beta block includes derivatized column
                    # FIXME: change i+nspatial to slater.to_beta
                    for beta_col_inds in it.combinations(all_col_inds, len(beta_row_inds)):
                        nonorth_sd = slater.create(0, col_removed, col_removed + self.nspatial,
                                                   *alpha_col_inds,
                                                   *(i + self.nspatial for i in beta_col_inds))
                        wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                        if len(beta_col_inds) == 0:
                            der_beta_coeff = 1.0
                        else:
                            der_beta_coeff = np.linalg.det(self.params[0]
                                                           [beta_row_inds, :]
                                                           [:, beta_col_inds])
                        beta_coeff = np.linalg.det(self.params[0]
                                                   [beta_row_inds+(row_removed, ), :]
                                                   [:, beta_col_inds+(col_removed, )])
                        output += wfn_coeff * (der_alpha_coeff * beta_coeff +
                                               alpha_coeff * der_beta_coeff)

                    # beta block does not include derivatized column
                    for beta_col_inds in it.combinations(all_col_inds, len(beta_row_inds) + 1):
                        nonorth_sd = slater.create(0, col_removed, *alpha_col_inds,
                                                   *(i + self.nspatial for i in beta_col_inds))
                        wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                        beta_coeff = np.linalg.det(self.params[0]
                                                   [beta_row_inds+(row_removed,), :]
                                                   [:, beta_col_inds])
                        output += wfn_coeff * der_alpha_coeff * beta_coeff
                # alpha block does not include the derivatized column
                for alpha_col_inds in it.combinations(all_col_inds, len(alpha_row_inds)+1):
                    alpha_coeff = np.linalg.det(self.params[0]
                                                [alpha_row_inds+(row_removed, ), :]
                                                [:, alpha_col_inds])
                    # beta block includes derivatized column
                    for beta_col_inds in it.combinations(all_col_inds, len(beta_row_inds)):
                        nonorth_sd = slater.create(0, col_removed + self.nspatial, *alpha_col_inds,
                                                   *(i + self.nspatial for i in beta_col_inds))
                        wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                        if len(beta_col_inds) == 0:
                            der_beta_coeff = 1.0
                        else:
                            der_beta_coeff = np.linalg.det(self.params[0]
                                                           [beta_row_inds, :][:, beta_col_inds])
                        output += wfn_coeff * alpha_coeff * der_beta_coeff

                return output

            # if only one of alpha and beta Slater determinants contains the orbital
                # elif slater.occ(alpha_sd, row_removed) != slater.occ(beta_sd, row_removed):
            if ((self.orbtype == 'restricted' and slater.occ(alpha_sd, row_removed)) or
                    (self.orbtype == 'unrestricted' and transform_ind == 0)):
                alpha_row_inds = slater.occ_indices(slater.annihilate(alpha_sd, row_removed))
                beta_row_inds = slater.occ_indices(beta_sd)
                row_sign = (-1) ** np.sum(np.array(alpha_row_inds) < row_removed)
                all_alpha_col_inds = (i for i in range(self.wfn.nspatial) if i != col_removed)
                all_beta_col_inds = range(self.wfn.nspatial)
            else:
                alpha_row_inds = slater.occ_indices(alpha_sd)
                beta_row_inds = slater.occ_indices(slater.annihilate(beta_sd, row_removed))
                row_sign = (-1) ** np.sum(np.array(beta_row_inds) < row_removed)
                all_alpha_col_inds = range(self.wfn.nspatial)
                all_beta_col_inds = (i for i in range(self.wfn.nspatial) if i != col_removed)

            for alpha_col_inds in it.combinations(all_alpha_col_inds, len(alpha_row_inds)):
                col_sign = 1
                if transform_ind == 0:
                    col_sign *= (-1) ** np.sum(np.array(alpha_col_inds) < col_removed)
                if len(alpha_col_inds) == 0:
                    alpha_coeff = 1.0
                else:
                    alpha_coeff = np.linalg.det(self.params[0]
                                                [alpha_row_inds, :][:, alpha_col_inds])
                for beta_col_inds in it.combinations(all_beta_col_inds, len(beta_row_inds)):
                    # FIXME: change i+nspatial to slater.to_beta
                    nonorth_sd = slater.create(0, *alpha_col_inds,
                                               *[i + self.nspatial for i in beta_col_inds])
                    if ((self.orbtype == 'restricted' and slater.occ(alpha_sd, row_removed))
                            or (self.orbtype == 'unrestricted' and transform_ind == 0)):
                        nonorth_sd = slater.create(nonorth_sd, col_removed)
                    else:
                        nonorth_sd = slater.create(nonorth_sd, col_removed + self.nspatial)

                    wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                    if transform_ind == 1:
                        col_sign *= (-1) ** np.sum(np.array(beta_col_inds) < col_removed)
                    if len(beta_col_inds) == 0:
                        beta_coeff = 1.0
                    else:
                        orb_ind = 0 if self.orbtype == 'restricted' else 1
                        beta_coeff = np.linalg.det(self.params[orb_ind]
                                                   [beta_row_inds, :][:, beta_col_inds])
                    output += wfn_coeff * row_sign * col_sign * alpha_coeff * beta_coeff
            return output

        # create cache
        if not hasattr(self, '_cache_fns'):
            self._cache_fns = {}

        # store the cached function
        self._cache_fns['overlap'] = _olp
        self._cache_fns['overlap derivative'] = _olp_deriv

    # FIXME: incredibly slow/bad approach
    # TODO: instead of all possible combinations (itertools), have something that selects a smaller
    #       subset
    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with an orthonormal Slater determinant.

        A wavefunction built using nonorthonormal Slater determinants, :math:`\mathbf{n}`, can be
        expressed with respect to orthonormal Slater determinants, :math:`\mathbf{m}`:

        .. math::
            \ket{\Psi}
            &= \sum_{\mathbf{n}} f(\mathbf{n}) \sum_{\mathbf{m}} |U(\mathbf{m}, \mathbf{n})|^+
            \ket{\mathbf{m}}\\
            &= \sum_{\mathbf{m}} \sum_{\mathbf{n}} f(\mathbf{n}) |U(\mathbf{m}, \mathbf{n})|^+
            \ket{\mathbf{m}}

        Then, the overlap with an orthonormal Slater determinant is

        .. math::
            \braket{\Phi_i | \Psi}
            &= \sum_{\mathbf{n}} f(\mathbf{n}) |U(\Phi_i, \mathbf{n})|^+

        where :math:`U(\Phi_i, \mathbf{n})` is the transformation matrix with rows and columns that
        correspond to the Slater determinants :math:`\Phi_i` and :math:`\mathbf{n}`, respectively.

        """
        if deriv is None:
            return self._cache_fns['overlap'](sd)

        # if derivatization
        elif isinstance(deriv, int):
            if deriv >= sum(self.nparams):
                return 0.0

            # get index of the transformation (if unrestricted)
            transform_ind = deriv // self.nparams[0]
            # convert parameter index to row and col index
            row_removed = (deriv % self.nparams[0]) // self.param_shape[transform_ind][1]

            # if either of these orbitals are not present in the Slater determinant, skip
            # FIXME: change i+nspatial to slater.to_beta
            if (self.orbtype == 'restricted' and
                    not (slater.occ(sd, row_removed) or
                         slater.occ(sd, row_removed + self.nspatial))):
                return 0.0
            elif (self.orbtype == 'unrestricted' and
                  not slater.occ(sd, row_removed + transform_ind*self.nspatial)):
                return 0.0
            elif self.orbtype == 'generalized' and not slater.occ(sd, row_removed):
                return 0.0

            return self._cache_fns['overlap derivative'](sd, deriv)
