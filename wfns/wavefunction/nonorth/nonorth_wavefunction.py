r"""Wavefunction with nonorthonormal orbitals.

A parameterized multideterminantal wavefunction can be written as

.. math::
    \ket{\Psi} = \sum_{\mathbf{m}} f(\mathbf{m}) \ket{\mathbf{m}}

where :math:`\ket{\mathbf{m}}` is a Slater determinant. If the Slater determinants are constructed
from nonorthonormal orbitals, then each Slater determinant can be expressed as a linear combination
of Slater determinants constructed from orthonormal orbitals.

.. math::
    \ket{\Psi}
    &= \sum_{\mathbf{n}} f(\mathbf{n}) \ket{\mathbf{n}}\\
    &= \sum_{\mathbf{n}} f(\mathbf{n}) \sum_{\mathbf{m}}
    |C(\mathbf{n}, \mathbf{m})|^- \ket{\mathbf{m}}\\
    &= \sum_{\mathbf{n}} \sum_{\mathbf{m}}
    f(\mathbf{n}) |C(\mathbf{n}, \mathbf{m})|^- \ket{\mathbf{m}}

where :math:`\ket{\mathbf{m}}` and :math:`\ket{\mathbf{n}}` are Slater determinants constructed from
orthonormal and nonorthonormal orbitals.
The nonorthonormal orbitals are constructed by linearly transforming the orbitals of
:math:`\ket{\mathbf{m}}` with :math:`C`.
The :math:`C(\mathbf{n}, \mathbf{m})` is a submatrix of :math:`C` where rows are selected according
to :math:`\ket{\mathbf{n}}` and columns to :math:`\ket{\mathbf{m}}`.
"""
from __future__ import absolute_import, division, print_function
import itertools as it
import functools
import numpy as np
from wfns.backend import slater
from wfns.wavefunction.base_wavefunction import BaseWavefunction
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction

__all__ = []


# FIXME: needs refactoring
class NonorthWavefunction(BaseWavefunction):
    r"""Wavefunction with nonorthonormal orbitals expressed with respect to orthonormal orbitals.

    Nonorthonormal orbitals are expressed with respect to orthonormal orbitals.

    Attributes
    ----------
    nelec : int
        Number of electrons
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction
    memory : float
        Memory available for the wavefunction
    nspin : int
        Number of orthonormal spin orbitals (alpha and beta)

    Properties
    ----------
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters
    nspatial : int
        Number of orthonormal spatial orbitals
    spin : float, None
        Spin of the wavefunction
        :math:`\frac{1}{2}(N_\alpha - N_\beta)` (Note that spin can be negative)
        None means that all spins are allowed
    seniority : int, None
        Seniority (number of unpaired electrons) of the wavefunction
        None means that all seniority is allowed
    template_params : np.ndarray
        Template of the wavefunction parameters
        Depends on the attributes given

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, memory=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_nspin(self, nspin)
        Assigns the number of spin orbitals
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_memory(self, memory=None)
        Assigns the memory allocated for the wavefunction
    assign_params(self, params)
        Assigns the parameters of the wavefunction
    get_overlap(self, sd, deriv=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    """
    def __init__(self, nelec, nspin, dtype=None, memory=None, wfn=None, orth_to_nonorth=None):
        r"""Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons
        nspin : int
            Number of spin orbitals
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type
            Default is `np.float64`
        memory : {float, int, str, None}
            Memory available for the wavefunction
            Default does not limit memory usage (i.e. infinite)
        wfn : BaseWavefunction
            Wavefunction that will be built up using nonorthnormal orbitals
        orth_to_nonorth : np.ndarray
            Transformation matrix from orthonormal orbitals to nonorthonormal orbitals
            :math:`\ket{\tilde{\phi}_i} = \sum_{j} \ket{\phi_j} T_{ji}`
            Default is an identity matrix.
        """
        super().__init__(nelec, nspin, dtype=dtype, memory=memory)
        self.assign_wfn(wfn)
        self.assign_params(orth_to_nonorth)

    @property
    def spin(self):
        """Spin of the wavefunction.

        Since the orbitals may mix regardless of the spin, the spin of the wavefunction is hard to
        determine.

        Returns
        -------
        spin
            Spin of the (composite) wavefunction if the orbitals are restricted or unrestricted
            None if the orbital is generalized
        """
        if self.orbtype in ['restricted', 'unrestricted']:
            return self.wfn.spin
        else:
            return None

    @property
    def seniority(self):
        """Seniority of the wavefunction."""
        if self.orbtype in ['restricted', 'unrestricted']:
            return self.wfn.seniority
        else:
            return None

    @property
    def template_params(self):
        """Return the shape of the wavefunction parameters.

        The orbital transformations are the wavefunction parameters.
        """
        return (np.eye(self.nspatial, self.wfn.nspatial, dtype=self.dtype), )

    @property
    def nparams(self):
        """Return the number of wavefunction parameters."""
        return tuple(i.size for i in self.params)

    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters."""
        return tuple(i.shape for i in self.params)

    @property
    def orbtype(self):
        """Return the orbital type."""
        if len(self.params) == 1 and self.params[0].shape[0] == self.nspatial:
            return 'restricted'
        elif len(self.params) == 2:
            return 'unrestricted'
        elif len(self.params) == 1 and self.params[0].shape[0] == self.nspin:
            return 'generalized'
        else:
            raise NotImplementedError('Unsupported orbital type.')

    def assign_wfn(self, wfn=None):
        """Assign the wavefunction.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction that will be built up using nonorthnormal orbitals

        Raises
        ------
        ValueError
            If the given wavefunction is not an instance of BaseWavefunction
            If the given wavefunction does not have the same number of electrons as the instantiated
            NonorthWavefunction
            If the given wavefunction does not have the same data type as the instantiated
            NonorthWavefunction.
            If the given wavefunction does not have the same memory as the instantiated
            NonorthWavefunction.
        """
        if wfn is None:
            wfn = CIWavefunction(self.nelec, self.nspin, dtype=self.dtype, memory=self.memory)
        elif not isinstance(wfn, BaseWavefunction):
            raise ValueError('Given wavefunction must be an instance of BaseWavefunction (or its'
                             ' child).')

        if wfn.nelec != self.nelec:
            raise ValueError('Given wavefunction does not have the same number of electrons as the'
                             ' the instantiated NonorthWavefunction.')
        elif wfn.dtype != self.dtype:
            raise ValueError('Given wavefunction does not have the same data type as the '
                             'instantiated NonorthWavefunction.')
        elif wfn.memory != self.memory:
            raise ValueError('Given wavefunction does not have the same memory as the '
                             'instantiated NonorthWavefunction.')
        self.wfn = wfn

    def assign_params(self, params=None):
        """Assign the orbital transformation matrix.

        Parameters
        ----------
        params : np.ndarray
            Transformation matrix

        Raises
        ------
        TypeError
            If transformation matrix is not a numpy array or 1- or 2-tuple/list of numpy arrays
            If transformation matrix is not a two dimension numpy array
            If transformation matrix is not a two dimension numpy array
        """
        if params is None:
            params = self.template_params

        if isinstance(params, np.ndarray):
            params = (params, )
        elif not (isinstance(params, (tuple, list)) and len(params) in [1, 2]):
            # FIXME: set orbtype? right now, the orbital type depends on the number of
            #        transformation matrices and they must match up with the hamiltonian
            # NOTE: maybe it's not a problem b/c orbital transformation is the only part that'll
            #       touch the different types of orbitals
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

        Parameters
        ----------
        sd : int, mpz
            Slater Determinant against which to project.
        deriv : int
            Index of the parameter to derivatize
            Default does not derivatize

        Returns
        -------
        overlap : float

        Raises
        ------
        TypeError
            If given Slater determinant is not compatible with the format used internally
        """
        if deriv is None:
            # if cached function has not been created yet
            if 'overlap' not in self._cache_fns:
                # assign memory allocated to cache
                if self.memory == np.inf:
                    memory = None
                else:
                    memory = int((self.memory - 5*8*sum(self.nparams)) / (sum(self.nparams) + 1))

                # create function that will be cached
                @functools.lru_cache(maxsize=memory, typed=False)
                def _olp(sd):
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
                                                           *[i + self.nspatial
                                                             for i in beta_col_inds])
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

                # store the cached function
                self._cache_fns['overlap'] = _olp
            # if cached function already exists
            else:
                # reload cached function
                _olp = self._cache_fns['overlap']

            return _olp(sd)

        # if derivatization
        elif isinstance(deriv, int):
            if deriv >= sum(self.nparams):
                return 0.0

            # get index of the transformation (if unrestricted)
            transform_ind = deriv // self.nparams[0]
            # convert parameter index to row and col index
            row_removed = (deriv % self.nparams[0]) // self.params_shape[transform_ind][1]

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

            # if cached function has not been created yet
            if 'overlap derivative' not in self._cache_fns:
                # assign memory allocated to cache
                if self.memory == np.inf:
                    memory = None
                else:
                    memory = int((self.memory - 5*8*sum(self.nparams))
                                 / (sum(self.nparams) + 1) * sum(self.nparams))

                # FIXME: giant fucking mess. Lot of repeated code. Messy, inconsistent namespace.
                @functools.lru_cache(maxsize=memory, typed=False)
                def _olp_deriv(sd, deriv):
                    # lots of repetition b/c slight variations with different orbital types
                    transform_ind = deriv // self.nparams[0]
                    row_removed = ((deriv % self.nparams[0]) // self.params_shape[transform_ind][1])
                    col_removed = ((deriv % self.nparams[0]) % self.params_shape[transform_ind][1])

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
                        alpha_row_inds = slater.occ_indices(slater.annihilate(alpha_sd,
                                                                              row_removed))
                        beta_row_inds = slater.occ_indices(slater.annihilate(beta_sd,
                                                                             row_removed))

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
                                nonorth_sd = slater.create(0, col_removed,
                                                           col_removed + self.nspatial,
                                                           *alpha_col_inds,
                                                           *(i + self.nspatial
                                                             for i in beta_col_inds))
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
                            for beta_col_inds in it.combinations(all_col_inds,
                                                                 len(beta_row_inds) + 1):
                                nonorth_sd = slater.create(0, col_removed,
                                                           *alpha_col_inds,
                                                           *(i + self.nspatial
                                                             for i in beta_col_inds))
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
                                nonorth_sd = slater.create(0, col_removed + self.nspatial,
                                                           *alpha_col_inds,
                                                           *(i + self.nspatial
                                                             for i in beta_col_inds))
                                wfn_coeff = self.wfn.get_overlap(nonorth_sd, deriv=None)
                                if len(beta_col_inds) == 0:
                                    der_beta_coeff = 1.0
                                else:
                                    der_beta_coeff = np.linalg.det(self.params[0]
                                                                   [beta_row_inds, :]
                                                                   [:, beta_col_inds])
                                output += wfn_coeff * alpha_coeff * der_beta_coeff

                        return output

                    # if only one of alpha and beta Slater determinants contains the orbital
                        # elif slater.occ(alpha_sd, row_removed) != slater.occ(beta_sd, row_removed):
                    if ((self.orbtype == 'restricted' and slater.occ(alpha_sd, row_removed)) or
                            (self.orbtype == 'unrestricted' and transform_ind == 0)):
                        alpha_row_inds = slater.occ_indices(slater.annihilate(alpha_sd,
                                                                              row_removed))
                        beta_row_inds = slater.occ_indices(beta_sd)
                        row_sign = (-1) ** np.sum(np.array(alpha_row_inds) < row_removed)
                        all_alpha_col_inds = (i for i in range(self.wfn.nspatial)
                                              if i != col_removed)
                        all_beta_col_inds = range(self.wfn.nspatial)
                    else:
                        alpha_row_inds = slater.occ_indices(alpha_sd)
                        beta_row_inds = slater.occ_indices(slater.annihilate(beta_sd,
                                                                             row_removed))
                        row_sign = (-1) ** np.sum(np.array(beta_row_inds) < row_removed)
                        all_alpha_col_inds = range(self.wfn.nspatial)
                        all_beta_col_inds = (i for i in range(self.wfn.nspatial)
                                             if i != col_removed)

                    for alpha_col_inds in it.combinations(all_alpha_col_inds, len(alpha_row_inds)):
                        col_sign = 1
                        if transform_ind == 0:
                            col_sign *= (-1) ** np.sum(np.array(alpha_col_inds) < col_removed)
                        if len(alpha_col_inds) == 0:
                            alpha_coeff = 1.0
                        else:
                            alpha_coeff = np.linalg.det(self.params[0]
                                                        [alpha_row_inds, :][:, alpha_col_inds])
                        for beta_col_inds in it.combinations(all_beta_col_inds,
                                                             len(beta_row_inds)):
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

                # store the cached function
                self._cache_fns['overlap derivative'] = _olp_deriv
            # if cached function already exists
            else:
                # reload cached function
                _olp_deriv = self._cache_fns['overlap derivative']

            return _olp_deriv(sd, deriv)
