"""Hard-coded Matrix Product State wavefunction."""
import numpy as np
from wfns.backend import slater
from wfns.wfn.base import BaseWavefunction


# TODO: make subclass of TensorNetworkState
# TODO: use TensorFlow or some other tensor library
class MatrixProductState(BaseWavefunction):
    r"""Matrix Product State wavefunction.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction.
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.

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

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, memory=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_dtype(self, dtype)
        Assign the data type of the parameters.
    assign_memory(self, memory=None)
        Assign memory available for the wavefunction.
    assign_params(self, params)
        Assign parameters of the wavefunction.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.
    assign_dimension(self, dimension=None)
        Assign the dimension of the matrices.
    get_occupation_indices(self, sd) : np.ndarray
        Return the occupation vector of the Slater determinant in the format used in MPS.
    get_matrix_shape(self, index) : 3-tuple of int
        Get the shape of a matrix.
    get_matrix_indices(self, index) : 2-tuple of int
        Get the start and end indices of a matrix.
    get_matrix(self, index) : np.ndarray
        Get the matrix that correspond to the spatial orbital of the given index.
    decompose_index(self, param_index) : 4-tuple of int
        Return the indices of the spatial orbital, occupation, row and column indices.

    """

    def __init__(self, nelec, nspin, dtype=None, memory=None, params=None, dimension=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type.
            Default is `np.float64`.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).

        """
        super().__init__(nelec, nspin, dtype=dtype, memory=memory)
        self.assign_dimension(dimension)
        self.assign_params(params)
        self._cache_fns = {}
        self.load_cache()

    # TODO: all of the auxiliary indices are fixed to be equal. This may need to be flexible
    def assign_dimension(self, dimension=None):
        """Assign the dimension of the parameters of the wavefunction.

        Parameters
        ----------
        dimension : {int, None}
            Dimension of the auxiliary index.
            All of the auxiliary indices will have the same dimension.

        Raises
        ------
        TypeError
            If dimension is not an integer or None.

        Note
        ----
        Needs `nspin` attribute.

        """
        # FIXME: need a better default for dimension
        if dimension is None:
            dimension = self.nspin

        if not isinstance(dimension, int):
            raise TypeError('Provided dimension must be an integer or None.')
        elif dimension <= 0:
            raise ValueError('Dimension of the matrices must be given as a nonnegative integer.')

        self.dimension = dimension

    def get_occupation_indices(self, sd):
        """Return the occupation vector of the Slater determinant in the format used in MPS.

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.

        Returns
        -------
        indices : np.ndarray
            Indices that correspond to the occupation of the Slater determinant.
            Entry of 0 means that the spatial orbital is not occupied.
            Entry of 1 means that the spatial orbital is singly occupied (alpha).
            Entry of 2 means that the spatial orbital is singly occupied (beta).
            Entry of 3 means that the spatial orbital is doubly occupied.

        Note
        ----
        Requires `nspin` attribute.

        """
        sd = slater.internal_sd(sd)
        indices = np.zeros(self.nspatial, dtype=int)
        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
        alpha_occ = list(slater.occ_indices(alpha_sd))
        beta_occ = list(slater.occ_indices(beta_sd))
        indices[alpha_occ] += 1
        indices[beta_occ] += 2
        return indices

    def get_matrix_shape(self, index):
        """Get the shape of a matrix.

        Parameters
        ----------
        index : int
            Index of the spatial orbital that corresponds to the desired matrix.

        Returns
        -------
        shape : tuple of int
            Shape of the selected matrix.

        Raises
        ------
        TypeError
            If index is not an integer
        ValueError
            If index is not greater than or equal to 0 and less than number of spatial orbitals.

        Note
        ----
        Requires attributes `nspin` and `dimension`.

        """
        if not isinstance(index, (int, np.int64)):
            raise TypeError('Given index must be an integer.')
        elif not 0 <= index < self.nspatial:
            raise ValueError('Given index must be greater than or equal to zero or less than K.')

        if index == 0:
            return (4, 1, self.dimension)
        elif index < self.nspatial - 1:
            return (4, self.dimension, self.dimension)
        else:
            return (4, self.dimension, 1)

    def get_matrix_indices(self, index):
        """Get the start and end indices of a matrix.

        Parameters
        ----------
        index : int
            Index of the spatial orbital that corresponds to the desired matrix.

        Returns
        -------
        start_index : int
            Index of the parameter that indicates the start (inclusive) of the matrix.
        end_index : int
            Index of the parameter that indicates the end (exclusive) of the matrix.

        Raises
        ------
        TypeError
            If index is not an integer
        ValueError
            If index is not greater than or equal to 0 and less than number of spatial orbitals.

        """
        size_matrix = np.prod(self.get_matrix_shape(index))
        if index == 0:
            return (0, size_matrix)
        else:
            # NOTE: assumes that matrix of index 1 has the same shape as the subsequent ones (except
            #       last one)
            start = (np.prod(self.get_matrix_shape(0)) +
                     np.prod(self.get_matrix_shape(1)) * (index - 1))
            return (start, start + np.prod(self.get_matrix_shape(index)))

    def get_matrix(self, index):
        """Get the matrix that correspond to the spatial orbital of the given index.

        Parameters
        ----------
        index : int
            Index of the spatial orbital.

        Returns
        -------
        matrix : np.ndarray
            Matrix that corresponds to the given index.

        """
        start_ind, end_ind = self.get_matrix_indices(index)
        matrix_shape = self.get_matrix_shape(index)
        return self.params[start_ind: end_ind].reshape(matrix_shape)

    def decompose_index(self, param_index):
        """Return the indices of the spatial orbital, occupation, row and column indices.

        Parameters
        ----------
        param_index : int
            Index of the parameter.

        Returns
        -------
        ind_spatial : int
            Index of the spatial orbital that corresponds to the parameter.
        ind_occ : int
            Index of the occupation of the spatial orbital that corresponds to the parameter.
        ind_row : int
            Row index of the matrix that correspond to the given spatial orbital and occupation.
        ind_col : int
            Column index of the matrix that correspond to the given spatial orbital and occupation.

        """
        start_index, end_index = self.get_matrix_indices(0)
        if start_index <= param_index < end_index:
            ind_spatial = 0
        else:
            # NOTE: assumes that matrix of index 1 has the same shape as the subsequent ones (except
            #       last one)
            ind_spatial = (param_index - end_index) // np.prod(self.get_matrix_shape(1))
            ind_spatial += 1
            param_index -= self.get_matrix_indices(ind_spatial)[0]

        matrix_shape = self.get_matrix_shape(ind_spatial)
        ind_occ, ind_row, ind_col = np.unravel_index([param_index], matrix_shape)
        return ind_spatial, np.asscalar(ind_occ), np.asscalar(ind_row), np.asscalar(ind_col)

    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of int
            Shape of the parameters.

        """
        return sum(np.prod(self.get_matrix_shape(i)) for i in range(self.nspatial))

    # TODO: the parameters can probably be changed to something more elegant (like a tensor object
    #       or something)
    @property
    def template_params(self):
        """Return the template of the parameters of the MPS wavefunction.

        Returns
        -------
        template_params : np.ndarray
            Default parameters for the MPS wavefunction.

        Notes
        -----
        Depends on attribute `dimension`.

        """
        # NOTE: assumes that matrix of index 1 has the same shape as the subsequent ones (except
        #       last one)
        #       assumes that there are K matrices
        matrices = [np.zeros(self.get_matrix_shape(0), dtype=self.dtype)]
        matrices += [np.zeros(self.get_matrix_shape(1), dtype=self.dtype)
                     for i in range(self.nspatial - 2)]
        matrices += [np.zeros(self.get_matrix_shape(self.nspatial-1), dtype=self.dtype)]

        ground_sd = slater.ground(self.nelec, self.nspin)
        occ_indices = self.get_occupation_indices(ground_sd)

        # adjust scale so that the overlap is 1
        scale = self.dimension ** (-1 / occ_indices.size)
        # set the first matrix
        matrices[0][occ_indices[0], 0, :] = scale
        # set the middle matrices
        for i, ind in enumerate(occ_indices[1:-1]):
            # select the diagonal
            diag_indices = np.arange(self.dimension)
            matrices[i+1][ind, diag_indices, diag_indices] = scale
        # set the last matrix
        matrices[-1][occ_indices[-1], :, 0] = scale
        # flatten and join together
        return np.hstack([matrix.flatten() for matrix in matrices])

    # TODO: the MPS structure may need to become compatible with the TNS in the future (and vice
    #       versa)
    # FIXME: move to the parent class
    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the MPS wavefunction.

        Parameters
        ----------
        params : {np.ndarray, MatrixProductState, None}
            Parameters of the MPS wavefunction.
            If MatrixProductState instance is given, then the parameters of this instance are used.
            Default uses the template parameters.
        add_noise : bool
            Flag to add noise to the given parameters.

        Raises
        ------
        TypeError
            If `params` is not a numpy array.
            If `params` does not have data type of `float`, `complex`, `np.float64` and
            `np.complex128`.
            If `params` has complex data type and wavefunction has float data type.
        ValueError
            If `params` does not have the same shape as the template_params.
            If given MatrixProductState instance does not correspond to the provided dimension.

        Notes
        -----
        Depends on dtype, template_params, and nparams.

        """
        # FIXME: move this part to the base wavefunction
        if isinstance(params, MatrixProductState):
            params = params.params
        super().assign_params(params=params, add_noise=add_noise)

    def _olp(self, sd):
        """Calculate the overlap with the Slater determinant.

        Parameters
        ----------
        sd : gmpy2.mpz
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.

        Returns
        -------
        olp : {float, complex}
            Overlap of the current instance with the given Slater determinant.

        """
        occ_indices = self.get_occupation_indices(sd)

        temp_matrix = self.get_matrix(0)[occ_indices[0], :, :]
        for i in range(1, occ_indices.size):
            temp_matrix = temp_matrix.dot(self.get_matrix(i)[occ_indices[i], :, :])
        return np.asscalar(temp_matrix)

    def _olp_deriv(self, sd, deriv):
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : gmpy2.mpz
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.
        deriv : int
            Index of the parameter with respect to which the overlap is derivatized.
            Assumed to correspond to the matrix elements that correspond to the given Slater
            determinant.

        Returns
        -------
        olp : {float, complex}
            Derivative of the overlap with respect to the given parameter.

        """
        deriv_matrix, _, deriv_row, deriv_col = self.decompose_index(deriv)

        occ_indices = self.get_occupation_indices(sd)

        left_temp, right_temp = 1.0, 1.0

        # if selected matrix is not the right most matrix
        if deriv_matrix < occ_indices.size - 1:
            # index of the matrix to the right of the given matrix
            index_right = deriv_matrix + 1

            # multiply matrix towards the right (from the derivatized matrix)
            right_temp = self.get_matrix(index_right)[occ_indices[index_right], deriv_col, :]
            for i in range(index_right + 1, occ_indices.size):
                right_temp = right_temp.dot(self.get_matrix(i)[occ_indices[i], :, :])

        # if selected matrix is not the left most matrix
        if deriv_matrix > 0:
            # index of the matrix to the left of the given matrix
            index_left = deriv_matrix - 1

            # multiply matrix towards the left (from the derivatized matrix)
            left_temp = self.get_matrix(index_left)[occ_indices[index_left], :, deriv_row]
            for i in reversed(range(index_left)):
                left_temp = self.get_matrix(i)[occ_indices[i], :, :].dot(left_temp)

        return np.asscalar(left_temp * right_temp)

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : int
            Index of the parameter to derivatize.
            Default does not derivatize.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        Raises
        ------
        TypeError
            If given Slater determinant is not compatible with the format used internally.

        """
        sd = slater.internal_sd(sd)

        # if no derivatization
        if deriv is None:
            return self._cache_fns['overlap'](sd)
        # if derivatization
        elif not isinstance(deriv, (int, np.int64)):
            raise TypeError('Given derivatization index must be an integer.')

        if not 0 <= deriv < self.nparams:
            return 0.0

        occ_indices = self.get_occupation_indices(sd)
        deriv_matrix, deriv_occ, *_ = self.decompose_index(deriv)
        if deriv_occ != occ_indices[deriv_matrix]:
            return 0.0

        return self._cache_fns['overlap derivative'](sd, deriv)
