"""Hard-coded determinant-ratio wavefunction."""
import functools
import numpy as np
from wfns.backend import slater
from wfns.wfn.base import BaseWavefunction


class DeterminantRatio(BaseWavefunction):
    r"""Wavefunction whose overlap is a ratio of determinants.

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
    numerator_mask : np.ndarray
        Mask for selecting the matrices that correspond to the numerator.
    params : np.ndarray
        Parameters for each matrix have been flattened, and joined together.
        Each matrix is assumed to have shape, (nelec, nspin).

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
    matrix_shape : 2-tuple of int
        Shape of each matrix.
    matrix_size : int
        Size of each matrix.
    num_matrices : int
        Number of matrices.
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
        Loaad the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.

    """
    def __init__(self, nelec, nspin, dtype=None, memory=None, numerator_mask=None, params=None):
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
        numerator_mask : {np.ndarray, None}
            Mask for selecting the matrices that correspond to the numerator.
            Controls the number of matrices used in the wavefunction.
        params : {np.ndarray, None}
            Parameters for each matrix have been flattened, and joined together.
            Each matrix is assumed to have shape, (nelec, nspin).

        """
        super().__init__(nelec, nspin, dtype=dtype, memory=memory)
        self.assign_numerator_mask(numerator_mask)
        self.assign_params(params)
        self.load_cache()

    def assign_numerator_mask(self, numerator_mask=None):
        """Assign the mask for selecting matrices that correspond to the numerator.

        Parameters
        ----------
        numerator_mask : {np.ndarray, None}
            Boolean mask for selecting the matrices that correspond to the numerator.
            Default has two matrices, first for the numerator and second for the demominator.
            Controls the number of matrices used in the wavefunction.

        Raises
        ------
        TypeError
            If numerator_mask is not a boolean numpy array with one or more element.

        """
        if numerator_mask is None:
            numerator_mask = np.array([True, False])

        if not (isinstance(numerator_mask, np.ndarray) and
                numerator_mask.dtype == bool and numerator_mask.size > 0):
            raise TypeError('Mask for the numerator must be given as a boolean numpy array with '
                            'one or more element.')

        self.numerator_mask = numerator_mask

    @property
    def matrix_shape(self):
        """Return the shape of each matrix.

        Returns
        -------
        matrix_shape : 2-tuple of int
            Shape of the matrix.
            Assumes each matrix has the same shape.

        """
        return (self.nelec, self.nspin)

    @property
    def matrix_size(self):
        """Return the size of each matrix.

        Returns
        -------
        matrix_size : int
            Size of the matrix.
            Assumes each matrix has the same shape.

        """
        nrow, ncol = self.matrix_shape
        return nrow * ncol

    @property
    def num_matrices(self):
        """Return the number of matrices.

        Returns
        -------
        num_matrices : int
            Number of matrices.

        """
        return self.numerator_mask.size

    # NOTE: all of the rows are assumed to be selected when the columns are selected.
    # TODO: select different columns depending on the matrix whose columns are selected
    def get_columns(self, sd, index):
        """Get the columns that correspond to the given Slater determinant for the given matrix.

        Parameters
        ----------
        sd : gmpy2.mpz
            Occupation vector of a Slater determinant given as a bitstring.
        index : int
            Index of the selected matrix.

        Returns
        -------
        col_indices : np.ndarray of int
            Indices of the columns that are associated with the given Slater determinant and the
            selected matrix.

        """
        return np.array(slater.occ_indices(sd))

    def get_matrix(self, index):
        """Get the matrix that correspond to the given index.

        Parameters
        ----------
        index : int
            Index of the selected matrix.

        Returns
        -------
        matrix : np.ndarray
            Selected matrix.

        Raises
        ------
        TypeError
            If given index is not an integer.
        ValueError
            If given index is not greater than or equal to zero and less than number of matrices.

        """
        if not isinstance(index, int):
            raise TypeError('index must be given as an integer.')
        elif not 0 <= index < self.num_matrices:
            raise ValueError('index must be greater than or equal to zero and less than the number '
                             'of matrices.')

        flat_matrix = self.params[index * self.matrix_size: (index + 1) * self.matrix_size]
        return flat_matrix.reshape(self.matrix_shape)

    def decompose_index(self, param_index):
        """Return the matrix, row and column indices that correspond to the given parameter index.

        Parameters
        ----------
        param_index : int
            Index of the parameter.

        Returns
        -------
        ind_matrix : int
            Index of the matrix that corresponds to the parameter.
        ind_row : int
            Row index of the matrix.
        ind_col : int
            Column index of the matrix.

        """
        _, ncol = self.matrix_shape
        ind_matrix = param_index // self.matrix_size
        param_index = param_index % self.matrix_size
        ind_row = param_index // ncol
        ind_col = param_index % ncol
        return ind_matrix, ind_row, ind_col

    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of int
            Shape of the parameters.

        """
        return (np.prod(self.matrix_shape) * self.num_matrices, )

    # TODO: the parameters can probably be changed to something more elegant (dimension of the
    #       matrice to be different?)
    @property
    def template_params(self):
        """Return the template of the parameters of the wavefunction.

        Returns
        -------
        template_params : np.ndarray
            Default parameters for the wavefunction.

        Notes
        -----
        Depends on attribute `dimension`.

        """
        # get ground slater determinant
        ground_sd = slater.ground(*self.matrix_shape)

        # make matrices
        matrix = np.zeros(self.matrix_shape, dtype=self.dtype)
        # NOTE: assume that same columns are selected for all matrices
        # NOTE: all of the rows are assumed to be selected when the columns are selected.
        matrix[np.arange(self.matrix_shape[0]), self.get_columns(ground_sd, 0)] = 1

        # flatten and join together
        return np.hstack([matrix.flatten()] * self.num_matrices)

    # TODO: each matrix is assumed to have the same shape (nelec, nspin). It may need to be made
    #       more flexible (let them have different shape or let each matrix have different shapes
    #       from one another)
    # FIXME: can probably removed because it doesn't do much different from the parent assign_param
    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, DeterminantRatio, None}
            Parameters of the DeterminantRatio wavefunction.
            If DeterminantRatio instance is given, then the parameters of this instance are used.
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
            If given DeterminantRatio instance does not correspond to the provided dimension.

        Notes
        -----
        Depends on dtype, nelec, nspin, and numerator_mask.

        """
        # FIXME: move this part to the base wavefunction
        if isinstance(params, DeterminantRatio):
            params = params.params
        super().assign_params(params=params, add_noise=add_noise)

    # FIXME: this method can be moved to the parent class
    def load_cache(self):
        """Load the overlaps of the wavefunction.

        Notes
        -----
        Needs to access `memory` and `params`.

        """
        # assign memory allocated to cache
        if self.memory == np.inf:
            memory = None
        else:
            memory = int((self.memory - 5*8*self.params.size) / (self.params.size + 1))

        # create function that will be cached
        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp(sd):
            """cached _olp method without caching the instance."""
            return self._olp(sd)

        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp_deriv(sd, deriv):
            """cached _olp_deriv method without caching the instance."""
            return self._olp_deriv(sd, deriv)

        # create cache
        if not hasattr(self, '_cache_fns'):
            self._cache_fns = {}

        # store the cached function
        self._cache_fns['overlap'] = _olp
        self._cache_fns['overlap derivative'] = _olp_deriv

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
        # NOTE: all of the rows are assumed to be selected when the columns are selected.
        determinants = np.array([np.linalg.det(self.get_matrix(i)[:, self.get_columns(sd, i)])
                                 for i in range(self.num_matrices)])
        numerator = np.prod(determinants[self.numerator_mask])
        denominator = np.prod(determinants[np.logical_not(self.numerator_mask)])
        return numerator / denominator

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
        deriv_matrix, deriv_row, deriv_col = self.decompose_index(deriv)

        # compute determinants of matrices that are not being derivatized
        # NOTE: all of the rows are assumed to be selected when the columns are selected.
        # ASSUME: selected matrix is square (i.e. has same number of electrons)
        determinants = np.array([np.linalg.det(self.get_matrix(i)[:, self.get_columns(sd, i)])
                                 for i in range(self.num_matrices) if i != deriv_matrix])
        new_numerator_mask = np.delete(self.numerator_mask, deriv_matrix)
        numerator = np.prod(determinants[new_numerator_mask])
        denominator = np.prod(determinants[np.logical_not(new_numerator_mask)])

        # derivatize selected matrix
        # NOTE: all of the rows are assumed to be selected when the columns are selected.
        # ASSUME: deriv_row is in rows and deriv_col is in cols (i.e. selected index corresponds to
        #         an occupied orbital of the given Slater determinant)
        matrix = self.get_matrix(deriv_matrix)
        rows = np.arange(matrix.shape[0])
        cols = self.get_columns(sd, deriv_matrix)
        # mask for finding the deriv_row and deriv_col from rows and cols, respectively
        rows_mask = rows != deriv_row
        cols_mask = cols != deriv_col
        # find sign that corresponds to the derivative
        sign_row = (-1) ** np.asscalar(np.where(np.logical_not(rows_mask))[0])
        sign_col = (-1) ** np.asscalar(np.where(np.logical_not(cols_mask))[0])
        # filter out the deriv_row and deriv_col
        rows = rows[rows_mask]
        cols = cols[cols_mask]

        minor = np.linalg.det(matrix[rows[:, None], cols[None, :]])
        deriv_determinant = sign_row * sign_col * minor

        # if derivatized matrix is a numerator
        if self.numerator_mask[deriv_matrix]:
            return deriv_determinant * numerator / denominator
        # if derivatized matrix is a denominator
        else:
            old_determinant = np.linalg.det(matrix[:, self.get_columns(sd, deriv_matrix)])
            return numerator / denominator * (-1) * old_determinant**(-2) * deriv_determinant

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

        if slater.total_occ(sd) != self.nelec:
            return 0.0

        # if no derivatization
        if deriv is None:
            return self._cache_fns['overlap'](sd)

        # if derivatization
        elif not isinstance(deriv, (int, np.int64)):
            raise TypeError('Given derivatization index must be an integer.')

        if not 0 <= deriv < self.nparams:
            return 0.0

        deriv_matrix, _, deriv_col = self.decompose_index(deriv)
        columns = [self.get_columns(sd, i) for i in range(self.num_matrices)]
        if deriv_col not in columns[deriv_matrix]:
            return 0.0

        return self._cache_fns['overlap derivative'](sd, deriv)
