"""Parent class for any objects with parameters."""
import abc
import numpy as np


class ParamContainer(abc.ABC):
    """Parameters that will something.

    Attributes
    ----------
    params : np.ndarray
        Parameters.

    """
    def __init__(self, params):
        """Initialize.

        Parameters
        ----------
        params : np.ndarray
            Parameters.

        """
        self.assign_params(params)

    def nparams(self):
        """Return the number of parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        return self.params.size

    def assign_params(self, params):
        """Assign parameters.

        Parameters
        ----------
        params : {int, float, complex, np.float64, np.complex128, np.ndarray}
            Parameters.

        Raises
        ------
        TypeError
            If given parameters are not given as a numpy array.
            If given parameters does not have data type of `int`, `float`, `complex`, `np.float64`
            or `np.complex128`.

        """
        if isinstance(params, (int, float, complex, np.float64, np.complex128)):
            params = np.array(params)

        if not isinstance(np.ndarray, params):
            raise TypeError('Parameters must be given as a numpy array.')
        elif params.dtype not in (int, float, complex, np.float64, np.complex128):
            raise TypeError('Parameters must have data type of `int`, `float`, `complex`, '
                            '`np.float64` and `np.complex128`.')
        self.params = params
