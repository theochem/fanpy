"""Parent class of the wavefunctions.

Contains information that all wavefunctions should have (probably?)
"""
from __future__ import absolute_import, division, print_function
import abc
import six
import numpy as np

__all__ = []


@six.add_metaclass(abc.ABCMeta)
class BaseWavefunction:
    """Base wavefunction class.

    Contains the necessary information to solve the wavefunction

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

    Properties
    ----------
    nspatial : int
        Number of spatial orbitals
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters

    Methods
    -------
    __init__(self, nelec, one_int, two_int, dtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_nspin(self, nspin)
        Assigns the number of spin orbitals
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_params(self, params)
        Assigns the parameters of the wavefunction

    Abstract Properties
    -------------------
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

    Abstract Methods
    ----------------
    get_overlap(self, sd, deriv=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    """
    def __init__(self, nelec, nspin, dtype=None, memory=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons
        nspin : int
            Number of spin orbitals
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type
            Default is `np.float64`
        """
        self.assign_nelec(nelec)
        self.assign_nspin(nspin)
        self.assign_dtype(dtype)
        # assign_params not included because it depends on template_params, which may involve
        # more attributes than is given above

    @property
    def nspatial(self):
        """Return the number of spatial orbitals."""
        return self.nspin // 2

    @property
    def nparams(self):
        """Return the number of wavefunction parameters."""
        return self.template_params.size

    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters."""
        return self.template_params.shape

    def assign_nelec(self, nelec):
        """Set the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons

        Raises
        ------
        TypeError
            If number of electrons is not an integer
        ValueError
            If number of electrons is not a positive number
        """
        if not isinstance(nelec, int):
            raise TypeError('Number of electrons must be an integer')
        elif nelec <= 0:
            raise ValueError('Number of electrons must be a positive integer')
        self.nelec = nelec

    def assign_nspin(self, nspin):
        """Set the number of spin orbitals.

        Parameters
        ----------
        nspin : int
            Number of spin orbitals

        Raises
        ------
        TypeError
            If number of spin orbitals is not an integer
        ValueError
            If number of spin orbitals is not a positive number
        NotImplementedError
            If number of spin orbitals is odd
        """
        if not isinstance(nspin, int):
            raise TypeError('Number of spin orbitals must be an integer.')
        elif nspin <= 0:
            raise ValueError('Number of spin orbitals must be a positive integer.')
        elif nspin % 2 == 1:
            raise NotImplementedError('Odd number of spin orbitals is not supported.')
        self.nspin = nspin

    def assign_dtype(self, dtype):
        """Set the data type of the parameters.

        Parameters
        ----------
        dtype : {float, complex, np.float64, np.complex128}
            Numpy data type
            If None then set to np.float64

        Raises
        ------
        TypeError
            If dtype is not one of float, complex, np.float64, np.complex128
        """

        if dtype is None or dtype in (float, np.float64):
            self.dtype = np.float64
        elif dtype in (complex, np.complex128):
            self.dtype = np.complex128
        else:
            raise TypeError("dtype must be float or complex")

    def assign_params(self, params=None, add_noise=False):
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

        Note
        ----
        Depends on dtype, template_params, and nparams
        """
        if params is None:
            params = self.template_params

        # check input
        if not isinstance(params, np.ndarray):
            raise TypeError('Parameters must be given as a np.ndarray')
        elif params.shape != self.params_shape:
            raise ValueError('Parameters must same shape as the template, {0}'
                             ''.format(self.params_shape))
        elif params.dtype not in (float, complex, np.float64, np.complex128):
            raise TypeError('Data type of the parameters must be one of `float`, `complex`, '
                            '`np.float64` and `np.complex128`')
        elif params.dtype in (complex, np.complex128) and self.dtype in (float, np.float64):
            raise TypeError('If the parameters are `complex`, then the `dtype` of the wavefunction '
                            'must be `np.complex128`')

        self.params = params.astype(self.dtype)

        # add random noise
        if add_noise:
            # set scale
            scale = 0.2 / self.nparams
            self.params += scale * (np.random.rand(*self.params_shape) - 0.5)
            if self.dtype in [complex, np.complex128]:
                self.params += 0.01j * scale * (np.random.rand(*self.params_shape).astype(complex)
                                                - 0.5)

        # create cached functions
        self._cache_fns = {}

    def clear_cache(self, key=None):
        """Clear the cache associated with the wavefunction.

        Parameters
        ----------
        key : str
            Key to the cached function in _cache_fns
            Default clears all cached functions

        Raises
        ------
        KeyError
            If given key is not present in the _cache_fns
        ValueError
            If cached function does not have decorator functools.lru_cache
        """
        try:
            if key is None:
                for fn in self._cache_fns:
                    print(fn)
                    fn.clear_cache()
            else:
                self._cache_fns[key].clear_cache()
        except KeyError as error:
            raise KeyError('Given function key is not present in _cache_fns') from error
        except AttributeError as error:
            raise AttributeError('Given cached function does not have decorator '
                                 '`functools.lru_cache`')

    @abc.abstractproperty
    def spin(self):
        """Return the spin of the wavefunction.

        ..math::
            \frac{1}{2}(N_\alpha - N_\beta)

        Returns
        -------
        float

        Note
        ----
        `None` means that all possible spins are allowed
        """
        pass

    @abc.abstractproperty
    def seniority(self):
        """Return the seniority (number of unpaired electrons) of the wavefunction.

        Returns
        -------
        int

        Note
        ----
        `None` means that all possible seniority are allowed
        """
        pass

    @abc.abstractproperty
    def template_params(self):
        """Return the template of the parameters of the given wavefunction.

        Returns
        -------
        np.ndarray

        Note
        ----
        May depend on other attributes or properties
        """
        pass

    @abc.abstractmethod
    def get_overlap(self, sd, deriv=None):
        """Return the overlap of the wavefunction with a Slater determinant.

        ..math::
            \big< \Phi_i \big| \Psi \big>

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
        pass
