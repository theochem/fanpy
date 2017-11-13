"""Parent class of the wavefunctions."""
from __future__ import absolute_import, division, print_function
import abc
import functools
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.param import ParamContainer

__all__ = []


@docstring_class(indent_level=1)
class BaseWavefunction(ParamContainer):
    r"""Base wavefunction class.

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

    """

    def __init__(self, nelec, nspin, dtype=None, memory=None):
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
        self.assign_nelec(nelec)
        self.assign_nspin(nspin)
        self.assign_dtype(dtype)
        self.assign_memory(memory)
        # assign_params not included because it depends on template_params, which may involve
        # more attributes than is given above

        # create cache
        self._cache_fns = {}

    @property
    def nspatial(self):
        """Return the number of spatial orbitals.

        Returns
        -------
        nspatial : int
            Number of spatial orbitals.

        """
        return self.nspin // 2

    @property
    def nparams(self):
        """Return the number of wavefunction parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        return self.template_params.size

    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of int
            Shape of the parameters.

        """
        return self.template_params.shape

    @property
    def spin(self):
        r"""Return the spin of the wavefunction.

        .. math::

            \frac{1}{2}(N_\alpha - N_\beta)

        Returns
        -------
        spin : float
            Spin of the wavefunction.

        Notes
        -----
        `None` means that all possible spins are allowed.

        """
        return None

    @property
    def seniority(self):
        """Return the seniority of the wavefunction.

        Seniority of a Slater determinant is its number of unpaired electrons. The seniority of the
        wavefunction is the expected number of unpaired electrons.

        Returns
        -------
        seniority : int
            Seniority of the wavefunction.

        Notes
        -----
        `None` means that all possible seniority are allowed.

        """
        return None

    def assign_nelec(self, nelec):
        """Set the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons.

        Raises
        ------
        TypeError
            If number of electrons is not an integer.
        ValueError
            If number of electrons is not a positive number.

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
            If number of spin orbitals is not an integer.
        ValueError
            If number of spin orbitals is not a positive number.
        NotImplementedError
            If number of spin orbitals is odd.

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
            Numpy data type.
            If None then set to np.float64.

        Raises
        ------
        TypeError
            If dtype is not one of float, complex, np.float64, np.complex128.

        """
        if dtype is None or dtype in (float, np.float64):
            self.dtype = np.float64
        elif dtype in (complex, np.complex128):
            self.dtype = np.complex128
        else:
            raise TypeError("dtype must be float or complex")

    def assign_memory(self, memory=None):
        """Assign memory associated with the wavefunction.

        Parameters
        ----------
        memory : {int, str, None}
            Memory available for the wavefunction.

        Raises
        ------
        ValueError
            If memory is given as a string and does not end with "mb" or "gb".
        TypeError
            If memory is not given as a None, int, float, or string.

        """
        if memory is None:
            memory = np.inf
        elif isinstance(memory, (int, float)):
            memory = float(memory)
        elif isinstance(memory, str):
            if 'mb' in memory.lower():
                memory = 1e6 * float(memory.rstrip('mb .'))
            elif 'gb' in memory.lower():
                memory = 1e9 * float(memory.rstrip('gb .'))
            else:
                raise ValueError('Memory given as a string should end with either "mb" or "gb".')
        else:
            raise TypeError('Memory should be given as a `None`, int, float, or string.')
        self.memory = memory

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
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

        Notes
        -----
        Depends on dtype, template_params, and nparams.

        """
        if params is None:
            params = self.template_params

        # check if numpy array and if dtype is one of int, float, or complex
        super().assign_params(params)
        params = self.params

        # check shape and dtype
        if params.size != self.nparams:
            raise ValueError('There must be {0} parameters.'.format(self.nparams))
        elif params.dtype in (complex, np.complex128) and self.dtype in (float, np.float64):
            raise TypeError('If the parameters are `complex`, then the `dtype` of the wavefunction '
                            'must be `np.complex128`')
        elif params.dtype not in [float, np.float64, complex, np.complex128]:
            raise TypeError('If the parameters are neither float or complex.')

        if len(params.shape) == 1:
            params = params.reshape(self.params_shape)
        elif params.shape != self.params_shape:
            raise ValueError('Parameters must either be flat or have the same shape as the '
                             'template, {0}.'.format(self.params_shape))

        self.params = params.astype(self.dtype)

        # add random noise
        if add_noise:
            # set scale
            scale = 0.2 / self.nparams
            self.params += scale * (np.random.rand(*self.params_shape) - 0.5)
            if self.dtype in [complex, np.complex128]:
                self.params += 0.01j * scale * (np.random.rand(*self.params_shape).astype(complex)
                                                - 0.5)

    def load_cache(self):
        """Load the functions that will be cached.

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
            memory = int((self.memory - 5*8*self.params.size) / (self.params.size + 1))

        # create function that will be cached
        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp(sd):
            """Calculate the overlap with the Slater determinant.

            Parameters
            ----------
            sd : gmpy2.mpz
                Occupation vector of a Slater determinant given as a bitstring.

            Raises
            ------
            NotImplementedError
                If called.

            """
            raise NotImplementedError

        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp_deriv(sd, deriv):
            """Calculate the derivative of the overlap with the Slater determinant.

            Parameters
            ----------
            sd : gmpy2.mpz
                Occupation vector of a Slater determinant given as a bitstring.
            deriv : int
                Index of the parameter with respect to which the overlap is derivatized.

            Raises
            ------
            NotImplementedError
                If called.

            """
            raise NotImplementedError

        # store the cached function
        self._cache_fns['overlap'] = _olp
        self._cache_fns['overlap derivative'] = _olp_deriv

    def clear_cache(self, key=None):
        """Clear the cache associated with the wavefunction.

        Parameters
        ----------
        key : str
            Key to the cached function in _cache_fns.
            Default clears all cached functions.

        Raises
        ------
        KeyError
            If given key is not present in the _cache_fns.
        ValueError
            If cached function does not have decorator functools.lru_cache.

        """
        try:
            if key is None:
                for fn in self._cache_fns.values():
                    fn.cache_clear()
            else:
                self._cache_fns[key].cache_clear()
        except KeyError as error:
            raise KeyError('Given function key is not present in _cache_fns') from error
        except AttributeError as error:
            raise AttributeError('Given cached function does not have decorator '
                                 '`functools.lru_cache`') from error

    @abc.abstractproperty
    def template_params(self):
        """Return the template of the parameters of the given wavefunction.

        Returns
        -------
        template_params : np.ndarray
            Default parameters of the wavefunction.

        Notes
        -----
        May depend on other attributes or properties.

        """
        pass

    @abc.abstractmethod
    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \braket{\Phi_i | \Psi}

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
        pass
