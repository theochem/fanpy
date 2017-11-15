"""Linear combination of different wavefunctions."""
import numpy as np
from wfns.wfn.base import BaseWavefunction


class LinearCombinationWavefunction(BaseWavefunction):
    """Linear combination of different wavefunctions.

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
    wfns : tuple of BaseWavefunction
        Wavefunctions that will be linearly combined.

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
    __init__(self, nelec, nspin, wfns, dtype=None, memory=None, params=None)
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
        Assign parameters of the wavefunction.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    assign_wfns(self, wfns)
        Assign the wavefunctions that will be linearly combined.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.

    """

    def __init__(self, nelec, nspin, wfns, dtype=None, memory=None, params=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        wfns : tuple of BaseWavefunction
            Wavefunctions that will be linearly combined.
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type.
            Default is `np.float64`.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).

        """
        super().__init__(nelec, nspin, dtype=dtype, memory=memory)
        self.assign_wfns(wfns)
        self.assign_params(params=params)

    @property
    def spin(self):
        """Return the spin of the wavefunction.

        Spin is certain when all of the wavefunctions have the same spin.

        Returns
        -------
        spin : {float, None}
            Spin of the wavefunction.
            `None` means that all spins are allowed.

        """
        if all(wfn.spin == self.wfns[0].spin for wfn in self.wfns):
            return self.wfns[0].spin
        else:
            return None

    @property
    def seniority(self):
        """Return the seniority of the wavefunction.

        Seniority is certain when all of the wavefunctions have the same seniority.

        Returns
        -------
        seniority : {int, None}
            Seniority of the wavefunction.
            `None` means that all senioritys are allowed.

        """
        if all(wfn.seniority == self.wfns[0].seniority for wfn in self.wfns):
            return self.wfns[0].seniority
        else:
            return None

    @property
    def template_params(self):
        """Return the template parameters of the wavefunction.

        First wavefunction given is assumed to be the most important wavefunction. It will be used
        as a reference.

        Returns
        -------
        template_params : np.ndarray
            Template parameters of the wavefunction.

        Notes
        -----
        Must have `wfns` defined.

        """
        params = np.zeros(len(self.wfns), dtype=self.dtype)
        params[0] = 1.0
        return params

    def assign_wfns(self, wfns):
        """Assign the wavefunctions that will be linearly combined.

        Parameters
        ----------
        wfns : tuple of BaseWavefunction
             Wavefunctions that will be linearly combined.

        Raises
        ------
        TypeError
            If wavefunction is not an instance of `BaseWavefunction`
        ValueError
            If the given wavefunction does not have the same number of electrons as the initialized
            value.
            If the given wavefunction does not have the same data type as the initialized value.
            If the given wavefunction does not have the same memory as the initialized value.
            If only one wavefunction is given.

        """
        if any(not isinstance(wfn, BaseWavefunction) for wfn in wfns):
            raise TypeError('Each wavefunction must be a instance of `BaseWavefunction`.')
        elif any(wfn.nelec != self.nelec for wfn in wfns):
            raise ValueError('Given wavefunction does not have the same number of electrons as the'
                             ' the instantiated NonorthWavefunction.')
        elif any(wfn.dtype != self.dtype for wfn in wfns):
            raise ValueError('Given wavefunction does not have the same data type as the '
                             'instantiated NonorthWavefunction.')
        elif any(wfn.memory != self.memory for wfn in wfns):
            raise ValueError('Given wavefunction does not have the same memory as the '
                             'instantiated NonorthWavefunction.')
        elif len(wfns) == 1:
            raise ValueError('Only one wavefunction is given.')

        self.wfns = tuple(wfns)

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \braket{\mathbf{m} | \Psi}

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
        wfn_contrib = np.array([wfn.get_overlap(sd, deriv=deriv) for wfn in self.wfns])
        return np.sum(self.params * wfn_contrib)
