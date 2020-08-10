"""Base class for composite wavefunctions that modifies one wavefunction."""
import os
import numpy as np
from wfns.wfn.base import BaseWavefunction


class BaseCompositeOneWavefunction(BaseWavefunction):
    """Base class for composite wavefunction that uses only one wavefunction.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    wfn : BaseWavefunction
        Wavefunction that is being modified.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    dtype
        Data type of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_memory(self, memory=None):
        Assign memory available for the wavefunction.
    assign_params(self, params=None, add_noise=False)
        Assign parameters of the wavefunction.
    enable_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.

    Abstract Methods
    ----------------
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.

    """

    # pylint: disable=W0223
    def __init__(self, nelec, nspin, wfn, memory=None, params=None, enable_cache=True):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).
        wfn : BaseWavefunction
            Wavefunction that will be modified.
        enable_cache : bool
            Option to cache the results of `_olp` and `_olp_deriv`.
            By default, `_olp` and `_olp_deriv` are cached.

        """
        super().__init__(nelec, nspin, memory=memory)
        self.assign_wfn(wfn)
        self.assign_params(params)
        if enable_cache:
            self.enable_cache()

    def assign_wfn(self, wfn):
        """Assign the wavefunction.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction that will be modified.

        Raises
        ------
        TypeError
            If the given wavefunction is not an instance of BaseWavefunction.
        ValueError
            If the given wavefunction does not have the same number of electrons as the initialized
            value.
            If the given wavefunction does not have the same data type as the initialized value.
            If the given wavefunction does not have the same memory as the initialized value.

        """
        if not isinstance(wfn, BaseWavefunction):
            raise TypeError("Given wavefunction must be an instance of BaseWavefunction.")
        if wfn.nelec != self.nelec:
            raise ValueError(
                "Given wavefunction does not have the same number of electrons as the"
                " the instantiated NonorthWavefunction."
            )
        if wfn.memory != self.memory:
            raise ValueError(
                "Given wavefunction does not have the same memory as the "
                "instantiated NonorthWavefunction."
            )
        self.wfn = wfn

    def save_params(self, filename):
        """Save parameters associated with the wavefunction.

        Since both the parameters of the composite wavefunction and the underlying wavefunction are
        needed to replicated the overlaps, they are saved as separate files, using the
        given filename as the root (removing the extension). The parameters of the underlying
        wavefunction are saved by appending the name of the wavefunction to the end of the root.

        Parameters
        ----------
        filename : str

        """
        root, ext = os.path.splitext(filename)
        np.save(filename, self.params)
        name = type(self.wfn).__name__
        self.wfn.save_params(f"{root}_{name}{ext}")
