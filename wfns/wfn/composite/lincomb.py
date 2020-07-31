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
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    dtype
        Data type of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, wfns, memory=None, params=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
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

    # pylint: disable=W0223
    def __init__(self, nelec, nspin, wfns, memory=None, params=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        wfns : tuple of BaseWavefunction
            Wavefunctions that will be linearly combined.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).

        """
        super().__init__(nelec, nspin, memory=memory)
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
        # pylint: disable=R1705
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
        # pylint: disable=R1705
        if all(wfn.seniority == self.wfns[0].seniority for wfn in self.wfns):
            return self.wfns[0].seniority
        else:
            return None

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
            Default is first wavefunction.
        add_noise : {bool, False}
            Option to add noise to the given parameters.
            Default is False.

        """
        if params is None:
            params = np.zeros(len(self.wfns))
            params[0] = 1.0

        super().assign_params(params=params, add_noise=add_noise)

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
            raise TypeError("Each wavefunction must be a instance of `BaseWavefunction`.")
        if any(wfn.nelec != self.nelec for wfn in wfns):
            raise ValueError(
                "Given wavefunction does not have the same number of electrons as the"
                " the instantiated NonorthWavefunction."
            )
        if any(wfn.memory != self.memory for wfn in wfns):
            raise ValueError(
                "Given wavefunction does not have the same memory as the "
                "instantiated NonorthWavefunction."
            )
        if len(wfns) == 1:
            raise ValueError("Only one wavefunction is given.")

        self.wfns = tuple(wfns)

    # FIXME: derivative should be with respect to the coefficients/params
    # FIXME: how to derivatize wrt parameters of the wavefunction?
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
        wfn_contrib = np.array([wfn.get_overlap(sd, deriv=deriv) for wfn in self.wfns])
        return np.sum(self.params * wfn_contrib)
