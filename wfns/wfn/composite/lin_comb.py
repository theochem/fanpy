"""Linear combination of different wavefunctions."""
import numpy as np
from wfns.wfn.base_wavefunction import BaseWavefunction


class LinearCombinationWavefunction(BaseWavefunction):
    """Linear combination of different wavefunctions.

    Attributes
    ----------
    wfns : tuple of BaseWavefunction
        Wavefunctions that will be linearly combined.

    """

    def __init__(self, nelec, nspin, wfns, dtype=None, memory=None, params=None):
        """

        Parameters
        ----------
        wfns : tuple of BaseWavefunction
            Wavefunctions that will be linearly combined.

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

    # FIXME: add caching
    def get_overlap(self, sd, deriv=None):
        wfn_contrib = np.array([wfn.get_overlap(sd, deriv=deriv) for wfn in self.wfns])
        return np.sum(self.params * wfn_contrib)
