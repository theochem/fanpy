"""Base class for composite wavefunctions that modifies one wavefunction."""
from wfns.wfn.base_wavefunction import BaseWavefunction


class BaseCompositeOneWavefunction(BaseWavefunction):
    """Base class for composite wavefunction that uses only one wavefunction.

    Note that `spin`, `seniority`, `template_params`, `get_overlap`

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that is being modified.

    """
    def __init__(self, nelec, nspin, wfn, dtype=None, memory=None, params=None):
        r"""

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction that will be modified.

        """
        super().__init__(nelec, nspin, dtype=dtype, memory=memory)
        self.assign_wfn(wfn)
        self.assign_params(params)
        self.load_cache()

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
            raise TypeError('Given wavefunction must be an instance of BaseWavefunction.')
        elif wfn.nelec != self.nelec:
            raise ValueError('Given wavefunction does not have the same number of electrons as the'
                             ' the instantiated NonorthWavefunction.')
        elif wfn.dtype != self.dtype:
            raise ValueError('Given wavefunction does not have the same data type as the '
                             'instantiated NonorthWavefunction.')
        elif wfn.memory != self.memory:
            raise ValueError('Given wavefunction does not have the same memory as the '
                             'instantiated NonorthWavefunction.')
        self.wfn = wfn
