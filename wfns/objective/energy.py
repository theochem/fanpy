"""Schrodinger equation rearranged for energy."""
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.objective.base_objective import BaseObjective
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.backend import slater, sd_list


@docstring_class(indent_level=1)
class Energy(BaseObjective):
    r"""Energy of the Schrodinger equations.

    .. math::

        E = \frac{\braket{\Psi | \hat{H} | \Psi}}{\braket{\Psi | \Psi}}

    Since this equation may be expensive (wavefunction may require many Slater determinants), we can
    insert projection operators onto the wavefunction.

    .. math::

        E = \frac{
          \bra{\Psi}
          \left(
            \sum_{\mathbf{m}_1 \in S_{left}} \ket{\mathbf{m}_1} \bra{\mathbf{m}_1}
          \right)
          \hat{H}
          \left(
            \sum_{\mathbf{m}_2 \in S_{right}} \ket{\mathbf{m}_2} \bra{\mathbf{m}_2}
          \right)
          \ket{\Psi}
        }{
          \bra{\Psi}
          \left(
            \sum_{\mathbf{m}_3 \in S_{norm}} \ket{\mathbf{m}_3} \bra{\mathbf{m}_3}
          \right)
          \ket{\Psi}
        }


    Attributes
    ----------
    _pspace_l : {tuple/list of int, tuple/list of CIWavefunction, None}
        States onto which the Schrodinger equation is projected on the left.
        By default, the largest space is used.
    _pspace_r : {tuple/list of int, tuple/list of CIWavefunction, None}
        States onto which the Schrodinger equation is projected on the right.
    _pspace_n : {tuple/list of int, tuple/list of CIWavefunction, None}
        States onto which the Schrodinger equation is projected on the right.

    """
    def __init__(self, wfn, ham, tmpfile='', param_types=None, pspace_l=None, pspace_r=None,
                 pspace_n=None):
        """

        Parameters
        ----------
        pspace_l : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected on the left.
            By default, the largest space is used.
        pspace_r : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected on the right.
            By default, `pspace_l` is used.
        pspace_n : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected in the norm.
            By default, `pspace_l` is used.

        """
        super().__init__(wfn, ham, tmpfile=tmpfile, param_types=param_types)
        self.assign_pspace(pspace_l, pspace_r, pspace_n)

    def assign_pspace(self, pspace_l, pspace_r, pspace_n):
        """Set the projection space.

        Parameters
        ----------
        pspace_l : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected on the left.
            By default, the largest space is used.
        pspace_r : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected on the right.
            By default, `pspace_l` is used.
        pspace_n : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected in the norm.
            By default, `pspace_l` is used.

        Raises
        ------
        TypeError
            If list/tuple of incompatible objects.
            If not list/tuple or None.

        """
        if pspace_l is None:
            pspace_l = sd_list.sd_list(self.wfn.nelec, self.wfn.nspatial, spin=self.wfn.spin,
                                       seniority=self.wfn.seniority)

        for pspace in [pspace_l, pspace_r, pspace_n]:
            if pspace is None:
                continue
            if not (isinstance(pspace, (list, tuple)) and
                    all(slater.is_sd_compatible(state) or isinstance(state, CIWavefunction)
                        for state in pspace)):
                raise TypeError('Projected space must be given as a list/tuple of Slater '
                                'determinants. See `backend.slater` for compatible Slater '
                                'determinant formats.')

        self._pspace_l = pspace_l
        self._pspace_r = pspace_r
        self._pspace_n = pspace_n

    @property
    def pspace_l(self):
        """Yield the projection space on the left side of the :math:`\braket{\Psi | H | \Psi}`.

        Yields
        ------
        pspace_l : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected on the left.
            By default, the largest space is used.

        Notes
        -----
        The projection space is generated instead of storing it to improve memory usage.

        """
        yield from self._pspace_l

    @property
    def pspace_r(self):
        """Yield the projection space on the right side of the :math:`\braket{\Psi | H | \Psi}`.

        Yields
        ------
        pspace_r : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected on the right.

        Notes
        -----
        The projection space is generated instead of storing it to improve memory usage, especially
        when `_pspace_r` is None.

        """
        if self._pspace_r is None:
            yield from self._pspace_l
        else:
            yield from self._pspace_r

    @property
    def pspace_n(self):
        """Yield the projection space on norm, :math:`\braket{\Psi | \Psi}`.

        Yields
        ------
        pspace_n : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected in the norm.

        Notes
        -----
        The projection space is generated instead of storing it to improve memory usage, especially
        when `_pspace_n` is None.

        """
        if self._pspace_n is None:
            yield from self._pspace_l
        else:
            yield from self._pspace_n

    def objective(self, params):
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Save params
        self.save_params(params)

        if self._pspace_r is None and self._pspace_n:
            return self.get_energy_one_proj(self.pspace_l)
        else:
            return self.get_enregy_two_proj(self.pspace_l, self.pspace_r, self.pspace_n)

    def gradient(self, params):
        """Return the gradient of the objective.

        Parameters
        ----------
        params : np.ndarray
            Parameter that changes the objective.

        Returns
        -------
        gradient : np.array(N,)
            Derivative of the objective with respect to each of the parameters.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Save params
        self.save_params(params)

        derivs = np.arange(params.size)
        if self._pspace_r is None and self._pspace_n:
            get_energy = np.vectorize(lambda deriv: self.get_energy_one_proj(self.pspace_l, deriv))
        else:
            get_energy = np.vectorize(lambda deriv: self.get_energy_two_proj(self.pspace_l,
                                                                             self.pspace_r,
                                                                             self.pspace_n, deriv))
        return get_energy(derivs)
