"""Schrodinger equation rearranged for energy."""
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.objective.base_objective import BaseObjective
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.backend import slater, sd_list


@docstring_class(indent_level=1)
class TwoSidedEnergy(BaseObjective):
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
        self.assign_pspaces(pspace_l, pspace_r, pspace_n)

    def assign_pspaces(self, pspace_l=None, pspace_r=None, pspace_n=None):
        """Set the projection space.

        Parameters
        ----------
        pspace_l : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected on the left.
            By default, `None` is stored but the largest space is yielded in the property
            `pspace_l`.
        pspace_r : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected on the right.
            By default, `None` is stored but the property `pspace_l` is yielded in the property
            `pspace_r`.
        pspace_n : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected in the norm.
            By default, `None` is stored but the property `pspace_n` is yielded in the property
            `pspace_r`.

        Raises
        ------
        TypeError
            If projection space is not a list or a tuple.
            If state in projection space is not given as a Slater determinant or a CI wavefunction.
        ValueError
            If given state in projection space does not have the same number of electrons as the
            wavefunction.
            If given state in projection space does not have the same number of spin orbitals as the
            wavefunction.

        """
        for pspace in [pspace_l, pspace_r, pspace_n]:
            if pspace is None:
                continue
            if not isinstance(pspace, (list, tuple)):
                raise TypeError('Projection space must be given as a list or a tuple.')
            for state in pspace:
                if slater.is_sd_compatible(state):
                    occs = slater.occ_indices(state)
                    if len(occs) != self.wfn.nelec:
                        raise ValueError('Given state does not have the same number of electrons as'
                                         ' the given wavefunction.')
                    elif any(i >= self.wfn.nspin for i in occs):
                        raise ValueError('Given state does not have the same number of spin '
                                         'orbitals as the given wavefunction.')
                else:
                    raise TypeError('Projection space must only contain Slater determinants.')

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
        if self._pspace_l is None:
            yield from sd_list.sd_list(self.wfn.nelec, self.wfn.nspatial, spin=self.wfn.spin,
                                       seniority=self.wfn.seniority)
        else:
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
            yield from self.pspace_l
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
            yield from self.pspace_l
        else:
            yield from self._pspace_n

    def objective(self, params):
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Save params
        self.save_params()

        return self.get_energy_two_proj(self.pspace_l, self.pspace_r, self.pspace_n)

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
        self.save_params()

        return np.array([self.get_energy_two_proj(self.pspace_l, self.pspace_r, self.pspace_n, i)
                         for i in range(params.size)])
