"""Schrodinger equation rearranged for energy."""
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.objective.base_objective import BaseObjective
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.backend import slater, sd_list


@docstring_class(indent_level=1)
class OneSidedEnergy(BaseObjective):
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
    _pspace : {tuple of int, tuple of CIWavefunction, None}
        States onto which the Schrodinger equation is projected on the left.

    """
    def __init__(self, wfn, ham, tmpfile='', param_types=None, pspace=None):
        """

        Parameters
        ----------
        pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected.
            By default, the largest space is used.

        """
        super().__init__(wfn, ham, tmpfile=tmpfile, param_types=param_types)
        self.assign_pspace(pspace)

    def assign_pspace(self, pspace=None):
        """Set the projection space.

        Parameters
        ----------
        pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the wavefunction is projected on the left.
            By default, `None` is stored but the largest space is yielded in the property
            `pspace`.

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
        if pspace is None:
            self._pspace = None
        elif isinstance(pspace, (list, tuple)):
            for state in pspace:
                if slater.is_sd_compatible(state):
                    occs = slater.occ_indices(state)
                    if len(occs) != self.wfn.nelec:
                        raise ValueError('Given state does not have the same number of electrons as'
                                         ' the given wavefunction.')
                    elif any(i >= self.wfn.nspin for i in occs):
                        raise ValueError('Given state does not have the same number of spin '
                                         'orbitals as the given wavefunction.')
                elif isinstance(state, CIWavefunction):
                    if state.nelec != self.wfn.nelec:
                        raise ValueError('Given state does not have the same number of electrons as'
                                         ' the given wavefunction.')
                    elif state.nspin != self.wfn.nspin:
                        raise ValueError('Given state does not have the same number of spin '
                                         'orbitals as the given wavefunction.')
                else:
                    raise TypeError('Projection space must only contain Slater determinants or '
                                    'CIWavefunctions.')
            self._pspace = tuple(pspace)
        else:
            raise TypeError('Projection space must be given as a list or a tuple.')

    @property
    def pspace(self):
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
        if self._pspace is None:
            yield from sd_list.sd_list(self.wfn.nelec, self.wfn.nspatial, spin=self.wfn.spin,
                                       seniority=self.wfn.seniority)
        else:
            yield from self._pspace

    def objective(self, params):
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Save params
        self.save_params()

        return self.get_energy_one_proj(self.pspace)

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

        return np.array([self.get_energy_one_proj(self.pspace, i) for i in range(params.size)])
