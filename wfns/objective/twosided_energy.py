"""Energy of the Schrodinger equation integrated against projected forms of the wavefunction."""
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.objective.base_objective import BaseObjective
from wfns.backend import slater, sd_list


@docstring_class(indent_level=1)
class TwoSidedEnergy(BaseObjective):
    r"""Energy of the Schrodinger equations integrated against projected forms of the wavefuntion.

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

    where :math:`S_{left}` and  :math:`S_{right}` are the projection spaces for the left and right
    side of the integral :math:`\braket{\Psi | \hat{H} | \Psi}`, respectively, and :math:`S_{norm}`
    is the projection space for the norm, :math:`\braket{\Psi | \Psi}`.

    Attributes
    ----------
    pspace_l : {tuple of int, None}
        States in the projection space of the left side of the integral
        :math:`\braket{\Psi | \hat{H} | \Psi}`.
        By default, the largest space is used.
    pspace_r : {tuple of int, None}
        States in the projection space of the right side of the integral
        :math:`\braket{\Psi | \hat{H} | \Psi}`.
        By default, the same space as `pspace_l` is used.
    pspace_n : {tuple of int, None}
        States in the projection space of the norm :math:`\braket{\Psi | \Psi}`.
        By default, the same space as `pspace_l` is used.

    """
    def __init__(self, wfn, ham, tmpfile='', param_selection=None, pspace_l=None, pspace_r=None,
                 pspace_n=None):
        """

        Parameters
        ----------
        pspace_l : {tuple/list of int, None}
            States in the projection space of the left side of the integral
            :math:`\braket{\Psi | \hat{H} | \Psi}`.
            By default, the largest space is used.
        pspace_r : {tuple/list of int, None}
            States in the projection space of the right side of the integral
            :math:`\braket{\Psi | \hat{H} | \Psi}`.
            By default, the same space as `pspace_l` is used.
        pspace_n : {tuple/list of int, None}
            States in the projection space of the norm :math:`\braket{\Psi | \Psi}`.
            By default, the same space as `pspace_l` is used.

        """
        super().__init__(wfn, ham, tmpfile=tmpfile, param_selection=param_selection)
        self.assign_pspaces(pspace_l, pspace_r, pspace_n)

    def assign_pspaces(self, pspace_l=None, pspace_r=None, pspace_n=None):
        """Set the projection space.

        Parameters
        ----------
        pspace_l : {tuple/list of int, None}
            States in the projection space of the left side of the integral
            :math:`\braket{\Psi | \hat{H} | \Psi}`.
            By default, the largest space is used.
        pspace_r : {tuple/list of int, None}
            States in the projection space of the right side of the integral
            :math:`\braket{\Psi | \hat{H} | \Psi}`.
            By default, the same space as `pspace_l` is used.
        pspace_n : {tuple/list of int, None}
            States in the projection space of the norm :math:`\braket{\Psi | \Psi}`.
            By default, the same space as `pspace_l` is used.

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
        if pspace_l is None:
            pspace_l = sd_list.sd_list(self.wfn.nelec, self.wfn.nspatial, spin=self.wfn.spin,
                                       seniority=self.wfn.seniority)

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

        self.pspace_l = tuple(pspace_l)
        self.pspace_r = tuple(pspace_r) if pspace_r is not None else pspace_r
        self.pspace_n = tuple(pspace_n) if pspace_n is not None else pspace_n

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
