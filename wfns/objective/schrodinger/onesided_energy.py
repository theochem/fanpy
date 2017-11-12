"""Energy of the Schrodinger equation integrated against a reference wavefunction."""
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.objective.schrodinger.base_schrodinger import BaseSchrodinger
from wfns.wfn.ci.base import CIWavefunction
from wfns.backend import slater, sd_list


@docstring_class(indent_level=1)
class OneSidedEnergy(BaseSchrodinger):
    r"""Energy of the Schrodinger equation integrated against a referene wavefunction.

    .. math::

        E = \frac{\braket{\Psi | \hat{H} | \Psi}}{\braket{\Psi | \Psi}}

    Since this equation may be expensive (wavefunction will probably require too many Slater
    determinants for a complete description), we use a reference wavefunction on one side of the
    integral.

    .. math::

        E &= \frac{\braket{\Phi | \hat{H} | \Psi}}{\braket{\Phi | \Psi}}\\

    where :math:`\Phi` is some reference wavefunction that can be a CI wavefunction

    .. math::

        \ket{\Phi} = \sum_{\mathbf{m} \in S} c_{\mathbf{m}} \ket{\mathbf{m}}

    or a projected form of wavefunction :math:`\Psi`

    .. math::

        \ket{\Phi} = \sum_{\mathbf{m} \in S} \braket{\Psi | \mathbf{m}} \ket{\mathbf{m}}

    where :math:`S` is the projection space.

    Attributes
    ----------
    refwfn : {tuple of int, tuple of CIWavefunction, None}
        Wavefunction against which the Schrodinger equation is integrated.
        Tuple of Slater determinants will be interpreted as a projection space, and the reference
        wavefunction will be the given wavefunction truncated to the given projection space.

    """
    def __init__(self, wfn, ham, tmpfile='', param_selection=None, refwfn=None):
        """

        Parameters
        ----------
        refwfn : {tuple/list of int, tuple/list of CIWavefunction, None}
            Wavefunction against which the Schrodinger equation is integrated.
            Tuple of Slater determinants will be interpreted as a projection space, and the
            reference wavefunction will be the given wavefunction truncated to the given projection
            space.
            By default, the given wavefunction is used as the reference by using a complete
            projection space.

        """
        super().__init__(wfn, ham, tmpfile=tmpfile, param_selection=param_selection)
        self.assign_refwfn(refwfn)

    def assign_refwfn(self, refwfn=None):
        """Set the reference wavefunction.

        Parameters
        ----------
        refwfn : {tuple/list of int, tuple/list of CIWavefunction, None}
            Wavefunction against which the Schrodinger equation is integrated.
            Tuple of Slater determinants will be interpreted as a projection space, and the
            reference wavefunction will be the given wavefunction truncated to the given projection
            space.
            By default, the given wavefunction is used as the reference by using a complete
            projection space.

        Raises
        ------
        TypeError
            If reference wavefunction is not a list or a tuple.
            If projection space (for the reference wavefunction) must be given as a list/tuple of
            Slater determinants.
        ValueError
            If given Slater determinant in projection space (for the reference wavefunction) does
            not have the same number of electrons as the wavefunction.
            If given Slater determinant in projection space (for the reference wavefunction) does
            not have the same number of spin orbitals as the wavefunction.
            If given reference wavefunction does not have the same number of electrons as the
            wavefunction.
            If given reference wavefunction does not have the same number of spin orbitals as the
            wavefunction.

        """
        if refwfn is None:
            self.refwfn = tuple(sd_list.sd_list(self.wfn.nelec, self.wfn.nspatial,
                                                spin=self.wfn.spin, seniority=self.wfn.seniority))
            # break out of function
            return

        if slater.is_sd_compatible(refwfn):
            refwfn = [refwfn]

        if isinstance(refwfn, (list, tuple)):
            for sd in refwfn:
                if slater.is_sd_compatible(sd):
                    occs = slater.occ_indices(sd)
                    if len(occs) != self.wfn.nelec:
                        raise ValueError('Given Slater determinant does not have the same number of'
                                         ' electrons as the given wavefunction.')
                    elif any(i >= self.wfn.nspin for i in occs):
                        raise ValueError('Given Slater determinant does not have the same number of'
                                         ' spin orbitals as the given wavefunction.')
                else:
                    raise TypeError('Projection space (for the reference wavefunction) must only '
                                    'contain Slater determinants.')
            self.refwfn = tuple(refwfn)
        elif isinstance(refwfn, CIWavefunction):
            if refwfn.nelec != self.wfn.nelec:
                raise ValueError('Given reference wavefunction does not have the same number of '
                                 'electrons as the given wavefunction.')
            elif refwfn.nspin != self.wfn.nspin:
                raise ValueError('Given reference wavefunction does not have the same number of '
                                 'spin orbitals as the given wavefunction.')
            self.refwfn = refwfn
        else:
            raise TypeError('Projection space must be given as a list or a tuple.')

    @property
    def num_eqns(self):
        return 1

    def objective(self, params):
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Save params
        self.save_params()

        return self.get_energy_one_proj(self.refwfn)

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

        return np.array([self.get_energy_one_proj(self.refwfn, i) for i in range(params.size)])
