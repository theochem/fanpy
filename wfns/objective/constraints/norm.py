"""Normalization constraint on the wavefunction."""
import numpy as np
from wfns.objective.base_objective import BaseObjective
from wfns.wfn.base import BaseWavefunction
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
from wfns.objective.schrodinger.base_schrodinger import BaseSchrodinger
from wfns.wfn.ci.ci_wavefunction import CIWavefunction
import wfns.backend.slater as slater


class NormConstraint(BaseObjective):
    """Normalization constraint on the wavefunction.

    .. math::

        \braket{\Psi | \Psi} - 1 = 0

    However, this may be difficult to evaluate because :math:`\Psi` contains too many Slater
    determinants. Normalization with respect to a reference wavefunction (intermediate
    normalization) can be used in these cases:

    .. math::

        \braket{\Phi | \Psi} - 1 = 0

    where :math:`\Phi` is some reference wavefunction.

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    refwfn : {tuple of int, tuple of CIWavefunction, None}
        Wavefunction against which the Schrodinger equation is integrated.
        Tuple of Slater determinants will be interpreted as a projection space, and the reference
        wavefunction will be the given wavefunction truncated to the given projection space.

    """
    def __init__(self, wfn, refwfn=None, param_selection=None, tmpfile=''):
        """Initialize the norm constraint.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction that defines the state of the system (number of electrons and excited
            state).
        refwfn : {tuple/list of int, tuple/list of CIWavefunction, None}
            Wavefunction against which the norm is integrated.
            Tuple of Slater determinants will be interpreted as a projection space, and the
            reference wavefunction will be the given wavefunction truncated to the given projection
            space.
            By default, the given wavefunction is used as the reference by using a complete
            projection space.

        """
        if not isinstance(wfn, BaseWavefunction):
            raise TypeError('Given wavefunction is not an instance of BaseWavefunction (or its '
                            'child).')
        self.wfn = wfn
        self.assign_refwfn(refwfn)
        super().__init__(param_selection, tmpfile=tmpfile)

    # NOTE: the overlap calculation is already defined in BaseSchrodinger
    wrapped_get_overlap = BaseSchrodinger.wrapped_get_overlap

    # NOTE: the reference wavefunction assignment is already defined in OneSidedEnergy
    assign_refwfn = OneSidedEnergy.assign_refwfn

    @property
    def num_eqns(self):
        return 1

    # NOTE: much of this code is copied from BaseSchrodinger.get_energy_one_proj
    def objective(self, params):
        """Normalization constraint of the wavefunction.

        .. math:

            \braket{\Phi | \Psi} - 1
            &= \sum_{\mathbf{m} \in S} f^*(\mathbf{m}) \braket{\mathbf{m} | \Psi} - 1

        where :math:`S` is a set of Slater determinants that are used to build :math:`\Phi`. Then,
        the vector of Slate determinants :math:`[\mathbf{m}]`, is denoted with `ref_sds` and the
        vector :math:`[f^*(\mathbf{m})]` is denoted by `ref_coeffs`.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Establish shortcuts
        get_overlap = np.vectorize(self.wrapped_get_overlap)
        ref = self.refwfn
        # Define reference
        if isinstance(ref, CIWavefunction):
            ref_sds = ref.sd_vec
            ref_coeffs = ref.params
        else:
            if slater.is_sd_compatible(ref):
                ref = [ref]
            ref_sds = ref
            ref_coeffs = get_overlap(ref)
        # Compute
        overlaps = get_overlap(ref_sds)
        return np.sum(ref_coeffs * overlaps) - 1

    # NOTE: much of this code is copied from BaseSchrodinger.get_energy_one_proj
    def gradient(self, params):
        """Gradient of the normalization constraint of the wavefunction.

        .. math:

            \frac{d}{dx} (\braket{\Phi | \Psi} - 1)
            &= \frac{d}{dx} \braket{\Phi | \Psi}\\
            &= \sum_{\mathbf{m} \in S} \frac{d}{dx} f^*(\mathbf{m}) \braket{\mathbf{m} | \Psi} +
            f^*(\mathbf{m}) \frac{d}{dx} \braket{\mathbf{m} | \Psi}

        where :math:`S` is a set of Slater determinants that are used to build :math:`\Phi`. Then,
        the vector of Slate determinants :math:`[\mathbf{m}]`, is denoted with `ref_sds` and the
        vector :math:`[f^*(\mathbf{m})]` is denoted by `ref_coeffs`.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Establish shortcuts
        get_overlap = np.vectorize(self.wrapped_get_overlap)
        ref = self.refwfn
        # Define reference
        if isinstance(ref, CIWavefunction):
            ref_sds = ref.sd_vec
            ref_coeffs = ref.params
        else:
            if slater.is_sd_compatible(ref):
                ref = [ref]
            ref_sds = ref
            ref_coeffs = get_overlap(ref)
        overlaps = get_overlap(ref_sds)

        d_norm = np.zeros(params.size)
        # FIXME: there is a better way to do this, but I'm hoping that the number of parameters is
        #        not terribly big (so it should be cheap to evaluate)
        for i in range(params.size):
            # get derivatives of reference wavefunction
            if isinstance(ref, CIWavefunction):
                ref_deriv = self.param_selection.derivative_index(ref, i)
                if ref_deriv is None:
                    d_ref_coeffs = 0.0
                else:
                    d_ref_coeffs = np.zeros(ref.nparams, dtype=float)
                    d_ref_coeffs[ref_deriv] = 1
            else:
                d_ref_coeffs = get_overlap(ref, i)
            # Compute
            d_norm[i] = np.sum(d_ref_coeffs * overlaps)
            d_norm[i] += np.sum(ref_coeffs * get_overlap(ref_sds, i))
        return d_norm
