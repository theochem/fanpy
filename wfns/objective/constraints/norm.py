"""Normalization constraint on the wavefunction."""
import numpy as np
import wfns.backend.slater as slater
from wfns.objective.base import BaseObjective
from wfns.objective.schrodinger.base import BaseSchrodinger
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
from wfns.wfn.base import BaseWavefunction
from wfns.wfn.ci.base import CIWavefunction


class NormConstraint(BaseObjective):
    r"""Normalization constraint on the wavefunction.

    .. math::

        \left< \Psi \middle| \Psi \right> - 1 = 0

    However, this may be difficult to evaluate because :math:`\Psi` contains too many Slater
    determinants. Normalization with respect to a reference wavefunction (intermediate
    normalization) can be used in these cases:

    .. math::

        \left<\Phi \middle| \Psi\right> - 1 = 0

    where :math:`\Phi` is some reference wavefunction.

    Attributes
    ----------
    tmpfile : str
        Name of the file that will store the parameters used by the objective method.
        By default, the parameter values are not stored.
        If a file name is provided, then parameters are stored upon execution of the objective
        method.
    param_selection : ParamMask
        Selection of parameters that will be used in the objective.
        Default selects the wavefunction parameters.
        Any subset of the wavefunction, composite wavefunction, and Hamiltonian parameters can be
        selected.
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    refwfn : {tuple of int, CIWavefunction}
        Wavefunction against which the Schrodinger equation is integrated.
        Tuple of Slater determinants will be interpreted as a projection space, and the reference
        wavefunction will be the given wavefunction truncated to the given projection space.

    Properties
    ----------
    params : {np.ndarray(K, )}
        Parameters of the objective at the current state.

    """

    def __init__(self, wfn, refwfn=None, param_selection=None, tmpfile=""):
        r"""Initialize the norm constraint.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction that defines the state of the system (number of electrons and excited
            state).
        refwfn : {tuple/list of int, CIWavefunction, None}
            Wavefunction against which the norm is integrated.
            Tuple of Slater determinants will be interpreted as a projection space, and the
            reference wavefunction will be the given wavefunction truncated to the given projection
            space.
            By default, the given wavefunction is used as the reference by using a complete
            projection space.
        param_selection : {list, tuple, ParamMask, None}
            Selection of parameters that will be used to construct the objective.
            For use as a constraint, the same `param_selection` should be provided as the objective
            instance that will be constrained.
            If list/tuple, then each entry is a 2-tuple of the parameter object and the numpy
            indexing array for the active parameters. See `ParamMask.__init__` for details.
        tmpfile : str
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective
            method.

        """
        if not isinstance(wfn, BaseWavefunction):
            raise TypeError(
                "Given wavefunction is not an instance of BaseWavefunction (or its " "child)."
            )
        self.wfn = wfn
        self.assign_refwfn(refwfn)
        super().__init__(param_selection, tmpfile=tmpfile)

    # NOTE: the overlap calculation is already defined in BaseSchrodinger
    wrapped_get_overlap = BaseSchrodinger.wrapped_get_overlap

    # NOTE: the reference wavefunction assignment is already defined in OneSidedEnergy
    assign_refwfn = OneSidedEnergy.assign_refwfn

    @property
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """
        return 1

    # NOTE: much of this code is copied from BaseSchrodinger.get_energy_one_proj
    def objective(self, params):
        r"""Return the norm of the wavefunction.

        .. math::

            \left< \Phi \middle| \Psi \right> - 1
            = \sum_{\mathbf{m} \in S} f^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right> - 1

        where :math:`S` is a set of Slater determinants that are used to build :math:`\Phi`. Then,
        the vector of Slate determinants :math:`[\mathbf{m}]`, is denoted with `ref_sds` and the
        vector :math:`[f^*(\mathbf{m})]` is denoted by `ref_coeffs`.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        objective_value : float
            Value of the objective for the given parameters.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Establish shortcuts
        get_overlap = self.wrapped_get_overlap
        ref = self.refwfn  # pylint: disable=E1101
        # Define reference
        if isinstance(ref, CIWavefunction):
            ref_sds = ref.sd_vec
            ref_coeffs = ref.params
        else:
            if slater.is_sd_compatible(ref):
                ref = [ref]
            ref_sds = ref
            ref_coeffs = np.array([get_overlap(i) for i in ref])
        # Compute
        overlaps = np.array([get_overlap(i) for i in ref_sds])
        return np.sum(ref_coeffs * overlaps) - 1

    # NOTE: much of this code is copied from BaseSchrodinger.get_energy_one_proj
    def gradient(self, params):
        r"""Gradient of the normalization constraint of the wavefunction.

        .. math::

            \frac{d}{dx} (\left< \Phi \middle| \Psi \right> - 1)
            &= \frac{d}{dx} \left< \Phi \middle| \Psi \right>\\
            &= \sum_{\mathbf{m} \in S}
               \frac{d}{dx} f^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right> +
               f^*(\mathbf{m}) \frac{d}{dx} \left< \mathbf{m} \middle| \Psi \right>\\

        where :math:`S` is a set of Slater determinants that are used to build :math:`\Phi`. Then,
        the vector of Slate determinants :math:`[\mathbf{m}]`, is denoted with `ref_sds` and the
        vector :math:`[f^*(\mathbf{m})]` is denoted by `ref_coeffs`.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        gradient : np.array(N,)
            Derivative of the objective with respect to each of the parameters.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Establish shortcuts
        get_overlap = self.wrapped_get_overlap
        ref = self.refwfn  # pylint: disable=E1101
        # Define reference
        if isinstance(ref, CIWavefunction):
            ref_sds = ref.sd_vec
            ref_coeffs = ref.params
        else:
            if slater.is_sd_compatible(ref):
                ref = [ref]
            ref_sds = ref
            ref_coeffs = np.array([get_overlap(i) for i in ref])
        overlaps = np.array([get_overlap(i) for i in ref_sds])

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
                d_ref_coeffs = np.array([get_overlap(k, i) for k in ref])
            # Compute
            d_norm[i] = np.sum(d_ref_coeffs * overlaps)
            d_norm[i] += np.sum(ref_coeffs * np.array([get_overlap(k, i) for k in ref_sds]))
        return d_norm
