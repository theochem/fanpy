"""Schrodinger equation as a least-squares problem."""
import numpy as np
from wfns.objective.schrodinger.system_nonlinear import SystemEquations


# FIXME: change name
# FIXME: inherited jacobian
class LeastSquaresEquations(SystemEquations):
    r"""Schrodinger equation as a system of equations represented as a least squares problem.

    .. math::

        \left(
            \left< \Phi_1 \middle| \hat{H} \middle| \Psi \middle>
             - E \middle< \Phi_1 \middle| \Psi \right>
        \right)^2 +
        \left(
            \left< \Phi_2 \middle| \hat{H} \middle| \Psi \middle>
             - E \middle< \Phi_2 \middle| \Psi \right>
        \right)^2 +
        \dots
        \left(
            \left< \Phi_K \middle| \hat{H} \middle| \Psi \middle>
             - E \middle< \Phi_K \middle| \Psi \right>
        \right)^2 +
        \left( f_{constraint}(\Psi, \hat{H}) \right)^2 = 0

    Energy can be a constant, a parameter that gets optimized, or a function of the wavefunction and
    hamiltonian parameters.

    .. math::

        E = \frac{\left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>}
                 {\left< \Phi_{ref} \middle| \Psi \right>}

    Additionally, the normalization constraint is added with respect to the reference state.

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    ham : BaseHamiltonian
        Hamiltonian that defines the system under study.
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
    pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
        States onto which the Schrodinger equation is projected.
        By default, the largest space is used.
    refwfn : {tuple/list of int, CIWavefunction}
        State with respect to which the energy and the norm are computed.
        If a list/tuple of Slater determinants are given, then the reference state is the given
        wavefunction truncated by the provided Slater determinants.
        Default is ground state HF.
    eqn_weights : np.ndarray
        Weights of each equation.
        By default, all equations are given weight of 1 except for the normalization constraint,
        which is weighed by the number of equations.
    energy : ParamContainer
        Energy used in the Schrodinger equation.
        Used to store/cache the energy.
    energy_type : {'fixed', 'variable', 'compute'}
        Type of the energy used in the Schrodinger equation.
        If 'fixed', the energy of the Schrodinger equation is fixed at the given value.
        If 'variable', the energy of the Schrodinger equation is optimized as a parameter.
        If 'compute', the energy of the Schrodinger equation is computed on-the-fly with respect to
        the reference.

    Properties
    ----------
    params : {np.ndarray(K, )}
        Parameters of the objective at the current state.
    nproj : int
        Number of states onto which the Schrodinger equation is projected.
    num_eqns : int
        Number of equations in the objective.

    Methods
    -------
    __init__(self, param_selection=None, tmpfile='')
        Initialize the objective.
    assign_param_selection(self, param_selection=None)
        Select parameters that will be active in the objective.
    assign_params(self, params)
        Assign the parameters to the wavefunction and/or hamiltonian.
    save_params(self)
        Save all of the parameters in the `param_selection` to the temporary file.
    wrapped_get_overlap(self, sd, deriv=None)
        Wrap `get_overlap` to be derivatized with respect to the parameters of the objective.
    wrapped_integrate_wfn_sd(self, sd, deriv=None)
        Wrap `integrate_wfn_sd` to be derivatized wrt the parameters of the objective.
    wrapped_integrate_sd_sd(self, sd1, sd2, deriv=None)
        Wrap `integrate_sd_sd` to be derivatized wrt the parameters of the objective.
    get_energy_one_proj(self, refwfn, deriv=None)
        Return the energy of the Schrodinger equation with respect to a reference wavefunction.
    get_energy_two_proj(self, pspace_l, pspace_r=None, pspace_norm=None, deriv=None)
        Return the energy of the Schrodinger equation after projecting out both sides.
    assign_pspace(self, pspace=None)
        Assign the projection space.
    assign_refwfn(self, refwfn=None)
        Assign the reference wavefunction.
    assign_constraints(self, constraints=None)
        Assign the constraints on the objective.
    assign_eqn_weights(self, eqn_weights=None)
        Assign the weights of each equation.
    objective(self, params) : float
        Return the squared sum of the values of the system of equations.
    gradient(self, params) : np.ndarray
        Return the gradient of the objective function.

    """

    @property
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """
        return 1

    def objective(self, params):
        r"""Return the squared sum of the values of the system of equations.

        .. math::

            f(\vec{x}) &=
            \left(
                \left< \Phi_1 \middle| \hat{H} \middle| \Psi \middle>
                - E \middle< \Phi_1 \middle| \Psi \right>
            \right)^2 +
            \left(
                \left< \Phi_2 \middle| \hat{H} \middle| \Psi \middle>
                - E \middle< \Phi_2 \middle| \Psi \right>
            \right)^2 +
            \dots
            \left(
                \left< \Phi_K \middle| \hat{H} \middle| \Psi \middle>
                - E \middle< \Phi_K \middle| \Psi \right>
            \right)^2 +
            \left( f_{constraint}(\Psi, \hat{H}) \right)^2\\
            &= f_1^2(\vec{x}) + f_2^2(\vec{x}) + \dots + f_K^2(\vec{x}) +
            f_{constraint}(\vec{x})^2\\

        where :math:`f_i` is the ith equation of the system of equations, :math:`K` is the number of
        states onto which the wavefunction is projected. The :math:`K+1`th equation correspond to
        constraints on the system of equations. We currently use only the normalization constraint.
        The norm is computed with respect to the reference state. The energy can be a constant, a
        parameter that gets optimized, or a function of the wavefunction and hamiltonian parameters.

        .. math::

            E = \frac{\left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>}
                     {\left< \Phi_{ref} \middle| \Psi \right>}

        Parameters
        ----------
        params : np.ndarray(N,)
            Parameters that describe the system of equations.

        Returns
        -------
        objective : float
            Output of the function that will be optimized.

        """
        # FIXME: monkey patched property
        # The property num_eqns need to be that of SystemEquations rather than the
        # LeastSquaresEquations for SystemEquations.objective to function properly. This occurs
        # because the number of equations goes to 1 in LeastSquaresEquations. We can add more
        # structure to differentiate between inherited and changed values, but this would require a
        # rewrite of the other parts of code. Since this problem occurs only in this module, we will
        # just patch in the fix here.
        orig_num_eqns = LeastSquaresEquations.num_eqns
        LeastSquaresEquations.num_eqns = SystemEquations.num_eqns

        system_eqns = super().objective(params)

        # patch back
        LeastSquaresEquations.num_eqns = orig_num_eqns

        return np.sum(system_eqns ** 2)

    def gradient(self, params):
        r"""Gradient of the objective function.

        If :math:`f(\vec{x})` is the objective function that corresponds to the system of equations,
        :math:`\{f_1(\vec{x}), f_2(\vec{x}), \dots, f_K(\vec{x})\}`, i.e.

        .. math::

            f(\vec{x}) =
            f_1^2(\vec{x}) + f_2^2(\vec{x}) + \dots + f_K^2(\vec{x}) + f_{constraint}(\vec{x})^2

        the gradient is

        .. math::

            G_j(\vec{x}) &= \frac{\partial f(\vec{x})}{\partial x_j}\\
            &= 2 f_1(\vec{x}) \frac{\partial f_1(\vec{x})}{\partial x_j}
            + 2 f_2(\vec{x}) \frac{\partial f_2(\vec{x})}{\partial x_j}
            \dots
            + 2 f_K(\vec{x}) \frac{\partial f_K(\vec{x})}{\partial x_j}
            + 2 f_{constraint}(\vec{x}) \frac{\partial f_{constraint}(\vec{x})}{\partial x_j}

        where :math:`\vec{x}` is the parameters at a given iteration.

        Parameters
        ----------
        params : np.ndarray(N,)
            Parameters that describe the system of equations.

        Returns
        -------
        grad : np.ndarray(nparams)
            Value of the gradient at the given parameter values.

        """
        # FIXME: monkey patched property
        # The property num_eqns need to be that of SystemEquations rather than the
        # LeastSquaresEquations for SystemEquations.objective to function properly. This occurs
        # because the number of equations goes to 1 in LeastSquaresEquations. We can add more
        # structure to differentiate between inherited and changed values, but this would require a
        # rewrite of the other parts of code. Since this problem occurs only in this module, we will
        # just patch in the fix here.
        temp = LeastSquaresEquations.num_eqns
        LeastSquaresEquations.num_eqns = SystemEquations.num_eqns

        system_eqns = super().objective(params)
        # FIXME: There is probably a better way to implement this. In jacobian method, all of the
        # integrals needed to create the system_eqns is already available. This implementation would
        # repeat certain operations (but might not be that bad with caching?)
        system_eqns = super().objective(params)[:, np.newaxis]
        system_eqns_jac = super().jacobian(params)

        # patch back
        LeastSquaresEquations.num_eqns = temp

        return 2 * np.sum(system_eqns * system_eqns_jac, axis=0)
