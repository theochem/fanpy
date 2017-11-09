"""Schrodinger equation as a least-squares problem."""
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.objective.system_nonlinear import SystemEquations


@docstring_class(indent_level=1)
class LeastSquaresEquations(SystemEquations):
    r"""Schrodinger equation as a system of equations represented as a least squares problem.

    .. math::

        (\braket{\Phi_1 | \hat{H} | \Psi} - E \braket{\Phi_1 | \Psi})^2 +
        (\braket{\Phi_2 | \hat{H} | \Psi} - E \braket{\Phi_2 | \Psi})^2 +
        \dots
        (\braket{\Phi_K | \hat{H} | \Psi} - E \braket{\Phi_K | \Psi})^2 +
        (f_{constraint}(\Psi, \hat{H}))^2 = 0

    Energy can be a constant, a parameter that gets optimized, or a function of the wavefunction and
    hamiltonian parameters.

    .. math::

        E = \frac{\braket{\Phi_{ref} | \hat{H} | \Psi}{\braket{\Phi_{ref} | \Psi}}

    Additionally, the normalization constraint is added with respect to the reference state.

    """
    @property
    def num_eqns(self):
        return 1

    def objective(self, params):
        r"""Least squares equation that corresponds to the system of equations.

        .. math::

            f(\vec{x}) &=
            (\braket{\Phi_1 | \hat{H} | \Psi} - E \braket{\Phi_1 | \Psi})^2 +
            (\braket{\Phi_2 | \hat{H} | \Psi} - E \braket{\Phi_2 | \Psi})^2 +
            \dots
            (\braket{\Phi_K | \hat{H} | \Psi} - E \braket{\Phi_K | \Psi})^2 +
            (f_{constraint}(\vec{x})^2\\
            &= f_1^2(\vec{x}) + f_2^2(\vec{x}) + \dots + f_K^2(\vec{x}) +
            f_{constraint}(\vec{x})^2\\

        where :math:`f_i` is the ith equation of the system of equations, :math:`K` is the number of
        states onto which the wavefunction is projected. The :math:`K+1`th equation correspond to
        constraints on the system of equations. We currently use only the normalization constraint.
        The norm is computed with respect to the reference state. The energy can be a constant, a
        parameter that gets optimized, or a function of the wavefunction and hamiltonian parameters.

        .. math::

            E = \frac{\braket{\Phi_{ref} | \hat{H} | \Psi}{\braket{\Phi_{ref} | \Psi}}

        Parameters
        ----------
        params : np.ndarray(N,)
            Parameters that describe the system of equations.

        Returns
        -------
        objective : float
            Output of the function that will be optimized.

        """
        system_eqns = super().objective(params)
        return np.sum(system_eqns**2)

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
        # FIXME: There is probably a better way to implement this. In jacobian method, all of the
        # integrals needed to create the system_eqns is already available. This implementation would
        # repeat certain operations (but might not be that bad with caching?)
        system_eqns = super().objective(params)[:, np.newaxis]
        system_eqns_jac = super().jacobian(params)
        return 2 * np.sum(system_eqns * system_eqns_jac, axis=0)
