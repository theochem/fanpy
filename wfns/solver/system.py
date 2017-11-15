"""Solvers for system of Schrodinger equations."""
from wfns.objective.schrodinger.least_squares import SystemEquations
from wfns.solver.wrappers import wrap_scipy


def least_squares(objective, save_file='', **kwargs):
    """Solve the system of Schrodinger equations as a least squares problem.

    This function wraps around `scipy.optimize.least_squares`.

    Parameters
    ----------
    objective : SystemEquations
        Objective that describes the system of equations.
    save_file : str
        File to which the results of the optimization is saved.
        By default, the results are not saved.
    kwargs : dict
        Keyword arguments to `scipy.optimize.least_squares`. See its documentation for details.
        By default, 'xtol', 'ftol', and 'gtol' are set to `1e-15`, 'max_nfev' is 1000 times the
        number of active parameters, and `jac` is the Jacobian provided by the objective.
        If Jacobian from the objective is not used, then `3-point` scheme should be used for
        parameters that are floats and `cs` scheme for parameters that are complex.

    Returns
    -------
    Dictionary with the following keys and values:
    success : bool
        True if optimization succeeded.
    params : np.ndarray
        Parameters at the end of the optimization.
    energy : float
        Energy of the system after optimization
    residuals  : np.ndarray
        Value of each equation in the system.
    cost : float
        Least squared sum of the values of each equation in the system.
    message : str
        Message returned by the optimizer.
    internal : OptimizeResults
        Returned value of the `scipy.optimize.least_squares`.

    Raises
    ------
    TypeError
        If objective is not SystemEquations instance.

    """
    from scipy.optimize import least_squares

    if not isinstance(objective, SystemEquations):
        raise TypeError('Given objective must be an instance of SystemEquations.')

    if kwargs == {}:
        kwargs = {'xtol': 1.0e-15, 'ftol': 1.0e-15, 'gtol': 1.0e-15,
                  'max_nfev': 1000*objective.params.size, 'jac': objective.jacobian}

    output = wrap_scipy(least_squares)(objective, save_file=save_file, **kwargs)
    output['energy'] = objective.energy.params
    output['residuals'] = output['internal'].fun
    output['cost'] = output['internal'].cost
    return output


def root(objective, save_file='', **kwargs):
    """Solve the system of Schrodinger equations by finding its root.

    This function wraps around `scipy.optimize.root`.

    Parameters
    ----------
    objective : SystemEquations
        Objective that describes the system of equations.
    save_file : str
        File to which the results of the optimization is saved.
        By default, the results are not saved.
    kwargs : dict
        Keyword arguments to `scipy.optimize.least_squares`. See its documentation for details.
        By default, 'method' is 'hybr', 'jac' is the Jacobian provided, and 'options' is set to
        `{'xtol': 1e-9}`.
        by the objective.
        If Jacobian from the objective is not used, then the preferred keyword argument for 'method'
        is 'krylov' and 'option' is `{'fatol': 1.0e-9, 'xatol': 1.0e-7}`.

    Returns
    -------
    Dictionary with the following keys and values:
    success : bool
        True if optimization succeeded.
    params : np.ndarray
        Parameters at the end of the optimization.
    energy : float
        Energy of the system after optimization
    message : str
        Message returned by the optimizer.
    internal : OptimizeResults
        Returned value of the `scipy.optimize.root`.

    Raises
    ------
    TypeError
        If objective is not SystemEquations instance.

    """
    from scipy.optimize import root

    if not isinstance(objective, SystemEquations):
        raise TypeError('Given objective must be an instance of SystemEquations.')
    elif objective.num_eqns != objective.params.size:
        raise ValueError('Given objective must have the same number of equations as the number of '
                         'parameters.')

    if kwargs == {}:
        kwargs = {'method': 'hybr', 'jac': objective.jacobian, 'options': {'xtol': 1.0e-9}}

    output = wrap_scipy(root)(objective, save_file=save_file, **kwargs)
    output['energy'] = objective.energy.params
    return output
