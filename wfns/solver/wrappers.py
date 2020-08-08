"""Wrapper for solvers from other modules.

Since each module uses a different API for its solvers, the input and output of these solvers are
wrapped such that they all have the same API within the `wfns.solver` module.

"""
from functools import wraps

import numpy as np


def wrap_scipy(func):
    """Convert solvers from `scipy.optimize` to a consistent format.

    Following methods are supported: `minimize`, `fmin`, `fmin_powell`, `fmin_cg`, `fmin_bfgs`,
    `fmin_l_bfgs_b`, `fmin_tnc`, `fmin_slsqp`, `leastsq`, `least_squares`, `basin_hopping`, `root`,
    `fsolve`, `broyden1`, `broyden2`, `newton_krylov`, `anderson`, `excitingmixing`, `linearmixing`,
    and `diagbroyden`. Other methods, such as `fmin_ncg` and `fmin_cobyla`, must be wrapped manually
    for consistent API with the others solvers in `wfns.solver`.

    Parameters
    ----------
    func(objective, x0, **kwargs) : function
        Scipy function that will be used to optimize the objective.
        Output is OptimizeResult instance.
        Second argument must be the initial guess (rather than boundary or ranges).

    Returns
    -------
    new_func(objective, save_file='', **kwargs) : function
        Wrapped scipy function.
        Output is a dictionary that is consistent with other solvers.
        'save_file' is the name of the file that will be saved.

    Examples
    --------
    >>> least_squares = wrap_scipy(scipy.optimize.least_squares)

    """

    @wraps(func)
    def new_func(objective, save_file="", **kwargs):
        """Solve the objective using the given function.

        This function wraps around `func`.

        Parameters
        ----------
        objective : BaseSchrodinger
            Objective.
        save_file : str
            File to which the results of the optimization is saved.
            By default, the results are not saved.
        kwargs : dict
            Keyword arguments to the solver. See its documentation for details.

        Returns
        -------
        Dictionary with the following keys and values:
        success : bool
            True if optimization succeeded.
        params : np.ndarray
            Parameters at the end of the optimization.
        message : str
            Message returned by the optimizer.
        internal : OptimizeResults
            Returned value of the `scipy.optimize.root`.

        """
        results = func(objective.objective, objective.active_params, **kwargs)

        output = {}
        output["success"] = results.success
        output["params"] = results.x
        output["message"] = results.message
        output["internal"] = results

        if save_file != "":
            np.save(save_file, output["params"])

        return output

    return new_func


def wrap_skopt(func):
    """Convert solvers from `skopt` to a consistent format.

    Following methods are supported: `dummy_minimize`, `forest_minimize`, `gbrt_minimize`, and
    `gp_minimize`.

    Parameters
    ----------
    func(objective, dimensions, **kwargs) : function
        Scikit-optimize function that will be used to optimize the objective.
        Output is OptimizeResult instance.
        Second argument must be the initial guess (rather than boundary or ranges).

    Returns
    -------
    new_func(objective, save_file='', **kwargs) : function
        Wrapped scikit-optimize function.
        Output is a dictionary that is consistent with other solvers.
        'save_file' is the name of the file that will be saved.

    Examples
    --------
    >>> forst_minimize = wrap_scipy(skopt.optimizer.forest_minimize)

    """

    @wraps(func)
    def new_func(objective, save_file="", **kwargs):
        """Solve the objective using the given function.

        This function wraps around `func`.

        Parameters
        ----------
        objective : BaseSchrodinger
            Objective.
        save_file : str
            File to which the results of the optimization is saved.
            By default, the results are not saved.
        kwargs : dict
            Keyword arguments to the solver. See its documentation for details.

        Returns
        -------
        Dictionary with the following keys and values:
        success : bool
            True if optimization succeeded.
        params : np.ndarray
            Parameters at the end of the optimization.
        message : str
            Message returned by the optimizer.
        internal : OptimizeResults
            Returned value of the `scipy.optimize.root`.

        """
        if "dimensions" not in kwargs:
            kwargs["dimensions"] = [(i - 0.5, i + 0.5) for i in objective.active_params]
        results = func(objective.objective, **kwargs)

        output = {}
        # output["success"] = results.success
        output["params"] = results.x
        # output["message"] = results.message
        output["internal"] = results

        if save_file != "":
            np.save(save_file, output["params"])

        return output

    return new_func
