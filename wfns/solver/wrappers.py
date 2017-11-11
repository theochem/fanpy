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
        Output is OptimizeResults instance.
        Second argument must be the initial guess (rather than boundary or ranges).

    Returns
    -------
    new_func(objective, save_file='', **kwargs) : function
        Wrapped scipy function.
        Output is a dictionary that is consistent with other solvers.
        'save_file' is the name of the file that will be saved.

    Examples
    --------
    >>> least_squares = wrap_scipy(least_squares)

    """
    @wraps(func)
    def new_func(objective, save_file='', **kwargs):
        results = func(objective.objective, objective.params, **kwargs)

        output = {}
        output['success'] = results.success
        output['params'] = results.x
        output['message'] = results.message
        output['internal'] = results

        if save_file != '':
            np.save(save_file, objective.all_params)

        return output
    return new_func
