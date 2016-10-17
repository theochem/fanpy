from __future__ import absolute_import, division, print_function

import os
import numpy as np
from scipy.optimize import root, least_squares

def solve(wavefunction, solver_type='least squares', jac=True, **kwargs):
    """
    Optimize `self.objective(params)` to solve the coefficient vector.

    Parameters
    ----------
    wavefunction : instance of ProjectionWavefunction
        Wavefunction to solve
    solver_type : str
        solver type
        One of 'least squares', 'root', 'cma'
        Default is Least Squares Nonlinear system of equation solver
    jacobian : bool
        Flag for using Jacobian
        Default is including
    kwargs : dict, optional
        Keywords to pass to the internal solver, `scipy.optimize.root`.

    Returns
    -------
    result : tuple
    See `scipy.optimize.root` documentation.
    """
    # check solver_type
    if solver_type not in ['least squares', 'root', 'cma']:
        raise TypeError('Given solver type, {0}, is not supported'.format(solver_type))

    # set options
    options = {}
    options["bounds"] = wavefunction.bounds

    # set default options and solver
    if solver_type == 'root':
        # if Jacobian included
        if jac:
            # Powell's hybrid method (MINPACK)
            options = {
                    "method": 'hybr',
                    "jac": wavefunction.jacobian,
                    "options": {
                        "xtol":1.0e-9,
                        },
                    }
        # if Jacobian is not included
        else:
            # Newton-Krylov Quasinewton method
            options = {
                "method": 'krylov',
                "jac": False,
                "options": {
                    "fatol":1.0e-9,
                    "xatol":1.0e-7,
                    },
                }
        solver = root
    if solver_type == 'least squares':
        # if Jacobian is included
        if jac:
            options = {
                "jac": wavefunction.jacobian,
                "xtol": 1.0e-15,
                "ftol": 1.0e-15,
                "gtol": 1.0e-15,
            }
        # if Jacobian is not included
        else:
            # use appropriate jacobian approximation
            if wavefunction.dtype == np.complex128:
                options["jac"] = "cs"
            else:
                options["jac"] = "3-point"
        solver = least_squares

    # overwrite options with user input
    options.update(kwargs)

    # Solve
    result = solver(wavefunction.objective, wavefunction.params, **options)

    # Save results
    if wavefunction.save_params:
        # FIXME: _temp business is quite ugly
        np.save('{0}.npy'.format(wavefunction.__class__.__name__), wavefunction.params)
        os.remove('{0}_temp.npy'.format(wavefunction.__class__.__name__))

    # warn user if solver didn't converge
    if not result.success:
        print('WARNING: Optimization did not succeed.')

    return result

