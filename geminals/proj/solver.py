from __future__ import absolute_import, division, print_function

import os
import numpy as np
from scipy.optimize import root, least_squares
from paraopt.cma import fmin_cma

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
    if solver_type not in ['least squares', 'root', 'cma', 'cma_guess']:
        raise TypeError('Given solver type, {0}, is not supported'.format(solver_type))

    # set options
    options = {}

    # set default options and solver
    if solver_type == 'root':
        solver = root
        objective = wavefunction.objective
        options["bounds"] = wavefunction.bounds
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
    elif solver_type == 'least squares':
        solver = least_squares
        objective = wavefunction.objective
        options["bounds"] = wavefunction.bounds
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
    elif 'cma' in solver_type:
        solver = fmin_cma
        objective = lambda x: np.sum(np.abs(wavefunction.objective(x, weigh_norm=False)))
        if jac:
            print('WARNING: Jacobian is not needed in CMA solver')
        options['npop'] = wavefunction.nparam*2
        if solver_type == 'cma':
            options['sigma0'] = 0.01
            options['verbose'] = True
            options['max_iter'] = 1000
        elif solver_type == 'cma_guess':
            options['sigma0'] = 0.1
            options['verbose'] = False
            options['max_iter'] = 50

    # overwrite options with user input
    options.update(kwargs)

    # Solve
    result = solver(objective, wavefunction.params, **options)

    # warn user if solver didn't converge
    if solver_type in ['least squares', 'root'] and not result.success:
        print('WARNING: Optimization did not succeed.')
    if solver_type in ['cma'] and result[1] != 'CONVERGED_WIDTH':
        print('WARNING: Optimization did not succeed.')

    # Save results
    if wavefunction.save_params:
        # FIXME: _temp business is quite ugly
        np.save('{0}.npy'.format(wavefunction.__class__.__name__), wavefunction.params)
        os.remove('{0}_temp.npy'.format(wavefunction.__class__.__name__))

    return result

