""" Solver for the ProjectedWavefunction

Function
--------
solve(proj_wfn, solver_type='least_squares', use_jac=True, save_file=None, **kwargs)
    Solves a given Projectedfunction
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from scipy.optimize import root, least_squares
from paraopt.cma import fmin_cma
from .proj_wavefunction import ProjectedWavefunction

def solve(proj_wfn, solver_type='least_squares', use_jac=True, save_file=None, **kwargs):
    """
    Optimize `self.objective(params)` to solve the coefficient vector.

    Parameters
    ----------
    proj_wfn : instance of ProjectedWavefunction
        Wavefunction to solve
    solver_type : {'least_squares', 'root', 'cma', 'cma_guess'}
        Optimization algorithm
        Default is Least Squares Nonlinear system of equation solver
        `least_squares` uses nonlinear least squares with bounds
        `root` finds the root of a function
        `cma` uses Covarian Matrix Adaptation to solve the condensed system of equations (compressed
        into one equation)
        `cma_guess` uses the `cma` algorithm with more relaxed conditions (used to get intitial
        guess for other algorithms)
    use_jac : bool
        Flag for using Jacobian in the optimization
        By default, the Jacobian is used (not used at all for CMA)
    save_file : str
        Name of the `.npy` file that will be used to store the parameters in the course of the
        optimization
        `.npy` will be appended to the end of the string
        Default is no save file
    kwargs : dict, optional
        Keywords to pass to the internal solver

    Returns
    -------
    result : dict
        Depends on solver
        'success' : bool

    Note
    ----
    For more information on the `root` optimization process, see `scipy.optimize.root`.
    For more information on the `least_squares` optimization process, see
    `scipy.optimize.least_squares`.
    For more information on the `cma` optimization process, see
    `https://en.wikipedia.org/wiki/CMA-ES`.
    """
    # check
    if not isinstance(proj_wfn, ProjectedWavefunction):
        raise TypeError('Given wavefunction must be a `ProjectedWavefunction`')
    if solver_type not in ['least_squares', 'root', 'cma', 'cma_guess']:
        raise TypeError('Given solver type, {0}, is not supported'.format(solver_type))
    if solver_type != 'least_squares' and proj_wfn.dtype == np.complex128:
        raise NotImplementedError('Complex numbers are only supported for `least_squares` solver')

    # set default options and solver
    solver = None
    objective = None
    options = {}
    init_guess = proj_wfn.params
    if solver_type == 'least_squares':
        solver, objective, options = get_least_squares(proj_wfn, use_jac, save_file)
    elif solver_type == 'root':
        solver, objective, options, init_guess = get_root(proj_wfn, use_jac, save_file)
    elif solver_type == 'cma':
        solver, objective, options = get_cma(proj_wfn, use_jac, save_file, is_guess=False)
    elif solver_type == 'cma_guess':
        solver, objective, options = get_cma(proj_wfn, use_jac, save_file, is_guess=True)
    # overwrite options with user input
    options.update(kwargs)

    # Solve
    result = solver(objective, init_guess, **options)
    output = {}
    if solver_type in ['least_squares', 'root']:
        output = {attr : getattr(result, attr) for attr in dir(result)}
    elif solver_type in ['cma', 'cma_guess']:
        output['success'] = (result[1] == 'CONVERGED_WIDTH')

    # warn user if solver didn't converge
    if not output['success']:
        print('WARNING: Optimization did not succeed.')

    # Save results
    if save_file is not None:
        np.save('{0}.npy'.format(save_file), proj_wfn.params)
        os.remove('{0}_temp.npy'.format(save_file))

    return output

def get_least_squares(proj_wfn, use_jac, save_file):
    """ Returns the tools needed to run a least_squares calculation

    Parameters
    ----------
    proj_wfn : instance of ProjectedWavefunction
        Wavefunction to solve
    use_jac : bool
        Flag for using Jacobian in the optimization
        By default, the Jacobian is used (not used at all for CMA)
    save_file : str
        Name of the `.npy` file that will be used to store the parameters in the course of the
        optimization
        `.npy` will be appended to the end of the string
        Default is no save file

    Returns
    -------
    least_squares : function
        Optimization procedure
    objective : function
        Objective function
    options : dict
        Default parameters for the optimizer
    """
    objective = lambda x: proj_wfn.objective(x, weigh_constraints=True, save_file=save_file)
    options = {'xtol': 1.0e-15, 'ftol': 1.0e-15, 'gtol': 1.0e-15}
    # if Jacobian is included
    if use_jac:
        options['jac'] = proj_wfn.jacobian
    # if Jacobian is not included
    elif proj_wfn.dtype == np.complex128:
        options['jac'] = 'cs'
    elif proj_wfn.dtype == np.float64:
        options['jac'] = '3-point'
    return least_squares, objective, options



def get_root(proj_wfn, use_jac, save_file):
    """ Returns the tools needed to run a root calculation

    Parameters
    ----------
    proj_wfn : instance of ProjectedWavefunction
        Wavefunction to solve
    use_jac : bool
        Flag for using Jacobian in the optimization
        By default, the Jacobian is used (not used at all for CMA)
    save_file : str
        Name of the `.npy` file that will be used to store the parameters in the course of the
        optimization
        `.npy` will be appended to the end of the string
        Default is no save file

    Returns
    -------
    root : function
        Optimization procedure
    objective : function
        Objective function
    options : dict
        Default parameters for the optimizer
    init_guess : np.ndarray
        Initial guess for the solver
    """
    objective = lambda x: proj_wfn.objective(x, weigh_constraints=False, save_file=save_file)
    init_guess = proj_wfn.params
    jac = proj_wfn.jacobian
    if proj_wfn.nproj + proj_wfn._nconstraints != proj_wfn.nparams:
        print('Cannot use root solver with over projection. Adding dummy variables.')
        objective = lambda x: proj_wfn.objective(x[:proj_wfn.nparams], weigh_constraints=False,
                                                 save_file=save_file)
        init_guess = np.hstack((init_guess, np.zeros(proj_wfn.nproj + proj_wfn._nconstraints
                                                     - proj_wfn.nparams)))
        jac = lambda x: np.hstack((proj_wfn.jacobian(x, weigh_constraints=False),
                                   np.zeros((proj_wfn.nproj + proj_wfn._nconstraints, proj_wfn.nproj
                                             + proj_wfn._nconstraints - proj_wfn.nparams))))
    options = {}
    # if Jacobian included
    if use_jac:
        # Powell's hybrid method (MINPACK)
        options = {'method': 'hybr', 'jac': jac, 'options': {'xtol':1.0e-9}}
    # if Jacobian is not included
    else:
        # Newton-Krylov Quasinewton method
        options = {'method': 'krylov', 'options': {'fatol':1.0e-9, 'xatol':1.0e-7}}
    return root, objective, options, init_guess

def get_cma(proj_wfn, use_jac, save_file, is_guess=False):
    """ Returns the tools needed to run a cma calculation

    Parameters
    ----------
    proj_wfn : instance of ProjectedWavefunction
        Wavefunction to solve
    use_jac : bool
        Flag for using Jacobian in the optimization
        By default, the Jacobian is used (not used at all for CMA)
    save_file : str
        Name of the `.npy` file that will be used to store the parameters in the course of the
        optimization
        `.npy` will be appended to the end of the string
        Default is no save file
    is_guess : bool
        True if result from this optimization will be used as a guess for another optimization
        False otherwise

    Returns
    -------
    cma : function
        Optimization procedure
    objective : function
        Objective function
    options : dict
        Default parameters for the optimizer
    """
    # objective = lambda x: np.sum(proj_wfn.objective(x, weigh_constraints=False,
    #                                                 save_file=save_file)
    #                              *np.hstack(([proj_wfn.get_overlap(i) for i in proj_wfn.pspace],
    #                                          1)))
    objective = lambda x: np.sum(np.abs(proj_wfn.objective(x, weigh_constraints=False,
                                                           save_file=save_file)))
    if use_jac:
        print('WARNING: Jacobian is not needed in CMA solver')
    options = {}
    options['npop'] = proj_wfn.nparams*2
    options['verbose'] = False
    # different setting for cma and cma_guess
    if is_guess:
        options['sigma0'] = 0.1
        options['max_iter'] = 100
        options['wtol'] = 1e-6
    else:
        options['sigma0'] = 0.001
        options['max_iter'] = 1000
        options['wtol'] = 1e-12
    return fmin_cma, objective, options
