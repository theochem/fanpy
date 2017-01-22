""" Solver for the ProjectedWavefunction

Function
--------
solve(proj_wfn, solver_type='least_squares', use_jac=True, save_file=None, **kwargs)
    Solves a given Projectedfunction
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from scipy.optimize import root, least_squares, minimize
from paraopt.cma import fmin_cma
from .proj_wavefunction import ProjectedWavefunction

def solve(proj_wfn, solver_type='least_squares', use_jac=True, save_file=None, **kwargs):
    """ Optimize `self.objective(params)` to solve the coefficient vector.

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
        `variational` minimizes :math:`\frac{\braket{\Phi | H | \Psi}}{\braket{\Psi | \Psi}}` where
        :math:`\Phi` is a truncation of :math:`\Psi` (onto the reference Slater determinants)
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
    if solver_type not in ['least_squares', 'root', 'cma', 'cma_guess', 'variational']:
        raise TypeError('Given solver type, {0}, is not supported'.format(solver_type))
    if solver_type != 'least_squares' and proj_wfn.dtype == np.complex128:
        raise NotImplementedError('Complex numbers are only supported for `least_squares` solver')

    # set default options and solver
    solver = None
    objective = None
    options = {}
    if solver_type == 'least_squares':
        solver, objective, options = get_least_squares(proj_wfn, use_jac, save_file)
    elif solver_type == 'root':
        solver, objective, options = get_root(proj_wfn, use_jac, save_file)
    elif solver_type == 'cma':
        solver, objective, options = get_cma(proj_wfn, use_jac, save_file, is_guess=False)
    elif solver_type == 'cma_guess':
        solver, objective, options = get_cma(proj_wfn, use_jac, save_file, is_guess=True)
    elif solver_type == 'variational':
        if 'ref_sds' in kwargs:
            ref_sds = kwargs['ref_sds']
            del kwargs['ref_sds']
        else:
            ref_sds = None
        solver, objective, options = get_var(proj_wfn, use_jac, save_file, ref_sds=ref_sds)
    # overwrite options with user input
    options.update(kwargs)

    # Solve
    result = solver(objective, proj_wfn.params, **options)
    output = {}
    if solver_type in ['least_squares', 'root', 'variational']:
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
    options = {'xtol': 1.0e-15, 'ftol': 1.0e-15, 'gtol': 1.0e-15, 'max_nfev': 1000*proj_wfn.nparams}
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
    """
    objective = lambda x: proj_wfn.objective(x, weigh_constraints=False, save_file=save_file)
    jac = proj_wfn.jacobian
    if proj_wfn.nproj + proj_wfn._nconstraints != proj_wfn.nparams:
        print('Cannot use root solver with over projection. Truncating projection space.')
        proj_wfn.assign_pspace(proj_wfn.pspace[:proj_wfn.nparams - proj_wfn._nconstraints])
    options = {}
    # if Jacobian included
    if use_jac:
        # Powell's hybrid method (MINPACK)
        options = {'method': 'hybr', 'jac': jac, 'options': {'xtol':1.0e-9}}
    # if Jacobian is not included
    else:
        # Newton-Krylov Quasinewton method
        options = {'method': 'krylov', 'options': {'fatol':1.0e-9, 'xatol':1.0e-7}}
    return root, objective, options

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

def get_var(proj_wfn, use_jac, save_file, ref_sds=None):
    """ Returns the tools needed to run scalar minimization on the energy

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
    variational : function
        Optimization procedure
    objective : function
        Objective function
    options : dict
        Default parameters for the optimizer

    Note
    ----
    Optimization is only valid if the wavefunction is projected on all states
    """
    def energy(params):
        """ Calculates the energy from given a set of parameters
        """
        proj_wfn.assign_params(params)
        proj_wfn.normalize()
        if save_file is not None:
            np.save('{0}_temp.npy'.format(save_file), params)
        return proj_wfn.compute_energy(ref_sds=ref_sds)

    def grad_energy(params):
        """ Calculates the gradient of energy from given a set of parameters
        """
        proj_wfn.assign_params(params)
        proj_wfn.normalize()
        return np.array([proj_wfn.compute_energy(ref_sds=ref_sds, deriv=i) for i in
                         range(params.size)])

    solver = minimize
    objective = energy
    options = {}
    if use_jac:
        options['method'] = 'BFGS'
        options['jac'] = grad_energy
        options['options'] = {'gtol':1e-8}
    else:
        options['method'] = 'Powell'
        options['options'] = {'xtol':1e-9, 'ftol':1e-9}

    return solver, objective, options
