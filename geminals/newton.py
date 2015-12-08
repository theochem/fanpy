#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import numpy as np
from copy import deepcopy as copy
from scipy.optimize import minimize_scalar, minimize

def newton(fun, x0, jac=None, args=None, **options):
    """
    Finds the roots of a non-linear set of equations using the Newton-Raphson method,
    given the multi-dimensional function, its Jacobian, and an initial guess `x0'.

    Parameters
    ----------
    fun : callable
        A callable that takes a one-index np.ndarray as an argument.
    x0 : one-index np.ndarray
        The initial guess for the solution.
    jac : callable
        A callable representing the Jacobian of `fun`.  It takes a takes a one-index
        np.ndarray as an argument, and returns a two-index len(x)-by-len(x) np.ndarray.
    args : unpack-able, optional
        Extra arguments to be passed to `fun`.
    options : dict, optional
        Some Newton-solver specific options that have default values.

    Options
    -------
    ftol : float
        If `fun` does not change by at least this much between iterations, then the system
        is considered converged and this function returns successfully.
    xtol : float
        If `x0` does not change by at least this much between iterations, then the system
        is considered converged and this function returns successfully.
    maxiter : int
        The maximum number of iterations this function will try before giving up.
    disp : bool
        If True, the solver will print out its progress at every iteration.

    Returns
    -------
    result : dict
        See scipy.optimize.OptimizeResult.

    """

    assert jac, "The Jacobian must be specified for the Newton method."

    defaults = { 'ftol': 1.0e-6,
                 'xtol': 1.0e-6,
                 'maxiter': 10000,
                 'disp': False,
               }
    defaults.update(options)
    ftol = defaults['ftol']**2
    xtol = defaults['xtol']**2
    maxiter = defaults['maxiter']
    disp = defaults['disp']

    if disp:
        def display(it, fval, df, dx):
            print("Iter: {},\t|F(x)| = {},\tF change: {},\tx change: {}" \
                .format(it, abs(fval), df, dx))
    else:
        def display(*anything):
            pass

    if args:
        objective = lambda x: fun(x, *args)
        jacobian = lambda x: jac(x, *args)
    else:
        objective = fun
        jacobian = jac

    def obj_scalar(x):
        tmp = objective(x)
        return 0.5*np.abs(tmp.dot(tmp))

    def gradient(x):
        tmp = jacobian(x)
        if len(tmp.shape) > 1:
            return np.array([ np.abs(tmp[:,i].dot(tmp[:,i])) for i in range(tmp.shape[1]) ])
        else:
            return np.abs(tmp.dot(tmp))

    success = True
    nit = maxiter
    fx_old = np.inf
    x0_old = np.inf

    for iter in range(maxiter):

        fx = objective(x0)
        jx = jacobian(x0)

        deltaf = np.abs(fx - fx_old)
        deltax = np.abs(x0 - x0_old)
        if deltaf.dot(deltaf) < ftol or deltax.dot(deltax) < xtol:
            nit = iter + 1
            break

        try:
            deltaxvec = np.linalg.solve(jx,-fx)

        except np.linalg.linalg.LinAlgError:
            print("INVERSION EXCEPTION")
            u, jxinv, v = np.linalg.svd(jx)
            jxinv **= -1
            jxinv = u.dot(jxinv.dot(v))
            deltaxvec = jxinv.dot(-fx)

        def search_obj(alpha):
            return obj_scalar(x0 + alpha*deltaxvec)

        search_result = minimize_scalar(search_obj,tol=1.0e-12)#,bounds=(-0.1,0.1))
        alpha = search_result.x
        print(alpha)

        fx_old = copy(fx)
        x0_old = copy(x0)
        x0 += alpha*deltaxvec
        print(x0)

        cond_0_l = obj_scalar(x0 + alpha*deltaxvec)
        cond_0_r = obj_scalar(x0) + 0.1*alpha*gradient(x0).transpose().dot(deltaxvec)
        cond_1_l = np.abs(gradient(x0 + alpha*deltaxvec).transpose().dot(deltaxvec))
        cond_1_r = 0.1*np.abs(gradient(x0).transpose().dot(deltaxvec))
        if np.all(cond_0_l <= cond_0_r) and np.all(cond_1_l <= cond_1_r):
            nit = iter
            break

    else:
        success = False

    return { 'x': x0,
             'success': success,
             'fun': fx,
             'jac': jx,
             'nit': nit,
           }

# vim: set textwidth=90 :
