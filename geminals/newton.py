#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import numpy as np


#def newton(fun, x0, jac=None, tol=1.0e-6, maxiter=1000, args=None):
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

    Returns
    -------
    result : dict
        See scipy.optimize.OptimizeResult.

    """

    assert jac is not None, "The Jacobian must be specified for the Newton method."

    defaults = { 'tol': 1.0e-6,
                 'maxiter': 100,
               }
    defaults.update(options)
    tol = defaults['tol']
    maxiter = defaults['maxiter']
    if args:
        objective = lambda x0: fun(x0, *args)
        jacobian = lambda x0: jac(x0, *args)
    else:
        objective = fun
        jacobian = jac

    success = True
    nit = maxiter

    for i in range(maxiter):

        # Get fun(x) first, so we can check convergence
        fx = objective(x0)
        if np.sqrt(np.abs(fx.dot(fx))) < tol:
            nit = i + 1
            break

        # Not converged, so we need jac(x)
        jx = jacobian(x0)

        # x_new = x_old - inv(J(x))*F(x)
        # J(x)*(x_new - x_old) = -F(x) ==> of the form Ax = b
        x0 += np.linalg.solve(jx,-fx)

    # If for loop finishes, we did not converge
    else:
        success = False

    # Return the results in a SciPy-compatible format
    return { 'x': x0,
             'success': success,
             'fun': fx,
             'jac': jx,
             'nit': nit,
           }


# vim: set textwidth=90 :
