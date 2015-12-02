#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import numpy as np

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
    tol : float
        If `fun` does not change by at least this much between iterations, then the system
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

    defaults = { 'tol': 1.0e-6,
                 'maxiter': 100,
                 'disp': False,
               }
    defaults.update(options)
    tol = defaults['tol']
    maxiter = defaults['maxiter']
    disp = defaults['disp']
    if disp:
        def display(it, fval, dt):
            print("Iter: {},\t|F(x)| = {},\tChange: {}".format(it, abs(fval), dt))
    else:
        def display(*anything):
            pass
    if args:
        objective = lambda x: fun(x, *args)
        jacobian = lambda x: jac(x, *args)
    else:
        objective = fun
        jacobian = jac
    success = True
    nit = maxiter
    fx = jx = None

    for i in range(maxiter):

        # Get fun(x) first, so we can check convergence
        fx = objective(x0)
        delta = np.sqrt(np.abs(fx.dot(fx)))
        display(i, fx, delta)
        if delta < tol:
            nit = i + 1
            break

        # Not converged, so we need jac(x)
        jx = jacobian(x0)

        # x_new = x_old - inv(J(x))*F(x)
        # J(x)*(x_old - x_new) = F(x) ==> of the form Ax = b
        try:
            x0 -= np.linalg.solve(jx,fx)
        except np.linalg.linalg.LinAlgError:
            # Jacobian went singular (this happens for some functions around the
            # solution...); use the pseudoinverse from SVD instead
            u, jxinv, v = np.linalg.svd(jx)
            jxinv **= -1
            jxinv = u.dot(jxinv.dot(v))
            x0 -= jxinv.dot(fx)

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
