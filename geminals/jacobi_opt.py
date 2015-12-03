#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import numpy as np
from copy import deepcopy as copy
from scipy.optimize import minimize_scalar

class JacobiOptimizer(object):
    """Jacobi-like optimizer of a function of a vector (or one-index array).

    Attributes
    ----------
    twopi : np.float64
        The constant 2*pi
    matrix : str
        The type of Jacobi matrix that will be used to transform the vector argument of
        the objective function.  One of 'nonorthogonal', 'orthogonal'.
    trans : callable
        The callable that applies the Jacobi matrix to the vector.
    trans_der : callable
        The callable that applies the derivative of the Jacobi matrix with
        respect to its parameter(s) to the vector.
    searcher : callable
        The type of one-dimensional search that will be used to determine the optimal
        value of the Jacobi matrix's parameters.  Not implemented yet.
    result : dict
        See scipy.optimize.OptimizeResult.

    Methods
    -------
    __call__(x0, fun, tol=1.0e-6, persist=100, maxiter=100000) :
        Performs the optimization.

    """

    # Define constants
    twopi = 2*np.pi

    def __init__(self, matrix='nonorthogonal', searcher=minimize_scalar):
        """Initializes the JacobiOptimizer instance.

        Parameters
        ----------
        matrix : str, optional
            The type of Jacobi matrix that will be used to transform the vector argument of
            the objective function.  One of 'nonorthogonal', 'orthogonal'.
        searcher : callable, optional
            The type of one-dimensional search that will be used to determine the optimal
            value of the Jacobi matrix's parameters.  Not implemented yet.

        Raises
        ------
        NotImplementedError :
            For the values of `matrix` that have not yet been implemented.

        """

        # Define the Jacobi-like transformation and its derivative w.r.t. theta
        if matrix is 'nonorthogonal':
            def trans(theta, x0, i, j):
                x0[i] = np.sin(theta)*x0[i] + np.cos(theta)*x0[j]
                x0[j] = np.sin(theta)*x0[j] + np.cos(theta)*x0[i]
        
            def trans_der(theta, x0, i, j):
                x0[i] = np.cos(theta)*x0[i] - np.sin(theta)*x0[j]
                x0[j] = np.cos(theta)*x0[j] - np.sin(theta)*x0[i]
        # elif matrix is 'orthogonal':
        #   ...
        else:
            raise NotImplementedError

        self.trans = trans
        self.trans_der = trans_der
        self.searcher = searcher
        self.result = None


    def __call__(self, x0, fun, tol=1.0e-6, persist=100, maxiter=100000):
        """
        Performs the Jacobi-like optimization of a one-index array, given an initial guess
        and an objective function of the array to minimize.
    
        Parameters
        ----------
        x0 : one-index np.ndarray, dtype=np.float64
            An initial guess at the optimal value.
        fun : callable
            A callable with a one-index array of `x0`'s size as a parameter, and that returns a
            scalar.
        tol : float, optional
            If `x0` does not change by at least this much over one iteration, the persistence
            counter goes up by one.  If it does change by at least `tol', the persistence
            counter is reset to zero.
        persist : int, optional
            If the persistence counter reaches this number, the system is considered converged
            and the function returns successfully.
        maxiter : int, optional
            If convergence has not been reached within this many iterations, the system
            returns a failure state.
    
        Returns
        -------
        result : dict
            See scipy.optimize.OptimizeResult.
    
        """
    
        # Initialize some variables
        nit = 0
        pcount = 0
        i_min = j_min = -1
        perms = [(i, j) for j in range(len(x0)) for i in range(len(x0))]
        fx_last = np.inf
        x0_last = np.array([np.inf for i in range(len(x0))])
        
        # Define an objective function of theta for the SciPy minimizer that modifies
        # `x0` elements in place
        def f(theta, x0, i, j):
            x0_i_backup, x0_j_backup = copy(x0[i]), copy(x0[j])
            self.trans(theta, x0, i, j)
            fx = fun(x0)
            x0[i], x0[j] = x0_i_backup, x0_j_backup
            return fx
    
        # Start iterating
        while nit < maxiter: 

            # For every pairwise transformation...
            fx_min, theta_min = fx_last, 0.0
            for (i, j) in perms:
                # Don't repeat the previous transformation
                if (i, j) == (i_min, j_min):
                    continue
                # Sort out the best transformation
                result = self.searcher( f, args=(x0, i, j), 
                                        method='bounded', bounds=(0.0, self.twopi) )
                if not result['success']:
                    continue
                else:
                    fx = result['fun']
                    theta = result['x']
                    if fx < fx_min:
                        fx_min = fx
                        theta_min = result['x']
                        i_min, j_min = i, j
    
            # Apply the transformation if it is significant
            nit += 1
            if not np.isclose(theta_min, 0.0, rtol=0.001):
                self.trans(theta_min, x0, i_min, j_min)
    
                # Check for convergence
                if np.allclose(x0, x0_last, atol=tol):
                    pcount += 1
                    if pcount > persist:
                        break
                else:
                    # Keep iterating
                    pcount = 0
                    fx_last = fx_min
                    del x0_last
                    x0_last = copy(x0)
    
        # While loop ended because we did not converge
        else:
            self.result = { 'success': False,
                            'status': 1,
                            'fun': fx_min,
                            'nit': nit,
                          }
    
        # Outside of while/else construction; optimization succeeded
        self.result = { 'success': True,
                        'status': 0,
                        'fun': fx_min,
                        'nit': nit,
                      }

        return self.result

# vim: set textwidth=90 :
