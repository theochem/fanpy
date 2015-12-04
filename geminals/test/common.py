#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import sys
import numpy as np

#
# Utilities for running tests
#

_common_tests = []

def _test_decorator(obj, slow=False):
    """Decorator to add a function to the test suite and, optionally, to mark it as a slow
    test.
    """
    def test_wrapper(*args, **kwargs):
        return obj(*args, **kwargs)
    if slow:
        setattr(test_wrapper, '_slowtest', True)
    _common_tests.append(test_wrapper)
    return test_wrapper

test = lambda obj : _test_decorator(obj, slow=False)
slow = lambda obj : _test_decorator(obj, slow=True)
fast = test

def _parse_test_opts():
    opts = {'fast': True, 'slow': False}
    if '--slow' in sys.argv or '-s' in sys.argv:
        opts['fast'] = False
        opts['slow'] = True
    elif '--all' in sys.argv or '-a' in sys.argv:
        opts['slow'] = True
    elif len(sys.argv) > 1:
        print("\nUsage: {} (--slow/-s | --all/-a).\n".format(sys.argv[0]))
        sys.exit(1)
    return opts

def run_tests():
    """Run fast and/or slow tests depending on whether or not they have the 'SLOWTEST'
    attribute.
    """
    opts = _parse_test_opts()
    fast = opts['fast']
    slow = opts['slow']
    for test in _common_tests:
        try:
            run = getattr(test, '_slowtest') and slow
        except AttributeError:
            run = fast
        finally:
            if run:
                test()

#
# Testing functions
#

def check_if_exception_raised(func, exception):
    """ Passes if given exception is raised

    Parameter
    ---------
    func : function object
        Function that contains the desired test code
    exception : Exception
        Exception that is raised

    Returns
    -------
    bool
        True if Exception is raised
        False if Exception is not raised
    """
    try:
        func()
    except exception:
        return True
    except:
        return False
    else:
        return False

# Stolen from Toon's romin (https://github.com/QuantumElephant/romin)
def deriv_error(f, g, x, eps_x=1e-4, order=8):
    '''Compute the error between difference of f at two points and its FD approximation

    Parameters
    ----------
    f : function
        Computes the function value for a given x.
    g : function
        Computes the derivative for a given x.
    x : float
        The center of the interval at which the test is performed.
    eps_x : float
            The half width of the interval. [default=1e-4]
    order : int (2, 4, 8, 16)
            The number of grid points in the quadrature. [default=8]

    This function computes the difference of f(x+eps_x) - f(x-eps_x). It also computes
    the integral of the derivative with Gaussian quadrature, which should be very close
    to the former.

    The functions f and g may return scalars or arrays. The return values will have
    compatible data types.

    Returns
    -------
    delta : float or np.ndarray
            The difference between f at the end points of the interval.
    delta_approx : float or np.ndarray
                   The approximation of delta computed with the derivative, ``g``.
    '''
    # Get the right quadrature points and weights
    if order not in gauss_legendre:
        raise ValueError('The order must be one of {}'.format(gauss_legendre.keys()))
    points, weights = gauss_legendre.get(order)
    # Compute the difference between f at two different points
    delta = f(x + eps_x) - f(x - eps_x)
    # Approximate that difference with Gaussian quadrature, with some sanity checks
    derivs = np.array([g(x + eps_x*p) for p in points])
    assert delta.shape == derivs.shape[1:delta.ndim+1]
    if len(derivs.shape) > 1:
        assert derivs.shape[1:] == delta.shape
    delta_approx = np.tensordot(weights, derivs, axes=1)*eps_x
    # Done
    return delta, delta_approx

# Stolen from Toon's romin (https://github.com/QuantumElephant/romin)
def deriv_error_array(f, g, x, eps_x=1e-4, order=8, nrep=None):
    '''Extension of deriv_error for functions that take arrays as arguments

    This function performs many one-dimensional tests with deriv_error along randomly
    chosen directions.

    Parameters
    ----------
    f : function
        Computes the function value for a given x.
    g : function
        Computes the derivative for a given x.
    x : np.ndarray
        The reference point for multiple calls to deriv_error.
    eps_x : float
            The half width of the interval for deriv_error. [default=1e-4]
    order : int (2, 4, 8, 16)
            The number of grid points in the quadrature. [default=8]
    nrep : int
           The number of random directions. [default=x.size**2]

    Returns
    -------
    delta : float or np.ndarray
            The difference between f at the end points of the interval, for multiple
            random directions
    delta_approx : float or np.ndarray
                   The approximation of delta computed with the derivative, ``g``, for
                   multiple random directions.
    '''
    if nrep is None:
        nrep = x.size**2
    # run different random line scans
    results = []
    for irep in xrange(nrep):
        # Generate a generalized random unit vector.
        while True:
            unit = np.random.normal(0, 1, x.shape)
            norm = np.sqrt((unit**2).sum())
            if norm > 1e-3:
                unit /= norm
                break
        # Define f and g along the one-dimensional scan
        def f_scan(x_scan):
            return f(x_scan*unit + x)
        def g_scan(x_scan):
            # nasty chain rule
            return np.tensordot(g(x_scan*unit + x), unit, axes=unit.ndim)
        # Collect results
        results.append(deriv_error(f_scan, g_scan, 0.0, eps_x, order))
    return results

# Stolen from Toon's romin (https://github.com/QuantumElephant/romin)
def deriv_check(f, g, xs, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3, discard=0.1,
                verbose=False):
    '''Checker for the implementation of partial derivatives

    Parameters
    ----------
    f : function
        Computes the function value for a given x.
    g : function
        Computes the derivative for a given x.
    x : np.ndarray
        The reference point for multiple calls to deriv_error.
    eps_x : float
            The half width of the interval for deriv_error. [default=1e-4]
    order : int (2, 4, 8, 16)
            The number of grid points in the quadrature. [default=8]
    nrep : int
           The number of random directions. [defaults=x.size**2]
    rel_ftol : float
              The allowed relative error between delta and delta_approx. [default=1e-3]
    discard : float
              The fraction of smallest deltas to discard, together with there
              corresponding deltas_approx. [default=0.1]
    verbose : bool
              If True, some info is printed on screen. [default=False].

    Raises
    ------
    AssertionError when any of the selected (delta, delta_approx) have a relative error
    larger than the given ``rel_ftol``.
    '''
    results = []
    if isinstance(xs, float) or isinstance(xs, np.ndarray):
        xs = [xs]
    for x in xs:
        if isinstance(x, float):
            results.append(deriv_error(f, g, x, eps_x, order))
        elif isinstance(x, np.ndarray):
            results.extend(deriv_error_array(f, g, x, eps_x, order, nrep))
        else:
            raise NotImplementedError
    # make arrays
    deltas = np.array([item[0] for item in results]).ravel()
    deltas_approx = np.array([item[1] for item in results]).ravel()
    # sort
    order = deltas.argsort()
    deltas = deltas[order]
    deltas_approx = deltas_approx[order]
    # chop part of
    ndiscard = int(len(deltas)*discard)
    deltas = deltas[ndiscard:]
    deltas_approx = deltas_approx[ndiscard:]
    # some info on screen
    if verbose:
        print('Deltas: {}'.format(deltas))
        print('Approx. deltas: {}'.format(deltas_approx))
        print('Number of comparisons: {:d}'.format(len(deltas)))
        ratios = abs(deltas - deltas_approx)/abs(deltas)
        print('Intrmdt: {}'.format(deltas - deltas_approx))
        print('Ratios: {}'.format(ratios))
        print('Best:  {:10.3e}'.format(ratios.min()))
        print('Worst: {:10.3e}'.format(ratios.max()))
        #abs(deltas - deltas_approx) < rel_ftol*abs(deltas)
    # final test
    assert np.all(abs(deltas - deltas_approx) <= rel_ftol*abs(deltas))

# Stolen from Toon's romin (https://github.com/QuantumElephant/romin)
# Gauss-Legendre quadrature grids (points and weights) for different orders.
gauss_legendre = {
     2: (np.array([-5.773502691896258e-01, 5.773502691896257e-01]),
         np.array([1.000000000000000e+00, 1.000000000000000e+00])),
     4: (np.array([-8.611363115940527e-01, -3.399810435848563e-01, 3.399810435848563e-01,
                   8.611363115940526e-01]),
         np.array([3.478548451374538e-01, 6.521451548625461e-01, 6.521451548625461e-01,
                   3.478548451374538e-01])),
     8: (np.array([-9.602898564975363e-01, -7.966664774136268e-01, -5.255324099163290e-01,
                   -1.834346424956498e-01, 1.834346424956498e-01, 5.255324099163289e-01,
                   7.966664774136267e-01, 9.602898564975362e-01]),
         np.array([1.012285362903763e-01, 2.223810344533745e-01, 3.137066458778873e-01,
                   3.626837833783619e-01, 3.626837833783619e-01, 3.137066458778873e-01,
                   2.223810344533745e-01, 1.012285362903763e-01])),
    16: (np.array([-9.894009349916499e-01, -9.445750230732326e-01, -8.656312023878318e-01,
                   -7.554044083550031e-01, -6.178762444026438e-01, -4.580167776572274e-01,
                   -2.816035507792589e-01, -9.501250983763744e-02, 9.501250983763743e-02,
                   2.816035507792589e-01, 4.580167776572274e-01, 6.178762444026437e-01,
                   7.554044083550030e-01, 8.656312023878316e-01, 9.445750230732325e-01,
                   9.894009349916498e-01]),
         np.array([2.715245941175409e-02, 6.225352393864789e-02, 9.515851168249277e-02,
                   1.246289712555339e-01, 1.495959888165767e-01, 1.691565193950025e-01,
                   1.826034150449236e-01, 1.894506104550685e-01, 1.894506104550685e-01,
                   1.826034150449236e-01, 1.691565193950025e-01, 1.495959888165767e-01,
                   1.246289712555339e-01, 9.515851168249277e-02, 6.225352393864789e-02,
                   2.715245941175409e-02])),
}

# vim: set textwidth=90 :
