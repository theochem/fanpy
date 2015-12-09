"""
Unit testing infrastructure and functions used over multiple unit test files.

"""

from __future__ import absolute_import, division, print_function
import sys
import numpy as np

#
# Utilities for running tests
#

_common_tests = []


def _test_decorator(obj, mark=False):
    """
    Decorator to add a function to the test suite and, optionally, to mark it as a slow
    test.

    Parameters
    ----------
    obj : callable
        A callable object to decorate.
    mark : bool, optional
        Whether to mark the test as slow.  Defaults to False.

    Returns
    -------
    test_wrapper : function
        The decorated callable.

    """

    def test_wrapper(*args, **kwargs):
        return obj(*args, **kwargs)

    if mark:
        setattr(test_wrapper, "_slowtest", True)
    _common_tests.append(test_wrapper)
    return test_wrapper


# Easier aliases for _test_decorator()
test = lambda obj: _test_decorator(obj, mark=False)
slow = lambda obj: _test_decorator(obj, mark=True)


def _parse_test_options():
    """
    Parse the command line options to determine whether to run fast and slow tests.

    Returns
    -------
    options : dict
        Contains "slow" and "fast" with bool values.

    Notes
    -----
    If invalid options are given, this function terminates Python with return value 1.

    """
    options = {"fast": True, "slow": False}
    if "--slow" in sys.argv or "-s" in sys.argv:
        options["fast"] = False
        options["slow"] = True
    elif "--all" in sys.argv or "-a" in sys.argv:
        options["slow"] = True
    elif len(sys.argv) > 1:
        print("\nUsage: {} (--slow/-s | --all/-a).\n".format(sys.argv[0]))
        sys.exit(1)
    return options


def run_tests():
    """
    Run fast and/or slow tests.

    """
    options = _parse_test_options()
    for fun in _common_tests:
        try:
            run = getattr(fun, "_slowtest") and options["slow"]
        except AttributeError:
            run = options["fast"]
        finally:
            if run:
                fun()


#
# Testing functions
#

def raises_exception(fun, exception=Exception):
    """
    Check if a test function (with no arguments) raises an exception.

    Parameters
    ----------
    fun : function
        The function that should raise an exception.
    exception : Exception, optional
        The exception that is expected to be raised.  Defaults to the base Exception
        object.

    Returns
    -------
    exception_raised : bool

    """

    try:
        fun()
    except exception:
        return True
    except:
        return False
    else:
        return False


def is_singular(matrix, tol=1.0e-12):
    """
    Check if a matrix is singular.

    Parameters
    ----------
    matrix : 2-index np.ndarray
        The matrix to check.
    tol : float, optional
        The minimum absolute value of the determinant needed to satisfy nonsingularity.

    Returns
    -------
    is_singular : bool

    """

    # Non-square matrix is singular
    if (len(matrix.shape) != 2) or (matrix.shape[0] != matrix.shape[1]):
        return True
    # Singular matrix has determinant equal to 0; use logarithmic determinant routine to
    # postpone over/underflow
    sgn, det = np.linalg.slogdet(matrix)
    if np.exp(det) < tol:
        return True
    # It passes
    return False


#
# Finite-difference approximation derivative/gradient/Jacobian checker
# Stolen from romin (https://github.com/QuantumElephant/romin)
#

def deriv_error(f, g, x, eps_x=1e-4, order=8):
    """
    Compute the error between differences of f() at two points and its finite-difference
    approximation.

    See romin documentation.

    """

    # Get the right quadrature points and weights
    if order not in gauss_legendre:
        raise ValueError("The order must be one of {}".format(gauss_legendre.keys()))
    points, weights = gauss_legendre.get(order)
    # Compute the difference between f at two different points
    delta = f(x + eps_x) - f(x - eps_x)
    # Approximate that difference with Gaussian quadrature, with some sanity checks
    derivs = np.array([g(x + eps_x * p) for p in points])
    assert delta.shape == derivs.shape[1:delta.ndim + 1]
    if len(derivs.shape) > 1:
        assert derivs.shape[1:] == delta.shape
    delta_approx = np.tensordot(weights, derivs, axes=1) * eps_x
    # Done
    return delta, delta_approx


def deriv_error_array(f, g, x, eps_x=1e-4, order=8, nrep=None):
    """
    Extension of deriv_error() for functions that take arrays as arguments.

    See romin documentation.

    """

    if nrep is None:
        nrep = x.size ** 2
    # Run different random line scans
    results = []
    for irep in xrange(nrep):
        # Generate a generalized random unit vector.
        while True:
            unit = np.random.normal(0, 1, x.shape)
            norm = np.sqrt((unit ** 2).sum())
            if norm > 1e-3:
                unit /= norm
                break

        # Define f and g along the one-dimensional scan
        def f_scan(x_scan):
            return f(x_scan * unit + x)

        def g_scan(x_scan):
            # Nasty chain rule
            return np.tensordot(g(x_scan * unit + x), unit, axes=unit.ndim)

        # Collect results
        results.append(deriv_error(f_scan, g_scan, 0.0, eps_x, order))
    return results


def deriv_check(f, g, xs, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3, discard=0.1, verbose=False):
    """
    Check the implementation of partial derivatives.


    See romin documentation.

    Raises
    ------
    AssertionError
        If any of the selected (delta, delta_approx) have a relative error larger than the
        given `rel_ftol`.

    """

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
    # Make arrays
    deltas = np.array([item[0] for item in results]).ravel()
    deltas_approx = np.array([item[1] for item in results]).ravel()
    # Sort
    order = deltas.argsort()
    deltas = deltas[order]
    deltas_approx = deltas_approx[order]
    # Chop part off
    ndiscard = int(len(deltas) * discard)
    deltas = deltas[ndiscard:]
    deltas_approx = deltas_approx[ndiscard:]
    # Some info on screen
    if verbose:
        print("Deltas: {}".format(deltas))
        print("Approx. deltas: {}".format(deltas_approx))
        print("Number of comparisons: {:d}".format(len(deltas)))
        ratios = abs(deltas - deltas_approx) / abs(deltas)
        print("Intrmdt: {}".format(deltas - deltas_approx))
        print("Ratios: {}".format(ratios))
        print("Best:  {:10.3e}".format(ratios.min()))
        print("Worst: {:10.3e}".format(ratios.max()))
    # Final test
    assert np.all(abs(deltas - deltas_approx) <= rel_ftol * abs(deltas))


# Gauss-Legendre quadrature grids (points and weights) for different orders
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
