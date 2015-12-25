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
    while reversed(_common_tests):
        fun = _common_tests.pop()
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


def make_hermitian(matrix):
    """
    Make a matrix or 4-index tensor Hermitian.

    Parameters
    ----------
    matrix: 2-index np.ndarray or 4-index np.ndarray
        The matrix to be made Hermitian.

    Returns
    -------
    hermitian: 2-index np.ndarray or 4-index np.ndarray
        The Hermitian matrix.

    Raises
    ------
    AssertionError
        If the matrix is not square (or hypercubic).
    TypeError
        If the matrix is not a 2- or 4- index tensor.

    """

    assert np.allclose(*matrix.shape) and len(matrix.shape) % 2 == 0, \
        "A non-square matrix cannot be made Hermitian."

    hermitian = np.zeros(matrix.shape)
    if len(matrix.shape) == 2:
        hermitian += matrix + matrix.transpose()
    elif len(matrix.shape) == 4:
        hermitian += matrix + np.einsum("jilk", matrix)
        hermitian += np.einsum("klij", hermitian)
    else:
        raise TypeError("Only two- and four- index tensors can be made Hermitian.")
    return hermitian


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

# vim: set textwidth=90 :
