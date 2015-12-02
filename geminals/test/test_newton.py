#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from newton import newton
import numpy as np
import numpy.testing as npt
from scipy.optimize import rosen, rosen_der

# Solve a well-behaved nonlinear system
tol = 1.0e-4
guess  = np.array([0.686, 1.371])
answer = np.array([1.0, 1.0])
lin = lambda x : x[0] - x[1]
lin_der = lambda x : [1, -1]
def fun(x0):
    return np.array([rosen(x0), lin(x0)])
def jac(x0):
    return np.array([rosen_der(x0), lin_der(x0)])
r_newton = newton(fun, guess, jac=jac)
npt.assert_allclose(r_newton['x'], answer, atol=tol, rtol=0)
# Test that additional arguments can be passed into the objective functions/jacobians
def fun(x0, test0, test1, test2):
    assert test0 and test1 and test2
    return np.array([rosen(x0), lin(x0)])
def jac(x0, test0, test1, test2):
    assert test0 and test1 and test2
    return np.array([rosen_der(x0), lin_der(x0)])
r_newton = newton(fun, guess, jac=jac, args=(True, 1, {'a':'b'}))
npt.assert_allclose(r_newton['x'], answer, atol=tol, rtol=0)


# vim: set textwidth=90 :
