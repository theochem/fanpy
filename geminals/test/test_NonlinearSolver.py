#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from NonlinearSolver import *
import numpy.testing as npt
from scipy.optimize import rosen, rosen_der


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

r_quasi = quasinewton(fun, guess, method='broyden2')
npt.assert_allclose(r_quasi['x'], answer, atol=tol, rtol=0)


# vim: set textwidth=90 :
