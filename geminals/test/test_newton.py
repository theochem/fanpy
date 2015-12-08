#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.testing as npt
from test.common import *
from newton import newton
from scipy.optimize import rosen, rosen_der

@test
def test_wellbehaved_system():
    """
    """
    guess  = np.array([0.678, 1.432])
    answer = np.array([1.0, 1.0])
    lin = lambda x : x[0] - x[1]
    lin_der = lambda x : [1, -1]
    def fun(x0):
        return np.array([rosen(x0), lin(x0)])
    def jac(x0):
        return np.array([rosen_der(x0), lin_der(x0)])
    r_newton = newton(fun, guess, jac=jac)
    npt.assert_allclose(r_newton['x'], answer, atol=0.001, rtol=0)

@test
def test_additional_arguments():
    """
    """
    guess  = np.array([0.686, 1.371])
    answer = np.array([1.0, 1.0])
    lin = lambda x : x[0] - x[1]
    lin_der = lambda x : [1, -1]
    def fun(x0, test0, test1, test2):
        assert test0 and test1 and test2
        return np.array([rosen(x0), lin(x0)])
    def jac(x0, test0, test1, test2):
        assert test0 and test1 and test2
        return np.array([rosen_der(x0), lin_der(x0)])
    r_newton = newton(fun, guess, jac=jac, args=(True, 1, {'a':'b'}))
    npt.assert_allclose(r_newton['x'], answer, atol=0.001, rtol=0)

run_tests()

# vim: set textwidth=90 :
