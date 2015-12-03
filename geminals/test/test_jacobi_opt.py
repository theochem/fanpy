#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import numpy.testing as npt
from test.common import slow, run_tests
from jacobi_opt import *
from scipy.optimize import rosen

@slow
def test_simple_nonorthogonal_jacobi():
    tol = 1.0e-4
    rosen_size = 4
    answer = [1.0 for i in range(rosen_size)]
    guess = [np.sqrt(np.abs(i - 0.5)) for i in range(rosen_size)]
    r_rosen = JacobiOptimizer()(guess, rosen)
    npt.assert_allclose(guess, answer, atol=tol, rtol=0)

tests = [ test_simple_nonorthogonsl_jacobi,
        ]

run_tests(tests)

# vim: set textwidth=90 :
