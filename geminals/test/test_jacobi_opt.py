#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from jacobi_opt import *
import numpy.testing as npt
from scipy.optimize import rosen


tol = 1.0e-4
rosen_size = 4

answer = [1.0 for i in range(rosen_size)]
guess = [np.sqrt(np.abs(i - 0.5)) for i in range(rosen_size)]
r_rosen = JacobiOptimizer()(guess, rosen)
npt.assert_allclose(guess, answer, atol=tol, rtol=0)


# vim: set textwidth=90 :
