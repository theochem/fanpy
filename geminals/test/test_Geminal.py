#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from Geminal import *
from HortonWrapper import *


matrix = np.array([ [1,2,3],
                    [4,5,6],
                    [7,8,9] ])
assert Geminal.permanent(matrix) == 450


file = 'test/li2.xyz'
basis = 'sto-3g'
nocc = 3
input = from_horton(file=file, basis=basis, nocc=nocc, guess=None)
maxiter = 10
solver=quasinewton
options = { 'options': { 'maxiter':maxiter,
                         'disp': True,
                         'xatol': 1.0e-6,
                       },
            'method': 'krylov',
          }

gem = Geminal(nocc, input['basis'].nbasis)
guess = np.zeros(gem.norbs*gem.npairs + 1)
guess[0] += input['energy']
guess[1:] += input['coeffs'].ravel()
result = gem(guess, *input['ham'], solver=solver, options=options)


# vim: set textwidth=90 :
