#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from Geminal import *
from HortonWrapper import *


matrix = np.array([ [1,2,3],
                    [4,5,6],
                    [7,8,9] ])
assert Geminal.permanent(matrix) == 450


fn = 'test/li2.xyz'
basis = 'sto-3g'
nocc = 3
inpt = from_horton(fn=fn, basis=basis, nocc=nocc, guess=None)
maxiter = 5
solver=quasinewton
options = { 'options': { 'maxiter':maxiter,
                         'disp': True,
                         'xatol': 1.0e-6,
                       },
            'method': 'krylov',
          }

gem = Geminal(nocc, inpt['basis'].nbasis)
guess = np.zeros(gem.norbs*gem.npairs + 1)
guess[0] += inpt['energy']
guess[1:] += inpt['coeffs'].ravel()

dets = []
ground = min(gem.pspace)
for i in range(2*gem.npairs):
    for j in range(2*gem.npairs, 2*gem.norbs):
        dets.append(gem.excite_single(ground, i, j))
for i in range(0, 2*gem.npairs, 2):
    for j in range(2*gem.npairs, 2*gem.norbs, 2):
        dets.append(gem.excite_double(ground, i, i+1, j, j+1))
#dets.extend(gem.pspace)
dets = list(set(dets))
if 0 in dets:
    dets.remove(0)

coeff = np.zeros((gem.npairs, gem.norbs))
coeff[:,:gem.npairs] += np.eye(gem.npairs)
coeff[:,gem.npairs:] += 0.1*np.random.rand(gem.npairs, (gem.norbs - gem.npairs)) - 0.05

guess = np.zeros(1 + gem.norbs*gem.npairs)
guess[0] = -14.5
guess[1:] += coeff.ravel()

result = gem(guess, *inpt['ham'], dets=dets, solver=solver, options=options)


# vim: set textwidth=90 :
