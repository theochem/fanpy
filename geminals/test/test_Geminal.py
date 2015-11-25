#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from Geminal import *
from HortonWrapper import *
from random import shuffle

# Test permanent evaluation
matrix = np.array([ [1,2,3],
                    [4,5,6],
                    [7,8,9] ])
assert Geminal.permanent(matrix) == 450

# Test excitations
assert Geminal.excite(0b001, 0, 2) == 0b100
assert Geminal.excite(0b000011, 0, 1, 4, 5) == 0b110000

# Test annihilation --> vacuum on virtual orbitals
assert Geminal.excite(0b010, 0, 2) == 0
assert Geminal.excite(0b000101, 0, 1, 4, 5) == 0

# Test creation --> vacuum on occupied orbitals
assert Geminal.excite(0b101, 0, 2) == 0
assert Geminal.excite(0b110011, 0, 1, 4, 5) == 0

#
# Test APIG optimization
#

# Define user input
fn = 'test/li2.xyz'
basis = 'sto-3g'
nocc = 3
maxiter = 10
solver=lstsq
options = { 'options': { 'maxiter':maxiter,
                         'disp': True,
                         'xatol': 1.0e-6,
                       },
          }

if solver is quasinewton: options['method'] = 'krylov'

# Make geminal, and guess, from HORTON's AP1roG module
inpt = from_horton(fn=fn, basis=basis, nocc=nocc, guess=None)
gem = Geminal(nocc, inpt['basis'].nbasis)
guess = np.zeros(gem.norbs*gem.npairs + 1)
guess[0] = inpt['energy']
guess[1:] = inpt['coeffs'].ravel()
#guess[0] = -5.0
#guess[1:] = np.random.rand(gem.npairs*gem.norbs)

# Projected Slater determinants are all single and double excitations
dets = []
ground = min(gem.pspace)
for i in range(0, 2*gem.npairs, 2):
    for j in range(2*gem.npairs, 2*gem.norbs, 2):
        dets.append(gem.excite(ground, i, i+1, j, j+1))
for i in range(2*gem.npairs):
    for j in range(2*gem.npairs, 2*gem.norbs):
        dets.append(gem.excite(ground, i, j))
dets = list(set(dets))
if 0 in dets:
    dets.remove(0)
shuffle(dets)

# Run the optimization
result = gem(guess, *inpt['ham'], dets=dets, solver=solver, options=options)
print("GUESS")
print(inpt['coeffs'])
print(inpt['energy'])
print("GEMINAL")
print(gem.coeffs)
print(result['x'][0])


# vim: set textwidth=90 :
