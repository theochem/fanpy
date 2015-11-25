#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from Geminal import *
from HortonWrapper import *
from random import shuffle
from slater_det import excite, is_occupied

# Test permanent evaluation
matrix = np.array([ [1,2,3],
                    [4,5,6],
                    [7,8,9] ])
assert Geminal.permanent(matrix) == 450

# Test excitations
assert excite(0b001, 0, 2) == 0b100
assert excite(0b000011, 0, 1, 4, 5) == 0b110000

# Test annihilation --> vacuum on virtual orbitals
assert excite(0b010, 0, 2) == 0
assert excite(0b000101, 0, 1, 4, 5) == 0

# Test creation --> vacuum on occupied orbitals
assert excite(0b101, 0, 2) == 0
assert excite(0b110011, 0, 1, 4, 5) == 0

# Test occupancy
assert is_occupied(0b100100, 2)
assert is_occupied(0b100100, 5)
assert not is_occupied(0b100100, 4)
assert not is_occupied(0b100100, 6)
assert not is_occupied(0b100100, 0)

#
# Test APIG optimization
#

# Define user input
fn = 'test/li2.xyz'
basis = 'cc-pvdz'
nocc = 3
maxiter = 10
solver=quasinewton
options = { 'options': { 'maxiter':maxiter,
                         'disp': True,
                         'xatol': 1.0e-6,
                       },
          }

if solver is quasinewton: options['method'] = 'krylov'

# Make geminal, and guess, from HORTON's AP1roG module
inpt = from_horton(fn=fn, basis=basis, nocc=nocc, guess=None)
basis  = inpt['basis']
coeffs = inpt['coeffs']
energy = inpt['energy']
ham    = inpt['ham']
guess = np.zeros(nocc*basis.nbasis + 1)
guess[0] = energy - 0.1
guess[1:] = coeffs.ravel()
gem = Geminal(nocc, basis.nbasis)

# Projected Slater determinants are all single and double excitations
dets = []
ground = min(gem.pspace)
for i in range(2*gem.npairs):
    for j in range(2*gem.npairs, 2*gem.norbs):
        dets.append(excite(ground, i, j))
for i in range(0, 2*gem.npairs, 2):
    for j in range(2*gem.npairs, 2*gem.norbs, 2):
        dets.append(excite(ground, i, i+1, j, j+1))
dets = list(set(dets))
if 0 in dets:
    dets.remove(0)
shuffle(dets)

# Run the optimization
#print("**********energy**********")
#print(gem.phi_H_psi(ground, coeffs, *ham))
result = gem(guess, *inpt['ham'], dets=dets, solver=solver, options=options)
print("GUESS")
print(inpt['coeffs'])
print(inpt['energy'])
print("GEMINAL")
print(gem.coeffs)
print(result['x'][0])


# vim: set textwidth=90 :
