""" Tests olsens.geminals.geminal

"""

#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from geminal import *
from HortonWrapper import *
from random import shuffle
from slater_det import excite, is_occupied

def check_if_exception_raised(func, exception):
    """ Passes if given exception is raised

    Parameter
    ---------
    func : function object
        Function that contains the desired test code
    exception : Exception
        Exception that is raised

    Returns
    -------
    bool
        True if Exception is raised
        False if Exception is not raised
    """
    try:
        func()
    except exception:
        return True
    except:
        return False
    else:
        return False

def test_init():
    """ Check if initialization with bad values raises error
    """
    # check first sets of init
    def f():
        gem = Geminal(1., 4, [0b0011])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = Geminal(1, 4., [0b0011])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = Geminal(1, 4, 0b0011)
    assert check_if_exception_raised(f, AssertionError)
    # check setters
    def f():
        gem = Geminal(5, 4, [0b0011])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = Geminal(1, 2, [0b0111])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = Geminal(1, 2, [0b0101])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = Geminal(1, 2, [0b110000])
    assert check_if_exception_raised(f, AssertionError)

def test_setters_getters():
    """ Check if setters and getters are working properly
    """
    gem = Geminal(1, 4, [0b00011])
    # check npairs
    assert gem.npairs == 1
    assert gem.npairs == gem._npairs
    # check npairs setter
    gem.npairs = 2
    assert gem.npairs == 2
    def f():
        gem.npairs = 5
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.npairs = 0.5
    assert check_if_exception_raised(f, AssertionError)
    # check nelec
    gem.npairs = 1
    assert gem.nelec == 2
    gem.npairs = 2
    assert gem.nelec == 4
    # check norbs
    assert gem.norbs == 4
    assert gem.norbs == gem._norbs
    # check norbs setter
    gem.norbs = 7
    assert gem.norbs == 7
    gem.npairs = 5
    def f():
        gem.norbs = 4
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.norbs = 9.0
    assert check_if_exception_raised(f, AssertionError)
    # check pspace
    assert gem.pspace == tuple([0b00011])
    assert gem.pspace == tuple([3])
    assert gem.pspace == gem._pspace
    # check pspace setter
    gem.npairs = 2
    gem.norbs = 4
    gem.pspace = tuple([0b001111])
    assert gem.pspace == tuple([0b001111])
    def f():
        gem.pspace = tuple([0b01])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.pspace = tuple([0b10111])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.pspace = tuple([0b1100000011])
    assert check_if_exception_raised(f, AssertionError)

def test_permanent():
    """ test permanent
    """
    # zero matrix
    matrix = np.zeros((6,6))
    assert Geminal.permanent(matrix) == 0
    # identity matrix
    matrix = np.identity(6)
    assert Geminal.permanent(matrix) == 1
    # one matrix
    matrix = np.zeros((6,6)) + 1
    assert Geminal.permanent(matrix) == np.math.factorial(6)
    # random matrix
    matrix = np.arange(1, 10).reshape((3,3))
    assert Geminal.permanent(matrix) == 450


'''
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
'''
