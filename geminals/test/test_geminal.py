""" Tests olsens.geminals.geminal

"""

#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from geminal import *
from horton_wrapper import *
from random import shuffle
from slater_det import excite_pairs, excite_orbs, is_occupied

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


# Define user input
fn = 'test/li2.xyz'
basis = 'sto-3g'
nocc = 3
maxiter = 20
solver=quasinewton
#solver=lstsq

options = { 'options': { 'maxiter':maxiter,
                         'disp': True,
                         #'xatol': 1.0e-10,
                         'fatol': 1.0e-9,
                         #'line_search': 'wolfe',
                       },
          }

if solver is quasinewton:
    options['method'] = 'krylov'

# Make geminal, and guess, from HORTON's AP1roG module
inpt = from_horton(fn=fn, basis=basis, nocc=nocc, guess=None)
basis  = inpt['basis']
coeffs = inpt['coeffs']
energy = inpt['energy']
core = inpt['ham'][2]
ham = gem.reduce_hamiltonian(*inpt['ham'][0:2])
#guess = coeffs.ravel()
guess = np.eye(nocc, M=basis.nbasis)
guess[:,nocc:] = 0.02*(np.random.rand(nocc, basis.nbasis - nocc) - 1.0)
guess = guess.ravel()
gem = Geminal(nocc, basis.nbasis)

# Run the optimization
print("**********energy**********")
print(gem.phi_H_psi(min(gem.pspace), coeffs, ham) + inpt['ham'][2])
print("Guess:\n{}".format(guess))

result = gem(guess.ravel(), *inpt['ham'][:2], solver=solver, options=options)

print("GUESS")
print(inpt['coeffs'])
print(inpt['energy'])# + inpt['ham'][2])
print("GEMINAL")
print(gem.coeffs)
print(gem.phi_H_psi(gem.ground, gem.coeffs, ham) + inpt['ham'][2])
print("OLP:\t{}".format(gem.overlap(gem.ground, gem.coeffs)))


# vim: set textwidth=90 :
