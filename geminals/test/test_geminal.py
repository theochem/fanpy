""" Tests olsens.geminals.geminal

"""

#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from copy import deepcopy as copy
from itertools import combinations
from scipy.optimize import root as quasinewton
from geminal import APIG
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
        gem = APIG(1., 4, [0b0011])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = APIG(1, 4., [0b0011])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = APIG(1, 4, 0b0011)
    assert check_if_exception_raised(f, AssertionError)
    # check setters
    def f():
        gem = APIG(5, 4, [0b0011])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = APIG(1, 2, [0b0111])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = APIG(1, 2, [0b0101])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem = APIG(1, 2, [0b110000])
    assert check_if_exception_raised(f, AssertionError)

def test_setters_getters():
    """ Check if setters and getters are working properly
    """
    gem = APIG(1, 4, [0b00011])
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
        gem.npairs = 0
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
    gem.pspace = tuple([0b0001111])
    assert gem.pspace == tuple([0b001111])
    gem.pspace = tuple([0b001111])
    assert gem.pspace == tuple([0b001111])
    def f():
        gem.pspace = tuple([0b0])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.pspace = tuple([0b01])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.pspace = tuple([0b10111])
    assert check_if_exception_raised(f, AssertionError)
    def f():
        gem.pspace = tuple([0b1100000011])
    assert check_if_exception_raised(f, AssertionError)

def test_generate_pspace():
    """ test Geminal.generate_pspace
    """
    def f():
        gem.generate_pspace()
    # 99 occupied, No virtuals (Not enough SD generated)
    gem = Geminal(99, 99, [int('11'*99, 2)])
    assert check_if_exception_raised(f, AssertionError)
    # 1 occupied, 1 virtual (Not enough SD generated)
    gem = Geminal(1, 2, [0b11])
    assert check_if_exception_raised(f, AssertionError)
    # 1 occupied, 99 virtuals (Not enough SD generated)
    gem = Geminal(1, 99, [0b11])
    assert check_if_exception_raised(f, AssertionError)
    # 2 occupied, 2 virtuals (Not enough SD generated)
    gem = Geminal(2, 4, [0b1111])
    assert check_if_exception_raised(f, AssertionError)
    # 2 occupied, 3 virtuals (Not enough SD generated)
    gem = Geminal(2, 5, [0b1111])
    assert check_if_exception_raised(f, AssertionError)
    # 2 occupied, 4 virtuals
    gem = Geminal(2, 6, [0b1111])
    pspace = gem.generate_pspace()
    assert len(pspace) == 13
    #  check that each of the generated slated determinant is in the total space
    all_sds = (sum(0b11 << i*2 for i in pairs)
               for pairs in combinations(range(gem.norbs), gem.npairs))
    for i in pspace:
        assert i in all_sds
    #  hand checking
    assert pspace == (0b1111,
                      0b110011, 0b11000011, 0b1100000011, 0b110000000011,
                      0b111100, 0b11001100, 0b1100001100, 0b110000001100,
                      0b11110000, 0b1100110000, 0b110000110000,
                      0b1111000000)
    #  check __init__
    gem2 = Geminal(2, 6)
    assert gem.generate_pspace() == gem2.generate_pspace()
    # 3 occupied, 3 virtuals
    gem = Geminal(3, 6, [0b111111])
    pspace = gem.generate_pspace()
    assert len(pspace) == 19
    #  check that each of the generated slated determinant is in the total space
    all_sds = [sum(0b11 << i*2 for i in pairs)
               for pairs in combinations(range(gem.norbs), gem.npairs)]
    for i in pspace:
        assert i in all_sds

def test_permanent():
    """ test permanent
    """
    # zero matrix
    matrix = np.zeros((6,6))
    assert APIG.permanent(matrix) == 0
    # identity matrix
    matrix = np.identity(6)
    assert APIG.permanent(matrix) == 1
    # one matrix
    matrix = np.zeros((6,6)) + 1
    assert APIG.permanent(matrix) == np.math.factorial(6)
    # random matrix
    matrix = np.arange(1, 10).reshape((3,3))
    assert APIG.permanent(matrix) == 450

def test_overlap():
    """ Tests Geminal.overlap
    """
    gem = Geminal(3, 6)
    coeff = np.random.rand(3, 6)
    # Bad Slater determinant
    sd = None
    assert gem.overlap(sd, coeff) == 0
    # Slater determinant with different number of electrons
    sd = 0b11
    assert gem.overlap(sd, coeff) == 0
    # Ground state SD
    sd = 0b111111
    assert gem.overlap(sd, coeff) == gem.permanent(coeff[:, :3])
    # Excited SD
    sd = 0b110011001100
    assert gem.overlap(sd, coeff) == gem.permanent(coeff[:, [1, 3, 5]])

# Define user input
fn = 'test/h4.xyz'
basis = '3-21g'
nocc = 2
maxiter = 100
solver=quasinewton
#solver=lstsq

options = { 'options': { 'maxiter':maxiter,
                         'disp': True,
                         'xatol': 1.0e-12,
                         'fatol': 1.0e-12,
                         #'line_search': 'wolfe',
                         #'eps': 1.0e-12,
                         #'factor': 0.1,
                       },
          }

if solver is quasinewton:
    options['method'] = 'krylov'

# Make geminal, and guess, from HORTON's AP1roG module
inpt = from_horton(fn=fn, basis=basis, nocc=nocc, guess=None)
#inpt = from_horton(fn=fn, basis=basis, nocc=nocc, guess='ap1rog')
basis  = inpt['basis']
coeffs = inpt['coeffs']
energy = inpt['energy']
core = inpt['ham'][2]
guess = coeffs.ravel() #- 0.01*np.random.rand(nocc*basis.nbasis)
#guess = 0.050*(2.0*(np.random.rand(nocc*(basis.nbasis - nocc)) - 1.0))
#guess = np.eye(nocc, M=basis.nbasis)
#guess[:,nocc:] = 0.02*(np.random.rand(nocc, basis.nbasis - nocc) - 1.0)
#guess = guess.ravel()
gem = APIG(nocc, basis.nbasis)
#gem = AP1roG(nocc, basis.nbasis)
ham = gem.reduce_hamiltonian(*inpt['ham'][0:2])
backup = copy(guess)

# Run the optimization
print("**********energy**********")
#print(gem.phi_H_psi(min(gem.pspace), coeffs, ham) + inpt['ham'][2])
#print("Guess:\n{}".format(guess))

result = gem(guess, *inpt['ham'][:2], solver=solver, options=options)

print("GUESS")
print(inpt['coeffs'])
print(inpt['energy'])# + inpt['ham'][2])
print("GEMINAL")
print(gem.coeffs)
print(gem.phi_H_psi(gem.ground, gem.coeffs, ham) + inpt['ham'][2])
print("OLP:\t{}".format(gem.overlap(gem.ground, gem.coeffs)))


# vim: set textwidth=90 :
