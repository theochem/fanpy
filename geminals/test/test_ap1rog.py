"""
Unit tests for geminals.apig.AP1roG.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from romin import deriv_check
from geminals.apig import APIG
from geminals.ap1rog import AP1roG
from geminals.horton_wrapper import ap1rog_from_horton
from geminals.slater_det import excite_orbs, add_pairs, excite_pairs
from geminals.test.common import *


@test
def test_init():
    """
    Check if AP1roG.__init__() works and that initialization errors are caught.

    """

    npairs = 1
    norbs = 4
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))

    # params of wrong dimension
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    params = np.ones((npairs + 1)*norbs)
    f = lambda: AP1roG(npairs, norbs, ham, init_params=params)
    assert raises_exception(f, AssertionError)

    # Non-np.ndarray params
    params = 1

    def f():
        gem = AP1roG(npairs, norbs, ham, init_params=params)

    assert raises_exception(f, AssertionError)


@test
def test_properties():
    """
    Test AP1roG's properties' getters and setters.

    """

    npairs = 2
    norbs = 5
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)), 1.0)
    params = np.ones(npairs*(norbs-npairs))
    gem = AP1roG(npairs, norbs, ham, init_params=params)
    assert np.allclose(gem._params, params)
    assert gem._exclude_ground
    assert not gem._normalize


@test
def test_generate_init_and_params():
    """
    Test AP1roG._generate_init() and AP1roG._construct_params().

    """

    npairs = 3
    norbs = 6
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = AP1roG(npairs, norbs, ham)
    x0 = gem._generate_init()
    assert x0.shape == (x0.size,) == (npairs * (norbs - npairs),)


@test
def test_generate_pspace():
    """
    Test AP1roG.generate_pspace().

    """

    # Hand-check that each of the generated Slater determinant is actually in the
    # projection space.
    npairs = 3
    norbs = 6
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = AP1roG(npairs, norbs, ham)
    assert sorted(gem.pspace) == sorted((0b000000111111, 0b000011001111,
                                         0b001100001111, 0b110000001111,
                                         0b000011110011, 0b001100110011,
                                         0b110000110011, 0b000011111100,
                                         0b001100111100, 0b110000111100,
                                         ))


@test
def test_overlap():
    """
    Test AP1roG.overlap(), and the partial derivatives of the overlaps.

    """

    npairs = 3
    norbs = 6
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = AP1roG(npairs, norbs, ham)
    params = np.eye(npairs, M=norbs)
    params[:, npairs:] += np.random.rand(npairs, (norbs - npairs))

    # Bad Slater determinant
    phi = None
    assert np.allclose(gem.overlap(phi, params), 0.0)

    # Slater determinant with different number of electrons
    phi = int("1" * (2 * npairs + 1), 2)
    assert np.allclose(gem.overlap(phi, params), 0.0)


@slow
def test_solve():
    """
    Test AP1roG.solve_params() and AP1roG.nonlin() by using them to optimize some AP1roG
    coefficients.

    """

    fn = "test/li2.xyz"
    basis = "3-21g"
    npairs = 3
    horton_result = ap1rog_from_horton(fn=fn, basis=basis, npairs=npairs, guess="ap1rog")
    horton_basis = horton_result["basis"]
    horton_energy = horton_result["energy"]
    horton_ham = horton_result["ham"]
    horton_params = horton_result["coeffs"]
    gem = AP1roG(npairs, horton_basis.nbasis, horton_ham)
    ap1rog_result = gem.solve_params()
    assert ap1rog_result["success"]
    assert np.allclose(gem.params[:, npairs:], horton_params, atol=1.0e-3)
    assert np.allclose(horton_energy, gem.energy)


@test
def test_nonlin_jac():
    """
    Verify via the finite-difference approximation that the output of AP1roG.nonlin_jac()
    represents the Jacobian of the output of AP1roG.nonlin().

    """

    fn = "test/h2.xyz"
    basis = "3-21g"
    npairs = 3
    horton_result = ap1rog_from_horton(fn=fn, basis=basis, npairs=npairs, guess="ap1rog")
    horton_basis = horton_result["basis"]
    horton_ham = horton_result["ham"]
    x0 = horton_result["coeffs"].ravel()
    gem = AP1roG(npairs, horton_basis.nbasis, horton_ham)
    fun = lambda x: gem.nonlin(x, gem.pspace)
    jac = lambda x: gem.nonlin_jac(x, gem.pspace)
    deriv_check(fun, jac, x0)


run_tests()

# vim: set textwidth=90 :
