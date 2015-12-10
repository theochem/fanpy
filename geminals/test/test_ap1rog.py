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

    # coeffs of wrong dimension
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    coeffs = np.ones((npairs + 1, norbs))

    def f():
        gem = AP1roG(npairs, norbs, ham, coeffs=coeffs)

    assert raises_exception(f, AssertionError)

    # Non-np.ndarray coeffs
    coeffs = 1

    def f():
        gem = AP1roG(npairs, norbs, ham, coeffs=coeffs)

    assert raises_exception(f, AttributeError)


@test
def test_properties():
    """
    Test AP1roG's properties' getters and setters.

    """

    npairs = 2
    norbs = 5
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)), 1.0)
    coeffs = np.ones((npairs, norbs))
    gem = AP1roG(npairs, norbs, ham, coeffs=coeffs)
    assert np.allclose(gem._coeffs, coeffs)
    assert gem._coeffs_optimized
    assert gem._exclude_ground
    assert not gem._normalize
    assert list(gem._row_indices) == list(range(0, npairs))
    assert list(gem._col_indices) == list(range(npairs, norbs))


@test
def test_generate_x0_and_coeffs():
    """
    Test AP1roG._generate_x0() and AP1roG._construct_coeffs().

    """

    npairs = 3
    norbs = 6
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = AP1roG(npairs, norbs, ham)
    x0 = gem._generate_x0()
    assert x0.shape == (x0.size,) == (npairs * (norbs - npairs),)
    coeffs = gem._construct_coeffs(x0)
    assert coeffs.shape == (npairs, norbs)
    assert np.allclose(x0, coeffs[:, gem.npairs:].ravel())


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
    coeffs = np.eye(npairs, M=norbs)
    coeffs[:, npairs:] += np.random.rand(npairs, (norbs - npairs))

    # Bad Slater determinant
    phi = None
    assert np.allclose(gem.overlap(phi, coeffs), 0.0)

    # Slater determinant with different number of electrons
    phi = int("1" * (2 * npairs + 1), 2)
    assert np.allclose(gem.overlap(phi, coeffs), 0.0)

    # Ground-state Slater determinant
    phi = gem.ground
    ap1rog = gem.overlap(phi, coeffs)
    apig = gem.permanent(coeffs[:, :npairs])
    answer = 1.0
    assert np.allclose(ap1rog, apig, answer)
    # Partial derivative of overlap
    gem._overlap_derivative = True
    gem._overlap_indices = (1, 1)
    ap1rog = gem.overlap(phi, coeffs)
    apig = gem.permanent_derivative(coeffs[:, :npairs], 1, 1)
    answer = 1.0
    assert np.allclose(ap1rog, apig, answer)
    gem._overlap_derivative = False
    gem._overlap_indices = None

    # Singly- pair-excited Slater determinant
    phi = 0b11001111
    cols = [0, 1, 3]
    ap1rog = gem.overlap(phi, coeffs)
    apig = gem.permanent(coeffs[:, cols])
    answer = coeffs[2, 3]
    assert np.allclose(ap1rog, apig, answer)
    # Partial derivative of overlap
    gem._overlap_derivative = True
    gem._overlap_indices = (1, 1)
    ap1rog = gem.overlap(phi, coeffs)
    apig = gem.permanent_derivative(coeffs[:, cols], 1, 1)
    answer = 1.0
    assert np.allclose(ap1rog, apig, answer)
    gem._overlap_indices = (0, 1)
    ap1rog = gem.overlap(phi, coeffs)
    apig = gem.permanent_derivative(coeffs[:, cols], 0, 1)
    answer = 0.0
    assert np.allclose(ap1rog, apig, answer)


@slow
def test_solve():
    """
    Test AP1roG.solve_coeffs() and AP1roG.nonlin() by using them to optimize some AP1roG
    coefficients.

    """

    fn = "test/li2.xyz"
    basis = "3-21g"
    npairs = 3
    horton_result = ap1rog_from_horton(fn=fn, basis=basis, npairs=npairs, guess="ap1rog")
    horton_basis = horton_result["basis"]
    horton_energy = horton_result["energy"]
    horton_ham = horton_result["ham"]
    horton_coeffs = horton_result["coeffs"]
    gem = AP1roG(npairs, horton_basis.nbasis, horton_ham)
    ap1rog_result = gem.solve_coeffs()
    assert ap1rog_result["success"]
    print(horton_result["energy"], gem.energy)
    print(gem.coeffs[:, npairs:], horton_coeffs)
    assert np.allclose(gem.coeffs[:, npairs:], horton_coeffs, atol=1.0e-3)
    assert np.allclose(horton_energy, gem.energy)


@test
def test_nonlin_jac():
    """
    Verify via the finite-difference approximation that the output of AP1roG.nonlin_jac()
    represents the Jacobian of the output of AP1roG.nonlin().

    """

    fn = "test/h2.xyz"
    basis = "3-21g"
    npairs = 2
    horton_result = ap1rog_from_horton(fn=fn, basis=basis, npairs=npairs, guess="ap1rog")
    horton_basis = horton_result["basis"]
    horton_ham = horton_result["ham"]
    x0 = horton_result["coeffs"].ravel()
    gem = AP1roG(npairs, horton_basis.nbasis, horton_ham)
    fun = lambda x: gem.nonlin(x, gem.pspace)
    jac = lambda x: gem.nonlin_jac(x, gem.pspace)
    # Discard most values because romin thinks 0 != 0 (this is a bug in romin!)
    deriv_check(fun, jac, x0, discard=0.8)


run_tests()

# vim: set textwidth=90 :
