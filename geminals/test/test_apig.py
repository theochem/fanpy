"""
Unit tests for geminals.apig.APIG.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from romin import deriv_check
from geminals.apig import APIG
from geminals.horton_wrapper import ap1rog_from_horton
from geminals.slater_det import excite_orbs, add_pairs, excite_pairs
from geminals.test.common import *


@test
def test_init():
    """
    Check if APIG.__init__() works and that initialization errors are caught.

    """

    npairs = 1
    norbs = 4
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))

    # Non-int npairs
    def f():
        gem = APIG(float(npairs), norbs, ham)

    assert raises_exception(f, AssertionError)

    # Non-int norbs
    def f():
        gem = APIG(npairs, float(norbs), ham)

    assert raises_exception(f, AssertionError)

    # npairs > norbs
    def f():
        gem = APIG(norbs + 1, norbs, ham)

    assert raises_exception(f, AssertionError)

    # No `ham` specified
    def f():
        gem = APIG(npairs, norbs)

    assert raises_exception(f, TypeError)

    # `ham` arrays of wrong dimension
    ham = (np.ones((norbs + 1, norbs)), np.ones((norbs, norbs, norbs, norbs)))

    def f():
        gem = APIG(npairs, norbs, ham)

    assert raises_exception(f, AssertionError)
    ham = (np.ones((norbs, norbs)), np.ones((npairs, npairs, npairs, npairs)))

    def f():
        gem = APIG(npairs, norbs, ham)

    assert raises_exception(f, AssertionError)

    # Non-Hermitian `ham` arrays
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    ham[0][:, npairs:] *= 0.0

    def f():
        gem = APIG(npairs, norbs, ham)

    assert raises_exception(f, AssertionError)
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    ham[1][:, npairs:, :, :] *= 0.0

    def f():
        gem = APIG(npairs, norbs, ham)

    assert raises_exception(f, AssertionError)

    # params of wrong dimension
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    params = np.ones((npairs + 1, norbs))

    def f():
        gem = APIG(npairs, norbs, ham, init_params=params)

    assert raises_exception(f, AssertionError)

    # Non-np.ndarray params
    params = 1

    def f():
        gem = APIG(npairs, norbs, ham, init_params=params)

    assert raises_exception(f, AssertionError)

    # pspace of wrong type
    params = np.ones((npairs, norbs))
    pspace = 1

    def f():
        gem = APIG(npairs, norbs, ham, pspace=pspace)

    assert raises_exception(f, AssertionError)

    # Slater det. with wrong number of electrons in pspace
    pspace = list(APIG(npairs, norbs, ham).pspace)
    badphi = add_pairs(min(pspace), norbs - 1)
    pspace.append(badphi)

    def f():
        gem = APIG(npairs, norbs, ham, pspace=pspace)

    assert raises_exception(f, AssertionError)
    pspace.remove(badphi)

    # Unrestricted Slater det. in pspace
    badphi = excite_orbs(min(pspace), 0, 2 * npairs + 2)
    pspace.append(badphi)

    def f():
        gem = APIG(npairs, norbs, ham, pspace=pspace)

    assert raises_exception(f, AssertionError)
    pspace.remove(badphi)

    # Slater det. with electrons in spatial orbital k > norbs in pspace
    badphi = excite_pairs(min(pspace), 0, norbs + 1)
    pspace.append(badphi)

    def f():
        gem = APIG(npairs, norbs, ham, pspace=pspace)

    assert raises_exception(f, AssertionError)
    pspace.remove(badphi)


@test
def test_properties():
    """
    Test APIG's properties' getters and setters.

    """

    npairs = 2
    norbs = 5
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)), 1.0)
    params = np.ones((npairs, norbs))
    pspace = APIG(npairs, norbs, ham).pspace
    gem = APIG(npairs, norbs, ham, init_params=params, pspace=pspace)
    assert gem._npairs == npairs
    assert gem.nelec == npairs * 2
    assert gem._norbs == norbs
    assert gem._ham == ham
    assert gem._ham[2] == gem.core_energy == ham[2]
    assert np.allclose(gem._params, params.ravel())
    assert sorted(gem._pspace) == sorted(pspace)
    assert gem.ground_sd == min(pspace)


@test
def test_generate_x0_and_params():
    """
    Test APIG._generate_x0() and APIG._construct_params().

    """

    npairs = 3
    norbs = 6
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = APIG(npairs, norbs, ham)
    x0 = gem._generate_init()
    assert x0.shape == (x0.size,) == (npairs * norbs,)


@test
def test_generate_pspace():
    """
    Test APIG.generate_pspace().

    """

    # Unable to generate enough Slater determinants
    npairs = norbs = 10
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))

    def f():
        APIG(npairs, norbs, ham)

    assert raises_exception(f, AssertionError)

    # Able to generate enough Slater determinants (with manual check)
    npairs = 1
    norbs = 2
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = APIG(npairs, norbs, ham)
    assert sorted(gem.pspace) == sorted((0b0011, 0b1100))

    # Again, but more complicated (check length of pspace)
    npairs = 5
    norbs = 13
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = APIG(npairs, norbs, ham)
    assert len(gem.pspace) == npairs * norbs

    # Hand-check that each of the generated Slater determinant is actually in the
    # projection space.
    npairs = 3
    norbs = 6
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = APIG(npairs, norbs, ham)
    assert sorted(gem.pspace) == sorted((0b000000111111, 0b000011001111,
                                         0b000011110011, 0b000011111100,
                                         0b001100001111, 0b001100110011,
                                         0b001100111100, 0b001111000011,
                                         0b001111001100, 0b001111110000,
                                         0b110000001111, 0b110000110011,
                                         0b110000111100, 0b110011000011,
                                         0b110011001100, 0b110011110000,
                                         0b111100000011, 0b111100001100,
                                         ))


@test
def test_permanent():
    """
    Test APIG.permanent() and APIG.permanent_derivative().

    """

    # Zero matrix
    matrix = np.zeros((6, 6))
    assert APIG.permanent(matrix) == 0

    # Identity matrix
    matrix = np.eye(6)
    assert APIG.permanent(matrix) == 1

    # One matrix
    matrix = np.ones((6, 6))
    assert APIG.permanent(matrix) == np.math.factorial(6)

    # Another matrix, hand-checked
    matrix = np.arange(1, 10).reshape((3, 3))
    assert APIG.permanent(matrix) == 450


@test
def test_overlap():
    """
    Test APIG.overlap().

    """

    npairs = 3
    norbs = 6
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = APIG(npairs, norbs, ham)
    params = np.random.rand(npairs*norbs)
    matrix = params.reshape(npairs, norbs)

    # Bad Slater determinant
    phi = None
    assert gem.overlap(phi, params) == 0

    # Slater determinant with different number of electrons
    phi = int("1" * (2 * npairs + 1), 2)
    assert gem.overlap(phi, params) == 0

    # Ground-state Slater determinant
    phi = int("1" * 2 * npairs, 2)
    assert gem.overlap(phi, params) == gem.permanent(matrix[:, :npairs])

    # Excited Slater determinant
    phi = int("1100" * npairs, 2)
    cols = [1, 3, 5]
    assert gem.overlap(phi, params) == gem.permanent(matrix[:, cols])

    # Partial derivative of overlap
    # Differentiated cofficient's column corresponds to occupied orbital
    assert 1.0 > gem.overlap(phi, params, 1) > 0.0

    # Differentiated cofficient's column does not correspond to occupied orbital
    assert gem.overlap(phi, params, 0) == 0


@test
def test_energy():
    """
    Test computation of energies in APIG.

    """

    # Let the wavefunction be the HF ground state; project onto HF ground-state
    npairs = 3
    norbs = 9

    # This requires non-trivial, Hermitian `ham` arrays
    one_ham = make_hermitian(np.random.rand(norbs, norbs))
    two_ham = make_hermitian(np.random.rand(norbs, norbs, norbs, norbs))
    ham = (one_ham, two_ham)
    params = np.eye(npairs, M=norbs)

    gem = APIG(npairs, norbs, ham, init_params=params)

    # h_{ii} + h_{\bar{i}\bar{i}}
    one_electron = sum(2 * ham[0][i, i] for i in range(npairs))
    # V_{i\bar{i}i\bar{i}
    two_electron = sum(ham[1][i, i, i, i] for i in range(npairs))
    # 4*V_{ijij} - 2*V_{ijji}
    two_electron += sum([4 * ham[1][i, j, i, j] - 2 * ham[1][i, j, j, i]
                         for j in range(npairs) for i in range(npairs) if j > i])

    # Verify each energy computation backend
    actual_energy = one_electron + two_electron
    computed_energy = sum(gem.compute_energy(gem.ground_sd, params))
    assert np.allclose(computed_energy, actual_energy)
    computed_energy = sum(super(APIG, gem).compute_energy(gem.ground_sd, params))
    assert np.allclose(computed_energy, actual_energy)
    computed_energies = gem.energies
    assert np.allclose(computed_energies["one_electron"], one_electron)
    assert np.allclose(computed_energy - computed_energies["one_electron"], two_electron)

    # Verify some minimal-basis computations by hand; let the wavefunction be a
    # combination of the HF ground state and its first pair-excitation; project onto HF
    # ground-state
    npairs = 2
    norbs = 5
    one_ham = make_hermitian(np.random.rand(norbs, norbs))
    two_ham = make_hermitian(np.random.rand(norbs, norbs, norbs, norbs))
    ham = (one_ham, two_ham)
    params = np.eye(npairs, M=norbs)
    params[:, 2] = 1
    gem = APIG(npairs, norbs, ham, init_params=params)
    one_electron = 2 * (ham[0][0, 0] + ham[0][1, 1])
    two_electron = 4 * ham[1][0, 1, 0, 1] - 2 * ham[1][0, 1, 1, 0] \
        + ham[1][0, 0, 0, 0] + ham[1][1, 1, 1, 1] \
        + ham[1][0, 0, 2, 2] + ham[1][1, 1, 2, 2]

    # Verify each energy computation backend
    actual_energy = one_electron + two_electron
    computed_energy = sum(gem.compute_energy(gem.ground_sd, params))
    assert np.allclose(computed_energy, actual_energy)
    computed_energy = sum(super(APIG, gem).compute_energy(gem.ground_sd, params))
    assert np.allclose(computed_energy, actual_energy)
    computed_energies = gem.energies
    assert np.allclose(computed_energies["one_electron"], one_electron)
    assert np.allclose(computed_energy - computed_energies["one_electron"], two_electron)

    # Add a double pair-excitation to the second and third spatial orbitals to the
    # previous wavefunction; project onto HF ground state
    params[:, 3] = 1
    gem = APIG(npairs, norbs, ham, init_params=params)
    two_electron += ham[1][0, 0, 3, 3] + ham[1][1, 1, 3, 3]

    # Verify each energy computation backend
    actual_energy = one_electron + two_electron
    computed_energy = sum(gem.compute_energy(gem.ground_sd, params))
    assert np.allclose(computed_energy, actual_energy)
    computed_energy = sum(super(APIG, gem).compute_energy(gem.ground_sd, params))
    assert np.allclose(computed_energy, actual_energy)
    computed_energies = gem.energies
    assert np.allclose(computed_energies["one_electron"], one_electron)
    assert np.allclose(computed_energy - computed_energies["one_electron"], two_electron)


@test
def test_solve():
    """
    Test APIG.solve_params() and APIG.nonlin() by using them to optimize some APIG
    coefficients, using output from HORTON as input to the APIG instance.

    """

    fn = "test/h2.xyz"
    basis = "cc-pvdz"
    npairs = 1
    horton_result = ap1rog_from_horton(fn=fn, basis=basis, npairs=npairs, guess="apig")
    horton_basis = horton_result["basis"]
    horton_ham = horton_result["ham"]
    horton_params = horton_result["coeffs"].ravel()
    gem = APIG(npairs, horton_basis.nbasis, horton_ham)
    apig_result = gem.solve_params(x0=horton_params, verbose=True)
    assert apig_result["success"]


@test
def test_nonlin_jac_basic():
    """
    Run basic tests on APIG.nonlin_jac().

    See test_nonlin_jac_slow() for more rigorous, but slow tests.

    """

    # Check for basic correspondence between APIG.nonlin() and APIG.nonlin_jac()
    npairs = 3
    norbs = 9
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = APIG(npairs, norbs, ham)
    params = np.random.rand(npairs * norbs).reshape(npairs, norbs)
    params[:, :npairs] += np.eye(npairs)
    x0 = params.ravel()
    fun = gem.nonlin(x0, gem.pspace)
    jac = gem.nonlin_jac(x0, gem.pspace)
    assert jac.shape == (fun.size, params.size)
    assert not is_singular(jac)

@slow
def test_nonlin_jac_slow():
    """
    Verify via the finite-difference approximation that the output of APIG.nonlin_jac()
    represents the Jacobian of the output of APIG.nonlin().

    """

    npairs = 2
    norbs = 5
    one_ham = make_hermitian(np.random.rand(norbs, norbs))
    two_ham = make_hermitian(np.random.rand(norbs, norbs, norbs, norbs))
    ham = (one_ham, two_ham)
    gem = APIG(npairs, norbs, ham)
    params = np.random.rand(npairs * norbs).reshape(npairs, norbs)
    params[:, :npairs] += np.eye(npairs)
    x0 = params.ravel()
    fun = lambda x: gem.nonlin(x, gem.pspace)
    jac = lambda x: gem.nonlin_jac(x, gem.pspace)
    deriv_check(fun, jac, x0)


run_tests()

# vim: set textwidth=90 :
