"""
Unit tests for geminals.fancy_ci.FancyCI.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from romin import deriv_check
from geminals.fancy_ci import FancyCI
from geminals.horton_wrapper import ap1rog_from_horton
from geminals.slater_det import excite_orbs, add_pairs, excite_pairs
from geminals.test.common import *


@test
def test_init():
    """
    Check that FancyCI.__init__() works and that initialization errors are caught.

    """

    nelec = 2
    norbs = 4
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))

    # Non-int nelec
    f = lambda: FancyCI(float(nelec), norbs, ham)
    assert raises_exception(f, AssertionError)

    # Non-int norbs
    f = lambda: FancyCI(nelec, float(norbs), ham)
    assert raises_exception(f, AssertionError)

    # nelec > norbs
    f = lambda: FancyCI(norbs+1, norbs, ham)
    assert raises_exception(f, AssertionError)

    # No `ham` specified
    f = lambda: FancyCI(nelec, norbs)
    assert raises_exception(f, TypeError)

    # `ham` arrays of wrong dimension
    ham = (np.ones((norbs + 1, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    f = lambda: FancyCI(nelec, norbs, ham)
    assert raises_exception(f, AssertionError)
    ham = (np.ones((norbs, norbs)), np.ones((nelec, nelec, nelec, nelec)))
    f = lambda: FancyCI(nelec, norbs, ham)
    assert raises_exception(f, AssertionError)

    # Non-Hermitian `ham` arrays
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    ham[0][:, nelec:] *= 0.0
    f = lambda: FancyCI(nelec, norbs, ham)
    assert raises_exception(f, AssertionError)
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    ham[1][:, nelec:, :, :] *= 0.0
    f = lambda: FancyCI(nelec, norbs, ham)
    assert raises_exception(f, AssertionError)

    # params of wrong dimension
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    params = np.ones((norbs, 1))
    f = lambda: FancyCI(nelec, norbs, ham, init_params=params)
    assert raises_exception(f, AssertionError)

    # Non-np.ndarray params
    params = 1
    f = lambda: FancyCI(nelec, norbs, ham, init_params=params)
    assert raises_exception(f, AssertionError)

    # pspace of wrong type
    params = np.ones(norbs)
    pspace = 1
    f = lambda: FancyCI(nelec, norbs, ham, pspace=pspace)
    assert raises_exception(f, AssertionError)

    # Slater det. with wrong number of electrons in pspace
    pspace = list(FancyCI(nelec, norbs, ham).pspace)
    badphi = add_pairs(min(pspace), norbs - 1)
    pspace.append(badphi)
    f = lambda: FancyCI(nelec, norbs, ham, pspace=pspace)
    assert raises_exception(f, AssertionError)
    pspace.remove(badphi)

    # Slater det. with electrons in spatial orbital k > norbs in pspace
    badphi = excite_pairs(min(pspace), 0, norbs + 1)
    pspace.append(badphi)
    f = lambda: FancyCI(nelec, norbs, ham, pspace=pspace)
    assert raises_exception(f, AssertionError)
    pspace.remove(badphi)

@test
def test_properties():
    """
    Test FancyCI's properties' getters and setters.

    """

    nelec = 2
    norbs = 5
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)), 1.0)
    params = np.ones(norbs)
    pspace = FancyCI(nelec, norbs, ham).pspace
    gem = FancyCI(nelec, norbs, ham, init_params=params, pspace=pspace)
    assert gem._nelec == nelec
    assert gem.nelec == nelec
    assert gem._norbs == norbs
    assert gem._ham == ham[:3]
    assert gem._ham[2] == gem.core_energy == ham[2]
    assert np.allclose(gem._params, params)
    assert sorted(gem._pspace) == sorted(pspace)
    assert gem.ground_sd == min(pspace)
    assert not gem._exclude_ground
    assert gem._normalize


@test
def test_generate_init_and_params():
    """
    Test FancyCI._generate_init()

    """

    nelec = 3
    norbs = 6
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = FancyCI(nelec, norbs, ham)
    init_guess = gem._generate_init()
    assert init_guess.shape == (init_guess.size,)


@test
def test_generate_pspace():
    """
    Test FancyCI.generate_pspace().

    """

    nelec = 1
    norbs = 2
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = FancyCI(nelec, norbs, ham)
    assert gem.pspace == (0b0001, 0b0010, 0b0100, 0b1000,)

    nelec = 2
    norbs = 4
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = FancyCI(nelec, norbs, ham)
    assert gem.pspace == (0b00000011, 0b00000101, 0b00001001, 0b00010001, 0b00100001, 0b01000001, 0b10000001,
                          0b00000110, 0b00001010, 0b00010010, 0b00100010, 0b01000010, 0b10000010,
                          0b00001100, 0b00010100, 0b00100100, 0b01000100, 0b10000100,
                          0b00011000, 0b00101000, 0b01001000, 0b10001000,
                          0b00110000, 0b01010000, 0b10010000,
                          0b01100000, 0b10100000,
                          0b11000000)

    nelec = 3
    norbs = 3
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = FancyCI(nelec, norbs, ham)
    assert gem.pspace == (0b000111, 0b001011, 0b010011, 0b100011,
                          0b001101, 0b010101, 0b100101,
                          0b001110, 0b010110, 0b100110,
                          0b011001, 0b101001, 0b110001,
                          0b011010, 0b101010, 0b110010,
                          0b011100, 0b101100, 0b110100,)

@test
def test_overlap():
    """
    Test FancyCI.overlap().

    """

    nelec = 3
    norbs = 6
    ham = (np.ones((norbs, norbs)), np.ones((norbs, norbs, norbs, norbs)))
    gem = FancyCI(nelec, norbs, ham)
    params = np.random.rand(norbs)

    # Bad Slater determinant
    phi = None
    assert gem.overlap(phi, params) == 0

    # Slater determinant with different number of electrons
    phi = int("1" * (2 * nelec + 1), 2)
    assert gem.overlap(phi, params) == 0

    phi = gem.ground_sd
    f = lambda: gem.overlap(phi, params)
    assert raises_exception(f, NotImplementedError)

    phi = gem.ground_sd
    f = lambda: gem.overlap(phi, params, differentiate_wrt=2)
    assert raises_exception(f, NotImplementedError)


run_tests()

# vim: set textwidth=90 :
