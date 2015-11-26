""" Tests for slater_det.py
"""

from slater_det import remove_orbs, add_orbs, excite, is_occupied

def test_remove_orbs():
    """ Test remove_orbs function
    """
    # Remove orbs that are not occupied
    assert remove_orbs(0b00110, 0) == 0
    assert remove_orbs(0b00110, 3) == 0
    assert remove_orbs(0b00110, 4) == 0
    assert remove_orbs(0b00110, 5) == 0
    assert remove_orbs(0b00110, 6) == 0
    # Remove orbs that are occupied
    assert remove_orbs(0b00110, 1) == 0b100
    assert remove_orbs(0b00110, 2) == 0b010
    # Remove multiple orbitals
    assert remove_orbs(0b01110, 1, 2) == 0b1000
    assert remove_orbs(0b01110, 2, 1) == 0b1000
    # Remove orb multiple times
    assert remove_orbs(0b00110, 2, 2) == 0
    assert remove_orbs(0b00110, 1, 1) == 0
    # Large index
    n = 9999999
    assert remove_orbs(0b1 | 1 << n, n) == 0b1

def test_add_orbs():
    """ Test add_orbs function
    """
    # Add orbs that are not occupied
    assert add_orbs(0b00110, 0) == 0b111
    assert add_orbs(0b00110, 3) == 0b1110
    assert add_orbs(0b00110, 4) == 0b10110
    assert add_orbs(0b00110, 5) == 0b100110
    assert add_orbs(0b00110, 6) == 0b1000110
    # Add orbs that are occupied
    assert add_orbs(0b00110, 1) == 0
    assert add_orbs(0b00110, 2) == 0
    # Add multiple orbitals
    assert add_orbs(0b01000, 1, 2) == 0b1110
    assert add_orbs(0b01000, 2, 1) == 0b1110
    # Add orb multiple times
    assert add_orbs(0b01010, 2, 2) == 0
    assert add_orbs(0b01100, 1, 1) == 0
    # Large index
    n = 9999999
    assert add_orbs(0b1, n) == 0b1 | 1 << n

def test_excite():
    """ Test excite function
    """
    # Excite from occupied to virtual
    assert excite(0b0001, 0, 1) == 0b10
    assert excite(0b0001, 0, 5) == 0b100000
    assert excite(0b1000, 3, 0) == 0b1
    # Excite from virtual to virtual
    assert excite(0b00100, 0, 1) == 0
    assert excite(0b00100, 9999, 1) == 0
    # Excite from occupied to occupied
    assert excite(0b1001, 3, 0) == 0
    # Large index
    assert excite(0b1001, 3, 999999) == 0b0001 | 1 << 999999

def test_is_occupied():
    """ Test is_occupied function
    """
    # Test occupancy
    assert is_occupied(0b100100, 2)
    assert is_occupied(0b100100, 5)
    assert not is_occupied(0b100100, 4)
    assert not is_occupied(0b100100, 6)
    assert not is_occupied(0b100100, 0)
