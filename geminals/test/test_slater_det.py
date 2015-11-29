""" Tests for slater_det.py
"""

from slater_det import remove_orbs, add_orbs, excite_orbs, is_occupied
from slater_det import remove_pairs, add_pairs, excite_pairs, is_pair_occupied

def test_remove_orbs():
    """ Test remove_orbs function
    """
    # Remove orbs that are not occupied
    assert remove_orbs(0b00110, 0) is None
    assert remove_orbs(0b00110, 3) is None
    assert remove_orbs(0b00110, 4) is None
    assert remove_orbs(0b00110, 5) is None
    assert remove_orbs(0b00110, 6) is None
    # Remove orbs that are occupied
    assert remove_orbs(0b00110, 1) == 0b100
    assert remove_orbs(0b00110, 2) == 0b010
    # Remove multiple orbitals
    assert remove_orbs(0b01110, 1, 2) == 0b1000
    assert remove_orbs(0b01110, 2, 1) == 0b1000
    # Remove orb multiple times
    assert remove_orbs(0b00110, 2, 2) is None
    assert remove_orbs(0b00110, 1, 1) is None
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
    assert add_orbs(0b00110, 1) is None
    assert add_orbs(0b00110, 2) is None
    # Add multiple orbitals
    assert add_orbs(0b01000, 1, 2) == 0b1110
    assert add_orbs(0b01000, 2, 1) == 0b1110
    # Add orb multiple times
    assert add_orbs(0b01100, 1, 1) is None
    assert add_orbs(0b01010, 2, 2) is None
    assert add_orbs(0b01100, 1, 1, 1) is None
    assert add_orbs(0b01010, 2, 2, 2) is None
    # Large index
    n = 9999999
    assert add_orbs(0b1, n) == 0b1 | 1 << n

def test_excite_orbs():
    """ Test excite_orbs function
    """
    # Excite spatial orbitals from occupied to virtual
    assert excite_orbs(0b0001, 0, 1) == 0b10
    assert excite_orbs(0b0001, 0, 5) == 0b100000
    assert excite_orbs(0b1000, 3, 0) == 0b1
    # Excite spatial orbitals from virtual to virtual
    assert excite_orbs(0b00100, 0, 1) is None
    assert excite_orbs(0b00100, 9999, 1) is None
    # Excite spatial orbitals from occupied to occupied
    assert excite_orbs(0b1001, 3, 0) is None
    # Large index
    assert excite_orbs(0b1001, 3, 999999) == 0b0001 | 1 << 999999
    # Excite to the same orbital
    assert excite_orbs(0b111100, 3, 3) == 0b111100
    assert excite_orbs(0b111100, 0, 0) is None
    assert excite_orbs(0b111100, 3, 3, 3, 3) is None

def test_remove_pairs():
    """ Test remove_pairs function
    """
    # Remove pairs that are not occupied
    assert remove_pairs(0b001100, 0) is None
    assert remove_pairs(0b001100, 2) is None
    assert remove_pairs(0b001100, 3) is None
    # Remove pairs that are partially occupied
    assert remove_pairs(0b001010, 0) is None
    assert remove_pairs(0b000101, 0) is None
    # Remove pairs that are occupied
    assert remove_pairs(0b001111, 1) == 0b11
    # Remove multiple pairs
    assert remove_pairs(0b11111, 0, 1) == 0b10000
    assert remove_pairs(0b11111, 1, 0) == 0b10000
    # Remove pairs multiple times
    assert remove_pairs(0b001111, 1, 1) is None
    assert remove_pairs(0b001111, 1, 1, 1) is None
    # Large index
    n = 999999
    assert remove_pairs(0b1 | 0b11 << n*2, n) == 0b1

def test_add_pairs():
    """ Test add_pairs function
    """
    # Add pairs that are not occupied
    assert add_pairs(0b001100, 0) == 0b1111
    assert add_pairs(0b001100, 2) == 0b111100
    assert add_pairs(0b001100, 3) == 0b11001100
    assert add_pairs(0b001100, 4) == 0b1100001100
    # Add pairs that are occupied
    assert add_pairs(0b001100, 1) is None
    # Add pairs that are partially occupied
    assert add_pairs(0b001001, 0) is None
    assert add_pairs(0b001001, 1) is None
    # Add multiple pairs
    assert add_pairs(0b01100, 0, 2) == 0b111111
    assert add_pairs(0b01100, 2, 0) == 0b111111
    # Add pairs multiple times
    assert add_pairs(0b00011, 0, 0) is None
    assert add_pairs(0b00011, 1, 1) is None
    assert add_pairs(0b00011, 0, 0, 0) is None
    assert add_pairs(0b00011, 1, 1, 1) is None
    # Large index
    n = 9999999
    assert add_pairs(0b1, n) == 0b1 | 0b11 << n*2

def test_excite_pairs():
    """ Test excite_pairs function
    """
    # Excite_Pairs from occupied to virtual
    assert excite_pairs(0b0011, 0, 1) == 0b1100
    assert excite_pairs(0b0011, 0, 2) == 0b110000
    assert excite_pairs(0b110000, 2, 0) == 0b11
    # Excite_Pairs from virtual to virtual
    assert excite_pairs(0b00100, 0, 2) is None
    assert excite_pairs(0b00100, 9999, 1) is None
    # Excite_Pairs from occupied to occupied
    assert excite_pairs(0b110011, 2, 0) is None
    # Excite pairs from partially occupied
    assert excite_pairs(0b100011, 2, 1) is None
    assert excite_pairs(0b100011, 2, 0) is None
    assert excite_pairs(0b100011, 2, 3) is None
    # Large index
    assert excite_pairs(0b0111, 0, 999999) == 0b100 | 0b11 << 999999*2
    # Excite to the same spatial orbital
    assert excite_pairs(0b111100, 1, 1) == 0b111100
    assert excite_pairs(0b111100, 0, 0) is None
    assert excite_pairs(0b111100, 1, 1, 1, 1) is None

def test_is_occupied():
    """ Test is_occupied function
    """
    # Test occupancy
    assert is_occupied(0b100100, 2)
    assert is_occupied(0b100100, 5)
    assert not is_occupied(0b100100, 4)
    assert not is_occupied(0b100100, 6)
    assert not is_occupied(0b100100, 0)

def test_is_pair_occupied():
    """ Test is_pair_occupied function
    """
    # Test occupancy
    assert is_pair_occupied(0b11111100, 1)
    assert is_pair_occupied(0b11111100, 2)
    assert is_pair_occupied(0b11111100, 3)
    assert not is_pair_occupied(0b11111100, 0)
    assert not is_pair_occupied(0b11111100, 4)
    assert not is_pair_occupied(0b11111100, 9)
test_remove_orbs()
test_add_orbs()
test_excite_orbs()
test_remove_pairs()
test_add_pairs()
test_excite_pairs()
