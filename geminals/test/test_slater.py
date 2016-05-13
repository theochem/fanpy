import sys
sys.path.append('../')
import slater as sd


def test_occ():
    """
    Test sd.occ

    """
    # Test occupancy
    assert sd.occ(0b100100, 2)
    assert sd.occ(0b100100, 5)
    assert not sd.occ(0b100100, 4)
    assert not sd.occ(0b100100, 6)
    assert not sd.occ(0b100100, 0)

def test_total_occ():
    """
    Test sd.total_occ
    """
    assert sd.total_occ(None) == 0
    assert sd.total_occ(0) == 0
    assert sd.total_occ(0b0) == 0

    assert sd.total_occ(0b1000000) == 1
    assert sd.total_occ(0b1010000) == 2
    assert sd.total_occ(0b1010100) == 3

def test_occ_indices():
    """
    Test sd.occ_indices
    """
    assert sd.occ_indices(None) == []
    assert sd.occ_indices(0) == []
    assert sd.occ_indices(0b0) == []

    assert sd.occ_indices(0b1000000) == [6]
    assert sd.occ_indices(0b1010000) == [4,6]
    assert sd.occ_indices(0b1010100) == [2,4,6]

def test_vir_indices():
    """
    Test sd.vir_indices
    """
    assert sd.vir_indices(None, 0) == []

    assert sd.vir_indices(0, 3) == [0,1,2]
    assert sd.vir_indices(0b0, 4) == [0,1,2,3]

    assert sd.vir_indices(0b10000, 7) == [0,1,2,3,5,6]
    assert sd.vir_indices(0b10000, 6) == [0,1,2,3,5]
    assert sd.vir_indices(0b10000, 5) == [0,1,2,3]
    # FIXME: notice that number of orbitals can be less than the highest occupied index
    assert sd.vir_indices(0b10000, 4) == [0,1,2,3]
    assert sd.vir_indices(0b10000, 3) == [0,1,2]
    assert sd.vir_indices(0b10000, 2) == [0,1]
    assert sd.vir_indices(0b10000, 1) == [0]

def test_annihilate():
    """
    Test sd.annihilate.

    """
    # Remove orbitals that are not occupied
    assert sd.annihilate(0b00110, 0) is None
    assert sd.annihilate(0b00110, 3) is None
    assert sd.annihilate(0b00110, 4) is None
    assert sd.annihilate(0b00110, 5) is None
    assert sd.annihilate(0b00110, 6) is None

    # Remove orbitals that are occupied
    assert sd.annihilate(0b00110, 1) == 0b100
    assert sd.annihilate(0b00110, 2) == 0b010

    # Remove multiple orbitals
    assert sd.annihilate(0b01110, 1, 2) == 0b1000
    assert sd.annihilate(0b01110, 2, 1) == 0b1000

    # Remove orbital multiple times
    assert sd.annihilate(0b00110, 2, 2) is None
    assert sd.annihilate(0b00110, 1, 1) is None

    # Large index
    n = 9999999
    assert sd.annihilate(0b1 | 1 << n, n) == 0b1

def test_create():
    """
    Test sd.create().

    """
    # Add orbitals that are not occupied
    assert sd.create(0b00110, 0) == 0b111
    assert sd.create(0b00110, 3) == 0b1110
    assert sd.create(0b00110, 4) == 0b10110
    assert sd.create(0b00110, 5) == 0b100110
    assert sd.create(0b00110, 6) == 0b1000110

    # Add orbitals that are occupied
    assert sd.create(0b00110, 1) is None
    assert sd.create(0b00110, 2) is None

    # Add multiple orbitals
    assert sd.create(0b01000, 1, 2) == 0b1110
    assert sd.create(0b01000, 2, 1) == 0b1110

    # Add orbital multiple times
    assert sd.create(0b01100, 1, 1) is None
    assert sd.create(0b01010, 2, 2) is None
    assert sd.create(0b01100, 1, 1, 1) is None
    assert sd.create(0b01010, 2, 2, 2) is None

    # Large index
    n = 9999999
    assert sd.create(0b1, n) == 0b1 | 1 << n

def test_excite():
    """
    Test excite().

    """
    # Excite spatial orbitals from occupied to virtual
    assert sd.excite(0b0001, 0, 1) == 0b10
    assert sd.excite(0b0001, 0, 5) == 0b100000
    assert sd.excite(0b1000, 3, 0) == 0b1

    # Excite spatial orbitals from virtual to virtual
    assert sd.excite(0b00100, 0, 1) is None
    assert sd.excite(0b00100, 9999, 1) is None

    # Excite spatial orbitals from occupied to occupied
    assert sd.excite(0b1001, 3, 0) is None

    # Large index
    assert sd.excite(0b1001, 3, 999999) == 0b0001 | 1 << 999999

    # Excite to the same orbital
    assert sd.excite(0b111100, 3, 3) == 0b111100
    assert sd.excite(0b111100, 0, 0) is None
    assert sd.excite(0b111100, 3, 3, 3, 3) is None

def test_ground():
    """
    Test sd.ground

    """
    assert sd.ground(1) == 0b1
    assert sd.ground(2) == 0b11
    assert sd.ground(3) == 0b111
    assert sd.ground(5) == 0b11111
    assert sd.ground(8) == 0b11111111

test_occ()
test_total_occ()
test_occ_indices()
test_vir_indices()
test_annihilate()
test_create()
test_excite()
test_ground()
