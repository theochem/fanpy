from nose.tools import assert_raises

from .. import slater


def test_occ():
    """
    Test slater.occ
    """
    # Test occupancy
    assert slater.occ(0b100100, 2)
    assert slater.occ(0b100100, 5)
    assert not slater.occ(0b100100, 4)
    assert not slater.occ(0b100100, 6)
    assert not slater.occ(0b100100, 0)


def test_total_occ():
    """
    Test slater.total_occ
    """
    assert slater.total_occ(None) == 0
    assert slater.total_occ(0) == 0
    assert slater.total_occ(0b0) == 0
    assert slater.total_occ(0b1000000) == 1
    assert slater.total_occ(0b1010000) == 2
    assert slater.total_occ(0b1010100) == 3


def test_annihilate():
    """
    Test slater.annihilate.
    """
    # Remove orbitals that are not occupied
    assert slater.annihilate(0b00110, 0) is None
    assert slater.annihilate(0b00110, 3) is None
    assert slater.annihilate(0b00110, 4) is None
    assert slater.annihilate(0b00110, 5) is None
    assert slater.annihilate(0b00110, 6) is None

    # Remove orbitals that are occupied
    assert slater.annihilate(0b00110, 1) == 0b100
    assert slater.annihilate(0b00110, 2) == 0b010

    # Remove multiple orbitals
    assert slater.annihilate(0b01110, 1, 2) == 0b1000
    assert slater.annihilate(0b01110, 2, 1) == 0b1000

    # Remove orbital multiple times
    assert slater.annihilate(0b00110, 2, 2) is None
    assert slater.annihilate(0b00110, 1, 1) is None

    # Large index
    n = 9999999
    assert slater.annihilate(0b1 | 1 << n, n) == 0b1


def test_create():
    """
    Test slater.create().
    """
    # Add orbitals that are not occupied
    assert slater.create(0b00110, 0) == 0b111
    assert slater.create(0b00110, 3) == 0b1110
    assert slater.create(0b00110, 4) == 0b10110
    assert slater.create(0b00110, 5) == 0b100110
    assert slater.create(0b00110, 6) == 0b1000110

    # Add orbitals that are occupied
    assert slater.create(0b00110, 1) is None
    assert slater.create(0b00110, 2) is None

    # Add multiple orbitals
    assert slater.create(0b01000, 1, 2) == 0b1110
    assert slater.create(0b01000, 2, 1) == 0b1110

    # Add orbital multiple times
    assert slater.create(0b01100, 1, 1) is None
    assert slater.create(0b01010, 2, 2) is None
    assert slater.create(0b01100, 1, 1, 1) is None
    assert slater.create(0b01010, 2, 2, 2) is None

    # Large index
    n = 9999999
    assert slater.create(0b1, n) == 0b1 | 1 << n


def test_excite():
    """
    Test excite().
    """
    # Excite spatial orbitals from occupied to virtual
    assert slater.excite(0b0001, 0, 1) == 0b10
    assert slater.excite(0b0001, 0, 5) == 0b100000
    assert slater.excite(0b1000, 3, 0) == 0b1

    # Excite spatial orbitals from virtual to virtual
    assert slater.excite(0b00100, 0, 1) is None
    assert slater.excite(0b00100, 9999, 1) is None

    # Excite spatial orbitals from occupied to occupied
    assert slater.excite(0b1001, 3, 0) is None

    # Large index
    assert slater.excite(0b1001, 3, 999999) == 0b0001 | 1 << 999999

    # Excite to the same orbital
    assert slater.excite(0b111100, 3, 3) == 0b111100
    assert slater.excite(0b111100, 0, 0) is None
    assert slater.excite(0b111100, 3, 3, 3, 3) is None


def test_ground():
    """
    Test slater.ground
    """
    assert_raises(AssertionError, lambda: slater.ground(2, 1))
    print(slater.ground(2, 2))
    assert slater.ground(2, 2) == 0b11
    assert_raises(AssertionError, lambda: slater.ground(2, 3))
    assert slater.ground(2, 4) == 0b0101
    assert slater.ground(2, 6) == 0b001001
    assert slater.ground(2, 8) == 0b00010001
    assert_raises(AssertionError, lambda: slater.ground(3, 2))
    assert slater.ground(3, 4) == 0b0111
    assert slater.ground(3, 6) == 0b001011


def test_occ_indices():
    """
    Test slater.occ_indices
    """
    assert slater.occ_indices(None) == ()
    assert slater.occ_indices(0) == ()
    assert slater.occ_indices(0b0) == ()

    assert slater.occ_indices(0b1000000) == (6,)
    assert slater.occ_indices(0b1010000) == (4, 6)
    assert slater.occ_indices(0b1010100) == (2, 4, 6)


def test_vir_indices():
    """
    Test slater.vir_indices
    """
    assert slater.vir_indices(None, 0) == ()

    assert slater.vir_indices(0, 3) == (0, 1, 2)
    assert slater.vir_indices(0b0, 4) == (0, 1, 2, 3)

    assert slater.vir_indices(0b10000, 7) == (0, 1, 2, 3, 5, 6)
    assert slater.vir_indices(0b10000, 6) == (0, 1, 2, 3, 5)
    assert slater.vir_indices(0b10000, 5) == (0, 1, 2, 3)
    # FIXME: notice that number of orbitals can be less than the highest occupied index
    assert slater.vir_indices(0b10000, 4) == (0, 1, 2, 3)
    assert slater.vir_indices(0b10000, 3) == (0, 1, 2)
    assert slater.vir_indices(0b10000, 2) == (0, 1)
    assert slater.vir_indices(0b10000, 1) == (0,)


def test_shared_indices():
    """
    Test slater.shared_indices
    """
    assert slater.shared_indices(None, None) == ()
    assert slater.shared_indices(0b001, None) == ()
    assert slater.shared_indices(None, 0b000) == ()
    assert slater.shared_indices(0b001, 0b000) == ()
    assert slater.shared_indices(0b111, 0b001) == (0,)
    assert slater.shared_indices(0b111, 0b101) == (0, 2)
    assert slater.shared_indices(0b111, 0b01101) == (0, 2)
    assert slater.shared_indices(0b111110, 0b101100) == (2, 3, 5)


def test_diff_indices():
    """
    Test slater.diff_indices
    """
    assert slater.diff_indices(None, None) == ((), ())
    assert slater.diff_indices(None, 0b000) == ((), ())
    assert slater.diff_indices(None, 0b010) == ((), (1,))
    assert slater.diff_indices(0b001, None) == ((0,), ())
    assert slater.diff_indices(0b001, 0b000) == ((0,), ())
    assert slater.diff_indices(0b001, 0b001) == ((), ())
    assert slater.diff_indices(0b011, 0b101) == ((1,), (2,))
    assert slater.diff_indices(0b101, 0b011) == ((2,), (1,))
    assert slater.diff_indices(0b111110, 0b101100) == ((1, 4), ())


def test_combine_spin():
    """
    Test slater.combine_spin
    """
    assert_raises(AssertionError, lambda: slater.combine_spin(0b0, 0b0, 0))
    assert_raises(AssertionError, lambda: slater.combine_spin(0b0, 0b0, -1))
    assert slater.combine_spin(0b1, 0b0, 1) == 0b1
    assert slater.combine_spin(0b0, 0b1, 1) == 0b10
    assert slater.combine_spin(0b111, 0b0, 3) == 0b111
    assert slater.combine_spin(0b000, 0b111, 3) == 0b111000
    assert slater.combine_spin(0b011, 0b001, 5) == 0b0000100011
    assert slater.combine_spin(0b001, 0b011, 5) == 0b0001100001
    assert slater.combine_spin(0b000, 0b111, 5) == 0b0011100000


def test_split_spin():
    """
    Test slater.split_spin
    """
    assert_raises(AssertionError, lambda: slater.split_spin(0b0, 0))
    assert_raises(AssertionError, lambda: slater.split_spin(0b0, -1))
    assert slater.split_spin(0b1, 1) == (0b1, 0b0)
    assert slater.split_spin(0b10, 1) == (0b0, 0b1)
    assert slater.split_spin(0b111, 3) == (0b111, 0b0)
    assert slater.split_spin(0b111000, 3) == (0b000, 0b111)
    assert slater.split_spin(0b0000100011, 5) == (0b011, 0b001)
    assert slater.split_spin(0b0001100001, 5) == (0b001, 0b011)
    assert slater.split_spin(0b0011100000, 5) == (0b000, 0b111)


def test_interleave():
    """
    Test slater.interleave
    """
    assert slater.interleave(0b11, 1) == 0b11
    assert slater.interleave(0b0011, 2) == 0b0101
    assert slater.interleave(0b000011, 3) == 0b000101
    assert slater.interleave(0b0101, 2) == 0b0011
    assert slater.interleave(0b000101, 3) == 0b010001


def test_deinterleave():
    """
    Test slater.deinterleave
    """
    assert slater.deinterleave(0b11, 1) == 0b11
    assert slater.deinterleave(0b0101, 2) == 0b0011
    assert slater.deinterleave(0b000101, 3) == 0b000011
    assert slater.deinterleave(0b0011, 2) == 0b0101
    assert slater.deinterleave(0b010001, 3) == 0b000101
