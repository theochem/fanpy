"""Test wfns.slater."""
from nose.tools import assert_raises
import gmpy2
from wfns.backend import slater


def test_occ():
    """Test slater.occ."""
    assert slater.occ(0b100100, 2)
    assert slater.occ(0b100100, 5)
    assert not slater.occ(0b100100, 4)
    assert not slater.occ(0b100100, 6)
    assert not slater.occ(0b100100, 0)


def test_is_alpha():
    """Test slater.is_alpha."""
    assert slater.is_alpha(0, 1)
    assert not slater.is_alpha(1, 1)

    assert slater.is_alpha(0, 4)
    assert slater.is_alpha(1, 4)
    assert slater.is_alpha(2, 4)
    assert slater.is_alpha(3, 4)
    assert not slater.is_alpha(4, 4)
    assert not slater.is_alpha(5, 4)
    assert not slater.is_alpha(6, 4)
    assert not slater.is_alpha(7, 4)

    assert_raises(ValueError, slater.is_alpha, -1, 4)
    assert_raises(ValueError, slater.is_alpha, 99, 4)


def test_spatial_index():
    """Test slater.spatial_index."""
    assert slater.spatial_index(0, 1) == 0
    assert slater.spatial_index(1, 1) == 0

    assert slater.spatial_index(0, 4) == 0
    assert slater.spatial_index(1, 4) == 1
    assert slater.spatial_index(2, 4) == 2
    assert slater.spatial_index(3, 4) == 3
    assert slater.spatial_index(4, 4) == 0
    assert slater.spatial_index(5, 4) == 1
    assert slater.spatial_index(6, 4) == 2
    assert slater.spatial_index(7, 4) == 3

    assert_raises(ValueError, slater.is_alpha, -1, 4)
    assert_raises(ValueError, slater.is_alpha, 99, 4)


def test_total_occ():
    """Test slater.total_occ."""
    assert slater.total_occ(None) == 0
    assert slater.total_occ(0) == 0
    assert slater.total_occ(0b0) == 0
    assert slater.total_occ(0b1000000) == 1
    assert slater.total_occ(0b1010000) == 2
    assert slater.total_occ(0b1010100) == 3


def test_annihilate():
    """Test slater.annihilate."""
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
    """Test slater.create."""
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
    """Test slater.excite."""
    # Check error
    assert_raises(ValueError, lambda: slater.excite(0b1111, 1, 2, 5))

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
    """Test slater.ground."""
    assert_raises(ValueError, lambda: slater.ground(2, 1))
    assert slater.ground(2, 2) == 0b11
    assert_raises(ValueError, lambda: slater.ground(2, 3))
    assert slater.ground(2, 4) == 0b0101
    assert slater.ground(2, 6) == 0b001001
    assert slater.ground(2, 8) == 0b00010001
    assert_raises(ValueError, lambda: slater.ground(3, 2))
    assert slater.ground(3, 4) == 0b0111
    assert slater.ground(3, 6) == 0b001011


def test_is_internal_sd():
    """Test slater.is_internal_sd."""
    assert slater.is_internal_sd(gmpy2.mpz(2))
    assert not slater.is_internal_sd(2)
    assert not slater.is_internal_sd([5])
    assert not slater.is_internal_sd(None)
    assert not slater.is_internal_sd((3))


def test_internal_sd():
    """Test slater.internal_sd."""
    # Check error
    assert_raises(TypeError, lambda: slater.internal_sd(None))
    assert_raises(TypeError, lambda: slater.internal_sd('5'))
    assert_raises(TypeError, lambda: slater.internal_sd([0, 3]))

    # integer
    assert slater.internal_sd(5) == gmpy2.mpz(5)
    assert isinstance(slater.internal_sd(5), type(gmpy2.mpz(5)))
    # gmpy2 object
    assert slater.internal_sd(gmpy2.mpz(5)) == gmpy2.mpz(5)
    assert isinstance(slater.internal_sd(gmpy2.mpz(5)), type(gmpy2.mpz(5)))


def test_occ_indices():
    """Test slater.occ_indices."""
    assert slater.occ_indices(None) == ()
    assert slater.occ_indices(0) == ()
    assert slater.occ_indices(0b0) == ()

    assert slater.occ_indices(0b1000000) == (6,)
    assert slater.occ_indices(0b1010000) == (4, 6)
    assert slater.occ_indices(0b1010100) == (2, 4, 6)


def test_vir_indices():
    """Test slater.vir_indices."""
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


def test_shared_orbs():
    """Test slater.shared_orbs."""
    assert slater.shared_orbs(0b001, 0b000) == ()
    assert slater.shared_orbs(0b111, 0b001) == (0,)
    assert slater.shared_orbs(0b111, 0b101) == (0, 2)


def test_diff_orbs():
    """Test slater.diff_orbs."""
    assert slater.diff_orbs(0b001, 0b000) == ((0,), ())
    assert slater.diff_orbs(0b001, 0b001) == ((), ())
    assert slater.diff_orbs(0b011, 0b101) == ((1,), (2,))
    assert slater.diff_orbs(0b101, 0b011) == ((2,), (1,))


def test_combine_spin():
    """Test slater.combine_spin."""
    assert_raises(ValueError, lambda: slater.combine_spin(0b0, 0b0, 0))
    assert_raises(ValueError, lambda: slater.combine_spin(0b0, 0b0, -1))
    assert slater.combine_spin(0b1, 0b0, 1) == 0b1
    assert slater.combine_spin(0b0, 0b1, 1) == 0b10
    assert slater.combine_spin(0b111, 0b0, 3) == 0b111
    assert slater.combine_spin(0b000, 0b111, 3) == 0b111000
    assert slater.combine_spin(0b011, 0b001, 5) == 0b0000100011
    assert slater.combine_spin(0b001, 0b011, 5) == 0b0001100001
    assert slater.combine_spin(0b000, 0b111, 5) == 0b0011100000


def test_split_spin():
    """Test slater.split_spin."""
    assert_raises(ValueError, lambda: slater.split_spin(0b0, 0))
    assert_raises(ValueError, lambda: slater.split_spin(0b0, -1))
    assert slater.split_spin(0b1, 1) == (0b1, 0b0)
    assert slater.split_spin(0b10, 1) == (0b0, 0b1)
    assert slater.split_spin(0b111, 3) == (0b111, 0b0)
    assert slater.split_spin(0b111000, 3) == (0b000, 0b111)
    assert slater.split_spin(0b0000100011, 5) == (0b011, 0b001)
    assert slater.split_spin(0b0001100001, 5) == (0b001, 0b011)
    assert slater.split_spin(0b0011100000, 5) == (0b000, 0b111)


def test_interleave_index():
    """Test slater.interleave_index."""
    # Check error
    assert_raises(ValueError, lambda: slater.interleave_index(-1, 4))
    assert_raises(ValueError, lambda: slater.interleave_index(8, 4))
    assert_raises(ValueError, lambda: slater.interleave_index(9, 4))
    # 1 spatial orbital
    assert slater.interleave_index(0, 1) == 0
    assert slater.interleave_index(1, 1) == 1
    # 2 spatial orbitals
    assert slater.interleave_index(0, 2) == 0
    assert slater.interleave_index(1, 2) == 2
    assert slater.interleave_index(2, 2) == 1
    assert slater.interleave_index(3, 2) == 3
    # 3 spatial orbitals
    assert slater.interleave_index(0, 3) == 0
    assert slater.interleave_index(1, 3) == 2
    assert slater.interleave_index(2, 3) == 4
    assert slater.interleave_index(3, 3) == 1
    assert slater.interleave_index(4, 3) == 3
    assert slater.interleave_index(5, 3) == 5


def test_deinterleave_index():
    """Test slater.deinterleave_index."""
    # Check error
    assert_raises(ValueError, lambda: slater.deinterleave_index(-1, 4))
    assert_raises(ValueError, lambda: slater.deinterleave_index(8, 4))
    assert_raises(ValueError, lambda: slater.deinterleave_index(9, 4))
    # 1 spatial orbital
    assert slater.deinterleave_index(0, 1) == 0
    assert slater.deinterleave_index(1, 1) == 1
    # 2 spatial orbitals
    assert slater.deinterleave_index(0, 2) == 0
    assert slater.deinterleave_index(1, 2) == 2
    assert slater.deinterleave_index(2, 2) == 1
    assert slater.deinterleave_index(3, 2) == 3
    # 3 spatial orbitals
    assert slater.deinterleave_index(0, 3) == 0
    assert slater.deinterleave_index(1, 3) == 3
    assert slater.deinterleave_index(2, 3) == 1
    assert slater.deinterleave_index(3, 3) == 4
    assert slater.deinterleave_index(4, 3) == 2
    assert slater.deinterleave_index(5, 3) == 5


def test_interleave():
    """Test slater.interleave."""
    assert_raises(ValueError, lambda: slater.interleave(0, -4))
    assert slater.interleave(0b11, 1) == 0b11
    assert slater.interleave(0b0011, 2) == 0b0101
    assert slater.interleave(0b000011, 3) == 0b000101
    assert slater.interleave(0b0101, 2) == 0b0011
    assert slater.interleave(0b000101, 3) == 0b010001


def test_deinterleave():
    """Test slater.deinterleave."""
    assert_raises(ValueError, lambda: slater.deinterleave(0, -4))
    assert slater.deinterleave(0b11, 1) == 0b11
    assert slater.deinterleave(0b0101, 2) == 0b0011
    assert slater.deinterleave(0b000101, 3) == 0b000011
    assert slater.deinterleave(0b0011, 2) == 0b0101
    assert slater.deinterleave(0b010001, 3) == 0b000101


def test_get_spin():
    """Test slater.get_spin."""
    # 0 spatial orbital
    assert_raises(ValueError, lambda: slater.get_spin(0b0000, 0))
    # 1 spatial orbital
    assert slater.get_spin(0b00, 1) == 0
    assert slater.get_spin(0b01, 1) == 0.5
    assert slater.get_spin(0b10, 1) == -0.5
    # 2 spatial orbital
    assert slater.get_spin(0b0000, 2) == 0
    assert slater.get_spin(0b0001, 2) == 0.5
    assert slater.get_spin(0b0010, 2) == 0.5
    assert slater.get_spin(0b0100, 2) == -0.5
    assert slater.get_spin(0b1000, 2) == -0.5
    assert slater.get_spin(0b0011, 2) == 1
    assert slater.get_spin(0b0101, 2) == 0
    assert slater.get_spin(0b1001, 2) == 0
    assert slater.get_spin(0b0110, 2) == 0
    assert slater.get_spin(0b1010, 2) == 0
    assert slater.get_spin(0b1100, 2) == -1
    assert slater.get_spin(0b0111, 2) == 0.5
    assert slater.get_spin(0b1011, 2) == 0.5
    assert slater.get_spin(0b1101, 2) == -0.5
    assert slater.get_spin(0b1110, 2) == -0.5
    assert slater.get_spin(0b1111, 2) == 0


def test_get_seniority():
    """Test slater.get_seniority."""
    # 0 spatial orbital
    assert_raises(ValueError, lambda: slater.get_seniority(0b0000, 0))
    assert_raises(ValueError, lambda: slater.get_seniority(0b0011, 0))
    # 1 spatial orbital
    assert slater.get_seniority(0b00, 1) == 0
    assert slater.get_seniority(0b01, 1) == 1
    assert slater.get_seniority(0b10, 1) == 1
    # 2 spatial orbital
    assert slater.get_seniority(0b0000, 2) == 0
    assert slater.get_seniority(0b0001, 2) == 1
    assert slater.get_seniority(0b0010, 2) == 1
    assert slater.get_seniority(0b0100, 2) == 1
    assert slater.get_seniority(0b1000, 2) == 1
    assert slater.get_seniority(0b0011, 2) == 2
    assert slater.get_seniority(0b0101, 2) == 0
    assert slater.get_seniority(0b1001, 2) == 2
    assert slater.get_seniority(0b0110, 2) == 2
    assert slater.get_seniority(0b1010, 2) == 0
    assert slater.get_seniority(0b1100, 2) == 2
    assert slater.get_seniority(0b0111, 2) == 1
    assert slater.get_seniority(0b1011, 2) == 1
    assert slater.get_seniority(0b1101, 2) == 1
    assert slater.get_seniority(0b1110, 2) == 1
    assert slater.get_seniority(0b1111, 2) == 0


def test_sign_perm():
    """Test slater.sign_perm."""
    assert slater.sign_perm([1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([1, 3, 6, 4], is_decreasing=False) == -1
    assert slater.sign_perm([1, 4, 3, 6], is_decreasing=False) == -1
    assert slater.sign_perm([1, 4, 6, 3], is_decreasing=False) == 1
    assert slater.sign_perm([1, 6, 3, 4], is_decreasing=False) == 1
    assert slater.sign_perm([1, 6, 4, 3], is_decreasing=False) == -1
    assert slater.sign_perm([3, 1, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([3, 1, 6, 4], is_decreasing=False) == 1
    assert slater.sign_perm([3, 4, 1, 6], is_decreasing=False) == 1
    assert slater.sign_perm([3, 4, 6, 1], is_decreasing=False) == -1
    assert slater.sign_perm([3, 6, 1, 4], is_decreasing=False) == -1
    assert slater.sign_perm([3, 6, 4, 1], is_decreasing=False) == 1
    assert slater.sign_perm([4, 1, 3, 6], is_decreasing=False) == 1
    assert slater.sign_perm([4, 1, 6, 3], is_decreasing=False) == -1
    assert slater.sign_perm([4, 3, 1, 6], is_decreasing=False) == -1
    assert slater.sign_perm([4, 3, 6, 1], is_decreasing=False) == 1
    assert slater.sign_perm([4, 6, 1, 3], is_decreasing=False) == 1
    assert slater.sign_perm([4, 6, 3, 1], is_decreasing=False) == -1
    assert slater.sign_perm([6, 1, 3, 4], is_decreasing=False) == -1
    assert slater.sign_perm([6, 1, 4, 3], is_decreasing=False) == 1
    assert slater.sign_perm([6, 3, 1, 4], is_decreasing=False) == 1
    assert slater.sign_perm([6, 3, 4, 1], is_decreasing=False) == -1
    assert slater.sign_perm([6, 4, 1, 3], is_decreasing=False) == -1
    assert slater.sign_perm([6, 4, 3, 1], is_decreasing=False) == 1

    assert slater.sign_perm([1, 3, 4, 6], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([1, 3, 6, 4], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([1, 4, 3, 6], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([1, 4, 6, 3], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([1, 6, 3, 4], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([1, 6, 4, 3], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([3, 1, 4, 6], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([3, 1, 6, 4], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([3, 4, 1, 6], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([3, 4, 6, 1], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([3, 6, 1, 4], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([3, 6, 4, 1], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([4, 1, 3, 6], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([4, 1, 6, 3], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([4, 3, 1, 6], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([4, 3, 6, 1], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([4, 6, 1, 3], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([4, 6, 3, 1], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([6, 1, 3, 4], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([6, 1, 4, 3], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([6, 3, 1, 4], [1, 3, 4, 6], is_decreasing=False) == 1
    assert slater.sign_perm([6, 3, 4, 1], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([6, 4, 1, 3], [1, 3, 4, 6], is_decreasing=False) == -1
    assert slater.sign_perm([6, 4, 3, 1], [1, 3, 4, 6], is_decreasing=False) == 1

    assert slater.sign_perm([1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([1, 3, 6, 4], is_decreasing=True) == -1
    assert slater.sign_perm([1, 4, 3, 6], is_decreasing=True) == -1
    assert slater.sign_perm([1, 4, 6, 3], is_decreasing=True) == 1
    assert slater.sign_perm([1, 6, 3, 4], is_decreasing=True) == 1
    assert slater.sign_perm([1, 6, 4, 3], is_decreasing=True) == -1
    assert slater.sign_perm([3, 1, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([3, 1, 6, 4], is_decreasing=True) == 1
    assert slater.sign_perm([3, 4, 1, 6], is_decreasing=True) == 1
    assert slater.sign_perm([3, 4, 6, 1], is_decreasing=True) == -1
    assert slater.sign_perm([3, 6, 1, 4], is_decreasing=True) == -1
    assert slater.sign_perm([3, 6, 4, 1], is_decreasing=True) == 1
    assert slater.sign_perm([4, 1, 3, 6], is_decreasing=True) == 1
    assert slater.sign_perm([4, 1, 6, 3], is_decreasing=True) == -1
    assert slater.sign_perm([4, 3, 1, 6], is_decreasing=True) == -1
    assert slater.sign_perm([4, 3, 6, 1], is_decreasing=True) == 1
    assert slater.sign_perm([4, 6, 1, 3], is_decreasing=True) == 1
    assert slater.sign_perm([4, 6, 3, 1], is_decreasing=True) == -1
    assert slater.sign_perm([6, 1, 3, 4], is_decreasing=True) == -1
    assert slater.sign_perm([6, 1, 4, 3], is_decreasing=True) == 1
    assert slater.sign_perm([6, 3, 1, 4], is_decreasing=True) == 1
    assert slater.sign_perm([6, 3, 4, 1], is_decreasing=True) == -1
    assert slater.sign_perm([6, 4, 1, 3], is_decreasing=True) == -1
    assert slater.sign_perm([6, 4, 3, 1], is_decreasing=True) == 1

    assert slater.sign_perm([1, 3, 6, 4], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([1, 4, 3, 6], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([1, 4, 6, 3], [1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([1, 6, 3, 4], [1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([1, 6, 4, 3], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([3, 1, 4, 6], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([3, 1, 6, 4], [1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([3, 4, 1, 6], [1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([3, 4, 6, 1], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([3, 6, 1, 4], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([3, 6, 4, 1], [1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([4, 1, 3, 6], [1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([4, 1, 6, 3], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([4, 3, 1, 6], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([4, 3, 6, 1], [1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([4, 6, 1, 3], [1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([4, 6, 3, 1], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([6, 1, 3, 4], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([6, 1, 4, 3], [1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([6, 3, 1, 4], [1, 3, 4, 6], is_decreasing=True) == 1
    assert slater.sign_perm([6, 3, 4, 1], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([6, 4, 1, 3], [1, 3, 4, 6], is_decreasing=True) == -1
    assert slater.sign_perm([6, 4, 3, 1], [1, 3, 4, 6], is_decreasing=True) == 1

    assert_raises(ValueError, slater.sign_perm, [1, 3, 6, 4], [1, 3, 6, 4], False)
    assert_raises(ValueError, slater.sign_perm, [1, 3, 6, 4], [1, 3, 6, 4], True)


def test_sign_swap():
    """Test slater.sign_swap."""
    assert_raises(ValueError, slater.sign_swap, None, 0, 1)
    assert_raises(ValueError, slater.sign_swap, 0b01, -1, 1)
    assert_raises(ValueError, slater.sign_swap, 0b01, 0, -1)
    assert_raises(ValueError, slater.sign_swap, 0b00, 0, 1)

    assert slater.sign_swap(0b000001, 0, 1) == 1
    assert slater.sign_swap(0b000001, 0, 2) == 1
    assert slater.sign_swap(0b000011, 0, 2) == -1
    assert slater.sign_swap(0b000001, 0, 3) == 1
    assert slater.sign_swap(0b000011, 0, 3) == -1
    assert slater.sign_swap(0b000101, 0, 3) == -1
    assert slater.sign_swap(0b000111, 0, 3) == 1
    assert slater.sign_swap(0b000010, 1, 0) == 1
    assert slater.sign_swap(0b000100, 2, 0) == 1
    assert slater.sign_swap(0b000110, 2, 0) == -1
    assert slater.sign_swap(0b001000, 3, 0) == 1
    assert slater.sign_swap(0b001010, 3, 0) == -1
    assert slater.sign_swap(0b001100, 3, 0) == -1
    assert slater.sign_swap(0b001110, 3, 0) == 1

    assert slater.sign_swap(0b000011, 0, 1) == -1
    assert slater.sign_swap(0b000101, 0, 2) == -1
    assert slater.sign_swap(0b000111, 0, 2) == 1
    assert slater.sign_swap(0b000011, 1, 0) == -1
    assert slater.sign_swap(0b000101, 2, 0) == -1
    assert slater.sign_swap(0b000111, 2, 0) == 1


def test_sign_excite():
    """Test slater.sign_excite."""
    assert slater.sign_excite(0b000001, [0], [1]) == 1
    assert slater.sign_excite(0b000001, [0], [2]) == 1
    assert slater.sign_excite(0b000011, [0], [2]) == -1
    assert slater.sign_excite(0b000001, [0], [3]) == 1
    assert slater.sign_excite(0b000011, [0], [3]) == -1
    assert slater.sign_excite(0b000101, [0], [3]) == -1
    assert slater.sign_excite(0b000111, [0], [3]) == 1
    assert slater.sign_excite(0b000010, [1], [0]) == 1
    assert slater.sign_excite(0b000100, [2], [0]) == 1
    assert slater.sign_excite(0b000110, [2], [0]) == -1
    assert slater.sign_excite(0b001000, [3], [0]) == 1
    assert slater.sign_excite(0b001010, [3], [0]) == -1
    assert slater.sign_excite(0b001100, [3], [0]) == -1
    assert slater.sign_excite(0b001110, [3], [0]) == 1

    assert slater.sign_excite(0b0011, [0], [2, 3]) == -1
    assert slater.sign_excite(0b0011, [0], [3, 2]) == 1
    assert slater.sign_excite(0b0011, [0, 1], [2]) == 1
    assert slater.sign_excite(0b0011, [1, 0], [2]) == -1
    assert slater.sign_excite(0b0011, [0, 1], [1, 2]) == -1
    assert slater.sign_excite(0b0011, [0, 1], [2, 1]) == 1
    assert slater.sign_excite(0b0011, [0, 1, 1], [2, 3]) is None
    assert slater.sign_excite(0b0011, [0, 1], [2, 2, 3]) is None
    assert slater.sign_excite(0b0011, [0, 2], [3]) is None
    assert slater.sign_excite(0b0011, [0], [1]) is None
