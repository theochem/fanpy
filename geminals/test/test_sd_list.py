from nose.tools import assert_raises
import gmpy2

from .. import sd_list


def test_generate_ci_sd_list():
    """
    Tests sd_list.generate_ci_sd_list
    """
    # assert assert_raises(AssertionError, lambda:sd_list.ci_sd_list(test, 0, [0]))
    # assert assert_raises(AssertionError, lambda:sd_list.ci_sd_list(test, 0, [-1]))
    # ground state determinant
    ground = [gmpy2.mpz(0b011011)]
    # single excitations
    singles = [gmpy2.mpz(0b011101), gmpy2.mpz(0b111001), gmpy2.mpz(0b001111),
               gmpy2.mpz(0b101011), gmpy2.mpz(0b011110), gmpy2.mpz(0b111010),
               gmpy2.mpz(0b010111), gmpy2.mpz(0b110011)]
    # double excitations
    doubles = [gmpy2.mpz(0b101101), gmpy2.mpz(0b111100), gmpy2.mpz(0b110101),
               gmpy2.mpz(0b101110), gmpy2.mpz(0b100111), gmpy2.mpz(0b110110)]
    # Check list of CI determinants
    assert sd_list.generate_ci_sd_list(3, 4, 0, []) == []
    assert sd_list.generate_ci_sd_list(3, 4, 1, []) == ground
    assert sd_list.generate_ci_sd_list(3, 4, 2, []) == ground + [singles[0]]
    assert sd_list.generate_ci_sd_list(3, 4, 10, [1]) == ground + singles
    assert sd_list.generate_ci_sd_list(3, 4, 10, [2]) == ground + doubles
    assert sd_list.generate_ci_sd_list(3, 4, 999, [1, 2]) == ground + singles + doubles
    assert sd_list.generate_ci_sd_list(3, 4, 999, [2, 1]) == ground + doubles + singles


def test_generate_doci_sd_list():
    """
    Tests sd_list.generate_doci_sd_list
    """
    # assert assert_raises(AssertionError, lambda:sd_list.doci_sd_list(test, 0, [0]))
    # assert assert_raises(AssertionError, lambda:sd_list.doci_sd_list(test, 0, [-1]))
    assert sd_list.generate_doci_sd_list(4, 4, 2, 0, []) == []
    # ground state determinant
    ground = [gmpy2.mpz(0b00110011)]
    # single pair excitations
    singles = [gmpy2.mpz(0b01010101), gmpy2.mpz(0b10011001),
               gmpy2.mpz(0b01100110), gmpy2.mpz(0b10101010)]
    # double pair excitations
    doubles = [gmpy2.mpz(0b11001100)]
    # Check single pair excitations
    assert sd_list.generate_doci_sd_list(4, 4, 2, 1, []) == ground
    assert sd_list.generate_doci_sd_list(4, 4, 2, 2, []) == ground + [singles[0]]
    assert sd_list.generate_doci_sd_list(4, 4, 2, 10, [1]) == ground + singles
    # Check single and double pair excitations
    assert sd_list.generate_doci_sd_list(4, 4, 2, 10, [2]) == ground + doubles
    assert sd_list.generate_doci_sd_list(4, 4, 2, 10, [1, 2]) == ground + singles + doubles
    assert sd_list.generate_doci_sd_list(4, 4, 2, 10, [2, 1]) == ground + doubles + singles
