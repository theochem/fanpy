"""Test wfns.sd_list."""
import pytest
import gmpy2
from wfns.backend import sd_list


def test_satisfies_conditions():
    """Test sd_list.satisfies_conditions."""
    assert sd_list.satisfies_conditions(0b1111, 2, None, None)

    assert not sd_list.satisfies_conditions(0b0001, 2, -1, None)
    assert not sd_list.satisfies_conditions(0b0001, 2, -0.5, None)
    assert not sd_list.satisfies_conditions(0b0001, 2, 0, None)
    assert sd_list.satisfies_conditions(0b0001, 2, 0.5, None)
    assert not sd_list.satisfies_conditions(0b0001, 2, 1, None)

    assert sd_list.satisfies_conditions(0b1100, 2, -1, None)
    assert not sd_list.satisfies_conditions(0b1100, 2, -0.5, None)
    assert not sd_list.satisfies_conditions(0b1100, 2, 0, None)
    assert not sd_list.satisfies_conditions(0b1100, 2, 0.5, None)
    assert not sd_list.satisfies_conditions(0b1100, 2, 1, None)

    assert sd_list.satisfies_conditions(0b1110, 2, None, 2)
    assert sd_list.satisfies_conditions(0b1110, 2, None, 1)
    assert not sd_list.satisfies_conditions(0b1110, 2, None, 0)

    assert sd_list.satisfies_conditions(0b1110, 2, -0.5, 1)
    assert not sd_list.satisfies_conditions(0b1110, 2, 0.5, 1)
    assert not sd_list.satisfies_conditions(0b1110, 2, -0.5, 0)


def test_ci_sd_list():
    """Test sd_list.sd_list."""
    with pytest.raises(TypeError):
        sd_list.sd_list(4.0, 3)
    with pytest.raises(TypeError):
        sd_list.sd_list(None, 3)
    with pytest.raises(TypeError):
        sd_list.sd_list('4', 3)
    with pytest.raises(TypeError):
        sd_list.sd_list(3, 4.0)
    with pytest.raises(TypeError):
        sd_list.sd_list(3, None)
    with pytest.raises(TypeError):
        sd_list.sd_list(3, '4')
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 3, num_limit=3.0)
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 3, num_limit='3')
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 3, exc_orders=3)
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 3, exc_orders=[3.0])
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 3, spin='3')
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 3, seniority='3')
    with pytest.raises(ValueError):
        sd_list.sd_list(4, 3, spin=0.5, seniority=0)
    with pytest.raises(ValueError):
        sd_list.sd_list(4, 3, spin=-1, seniority=1)

    assert sd_list.sd_list(4, 3, num_limit=0) == []
    assert sd_list.sd_list(4, 3, num_limit=1) == [gmpy2.mpz(0b011011)]
    assert sd_list.sd_list(4, 3, num_limit=2) == [gmpy2.mpz(0b011011), gmpy2.mpz(0b011101)]
    assert sd_list.sd_list(4, 3, num_limit=10, exc_orders=[1]) == [gmpy2.mpz(0b011011),
                                                                   gmpy2.mpz(0b011101),
                                                                   gmpy2.mpz(0b111001),
                                                                   gmpy2.mpz(0b001111),
                                                                   gmpy2.mpz(0b101011),
                                                                   gmpy2.mpz(0b011110),
                                                                   gmpy2.mpz(0b111010),
                                                                   gmpy2.mpz(0b010111),
                                                                   gmpy2.mpz(0b110011)]
    assert sd_list.sd_list(4, 3, num_limit=10, exc_orders=[2]) == [gmpy2.mpz(0b011011),
                                                                   gmpy2.mpz(0b101101),
                                                                   gmpy2.mpz(0b111100),
                                                                   gmpy2.mpz(0b110101),
                                                                   gmpy2.mpz(0b101110),
                                                                   gmpy2.mpz(0b100111),
                                                                   gmpy2.mpz(0b110110)]
    assert sd_list.sd_list(4, 3, exc_orders=[1, 2]) == [gmpy2.mpz(0b011011), gmpy2.mpz(0b011101),
                                                        gmpy2.mpz(0b111001), gmpy2.mpz(0b001111),
                                                        gmpy2.mpz(0b101011), gmpy2.mpz(0b011110),
                                                        gmpy2.mpz(0b111010), gmpy2.mpz(0b010111),
                                                        gmpy2.mpz(0b110011), gmpy2.mpz(0b101101),
                                                        gmpy2.mpz(0b111100), gmpy2.mpz(0b110101),
                                                        gmpy2.mpz(0b101110), gmpy2.mpz(0b100111),
                                                        gmpy2.mpz(0b110110)]
    assert sd_list.sd_list(4, 3, exc_orders=[2, 1]) == [gmpy2.mpz(0b011011), gmpy2.mpz(0b101101),
                                                        gmpy2.mpz(0b111100), gmpy2.mpz(0b110101),
                                                        gmpy2.mpz(0b101110), gmpy2.mpz(0b100111),
                                                        gmpy2.mpz(0b110110), gmpy2.mpz(0b011101),
                                                        gmpy2.mpz(0b111001), gmpy2.mpz(0b001111),
                                                        gmpy2.mpz(0b101011), gmpy2.mpz(0b011110),
                                                        gmpy2.mpz(0b111010), gmpy2.mpz(0b010111),
                                                        gmpy2.mpz(0b110011)]
    assert sd_list.sd_list(4, 3, exc_orders=[1, 2], spin=0) == [gmpy2.mpz(0b011011),
                                                                gmpy2.mpz(0b011101),
                                                                gmpy2.mpz(0b101011),
                                                                gmpy2.mpz(0b011110),
                                                                gmpy2.mpz(0b110011),
                                                                gmpy2.mpz(0b101101),
                                                                gmpy2.mpz(0b110101),
                                                                gmpy2.mpz(0b101110),
                                                                gmpy2.mpz(0b110110)]
    assert sd_list.sd_list(4, 3, exc_orders=[1, 2], spin=0.5) == []
    assert sd_list.sd_list(4, 3, exc_orders=[1, 2], spin=-0.5) == []
    assert sd_list.sd_list(4, 3, exc_orders=[1, 2], spin=1) == [gmpy2.mpz(0b001111),
                                                                gmpy2.mpz(0b010111),
                                                                gmpy2.mpz(0b100111)]
    assert sd_list.sd_list(4, 3, exc_orders=[1, 2], spin=-1) == [gmpy2.mpz(0b111001),
                                                                 gmpy2.mpz(0b111010),
                                                                 gmpy2.mpz(0b111100)]


def test_doci_sd_list():
    """Test sd_list.doci_sd_list."""
    assert sd_list.sd_list(4, 4, num_limit=0, seniority=0) == []
    assert sd_list.sd_list(4, 4, num_limit=1, seniority=0) == [gmpy2.mpz(0b00110011)]
    assert sd_list.sd_list(4, 4, num_limit=2, seniority=0) == [gmpy2.mpz(0b00110011),
                                                               gmpy2.mpz(0b01010101)]
    assert sd_list.sd_list(4, 4, exc_orders=[2], seniority=0) == [gmpy2.mpz(0b00110011),
                                                                  gmpy2.mpz(0b01010101),
                                                                  gmpy2.mpz(0b10011001),
                                                                  gmpy2.mpz(0b01100110),
                                                                  gmpy2.mpz(0b10101010)]
    assert sd_list.sd_list(4, 4, exc_orders=[4], seniority=0) == [gmpy2.mpz(0b00110011),
                                                                  gmpy2.mpz(0b11001100)]
    assert sd_list.sd_list(4, 4, exc_orders=[2, 4], seniority=0) == [gmpy2.mpz(0b00110011),
                                                                     gmpy2.mpz(0b01010101),
                                                                     gmpy2.mpz(0b10011001),
                                                                     gmpy2.mpz(0b01100110),
                                                                     gmpy2.mpz(0b10101010),
                                                                     gmpy2.mpz(0b11001100)]
    assert sd_list.sd_list(4, 4, exc_orders=[4, 2], seniority=0) == [gmpy2.mpz(0b00110011),
                                                                     gmpy2.mpz(0b11001100),
                                                                     gmpy2.mpz(0b01010101),
                                                                     gmpy2.mpz(0b10011001),
                                                                     gmpy2.mpz(0b01100110),
                                                                     gmpy2.mpz(0b10101010)]
