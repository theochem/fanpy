import sys
sys.path.append('../')
import sd_list
from common import raises_exception
import gmpy2

# A test wavefunction (must have nspatial, nelec, npair)
class TestWavefunction:
    nspatial = 3
    nelec = 4
    npair = 2
test = TestWavefunction()

def test_ci_sd_list():
    """ Tests sd_list.ci_sd_list
    """
    assert raises_exception(lambda:sd_list.ci_sd_list(test, 0, [0]))
    assert raises_exception(lambda:sd_list.ci_sd_list(test, 0, [-1]))
    assert sd_list.ci_sd_list(test, 0, []) == []
    assert sd_list.ci_sd_list(test, 1, []) == [gmpy2.mpz(0b011011)]
    assert sd_list.ci_sd_list(test, 2, []) == [gmpy2.mpz(0b011011), gmpy2.mpz(0b011101)]
    assert sd_list.ci_sd_list(test, 10, [1]) == [gmpy2.mpz(0b011011),
                                                 gmpy2.mpz(0b011101),
                                                 gmpy2.mpz(0b111001),
                                                 gmpy2.mpz(0b001111),
                                                 gmpy2.mpz(0b101011),
                                                 gmpy2.mpz(0b011110),
                                                 gmpy2.mpz(0b111010),
                                                 gmpy2.mpz(0b010111),
                                                 gmpy2.mpz(0b110011),]
    assert sd_list.ci_sd_list(test, 10, [2]) == [gmpy2.mpz(0b011011),
                                                 gmpy2.mpz(0b101101),
                                                 gmpy2.mpz(0b111100),
                                                 gmpy2.mpz(0b110101),
                                                 gmpy2.mpz(0b101110),
                                                 gmpy2.mpz(0b100111),
                                                 gmpy2.mpz(0b110110),]
    assert sd_list.ci_sd_list(test, 999, [1, 2]) == [gmpy2.mpz(0b011011),
                                                     gmpy2.mpz(0b011101),
                                                     gmpy2.mpz(0b111001),
                                                     gmpy2.mpz(0b001111),
                                                     gmpy2.mpz(0b101011),
                                                     gmpy2.mpz(0b011110),
                                                     gmpy2.mpz(0b111010),
                                                     gmpy2.mpz(0b010111),
                                                     gmpy2.mpz(0b110011),
                                                     gmpy2.mpz(0b101101),
                                                     gmpy2.mpz(0b111100),
                                                     gmpy2.mpz(0b110101),
                                                     gmpy2.mpz(0b101110),
                                                     gmpy2.mpz(0b100111),
                                                     gmpy2.mpz(0b110110),]
    assert sd_list.ci_sd_list(test, 999, [2, 1]) == [gmpy2.mpz(0b011011),
                                                     gmpy2.mpz(0b101101),
                                                     gmpy2.mpz(0b111100),
                                                     gmpy2.mpz(0b110101),
                                                     gmpy2.mpz(0b101110),
                                                     gmpy2.mpz(0b100111),
                                                     gmpy2.mpz(0b110110),
                                                     gmpy2.mpz(0b011101),
                                                     gmpy2.mpz(0b111001),
                                                     gmpy2.mpz(0b001111),
                                                     gmpy2.mpz(0b101011),
                                                     gmpy2.mpz(0b011110),
                                                     gmpy2.mpz(0b111010),
                                                     gmpy2.mpz(0b010111),
                                                     gmpy2.mpz(0b110011),]

test.nspatial = 4
def test_doci_sd_list():
    """ Tests sd_list.doci_sd_list
    """
    assert raises_exception(lambda:sd_list.doci_sd_list(test, 0, [0]))
    assert raises_exception(lambda:sd_list.doci_sd_list(test, 0, [-1]))
    assert sd_list.doci_sd_list(test, 0, []) == []
    assert sd_list.doci_sd_list(test, 1, []) == [gmpy2.mpz(0b00110011)]
    assert sd_list.doci_sd_list(test, 2, []) == [gmpy2.mpz(0b00110011), gmpy2.mpz(0b01010101)]
    assert sd_list.doci_sd_list(test, 10, [1]) == [gmpy2.mpz(0b00110011),
                                                   gmpy2.mpz(0b01010101),
                                                   gmpy2.mpz(0b10011001),
                                                   gmpy2.mpz(0b01100110),
                                                   gmpy2.mpz(0b10101010),]
    assert sd_list.doci_sd_list(test, 10, [2]) == [gmpy2.mpz(0b00110011),
                                                   gmpy2.mpz(0b11001100),]
    assert sd_list.doci_sd_list(test, 10, [1,2]) == [gmpy2.mpz(0b00110011),
                                                     gmpy2.mpz(0b01010101),
                                                     gmpy2.mpz(0b10011001),
                                                     gmpy2.mpz(0b01100110),
                                                     gmpy2.mpz(0b10101010),
                                                     gmpy2.mpz(0b11001100),]
    assert sd_list.doci_sd_list(test, 10, [2,1]) == [gmpy2.mpz(0b00110011),
                                                     gmpy2.mpz(0b11001100),
                                                     gmpy2.mpz(0b01010101),
                                                     gmpy2.mpz(0b10011001),
                                                     gmpy2.mpz(0b01100110),
                                                     gmpy2.mpz(0b10101010)]
