""" Tests wfns.proj.proj_hamiltonian
"""
import numpy as np
from nose.tools import assert_raises
from wfns.proj.proj_wavefunction import ProjectionWavefunction
from wfns.proj.proj_hamiltonian import hamiltonian, sen0_hamiltonian

# FIXME: ProjectionWavefunction will end up shedding a lot of its abstract methods and properties
class TestProjectionWavefunction(ProjectionWavefunction):
    """ Child of ProjWavefunction used to test ProjWavefunction

    Because ProjWavefunction is an abstract class
    """
    def __init__(self):
        # over write
        pass

    @property
    def template_coeffs(self):
        pass

    def compute_pspace(self, num_sd):
        pass

    def compute_hamiltonian(self, sd, deriv=None):
        pass

    def normalize(self):
        pass

    def compute_overlap(self, sd, deriv=None):
        if sd == 0b0101:
            return 1
        elif sd == 0b1010:
            return 2
        elif sd == 0b1100:
            return 3
        return 0


def test_hamiltonian():
    """ Tests wfns.proj.proj_hamiltonian.hamiltonian
    """
    test_wfn = TestProjectionWavefunction()
    test_wfn.one_int = (np.array([[1, 2], [3, 4]], dtype=float), )
    test_wfn.two_int = (np.array([[[[5, 6], [7, 8]],
                                   [[9, 10], [11, 12]]],
                                  [[[13, 14], [15, 16]],
                                   [[17, 18], [19, 20]]]], dtype=float), )
    test_wfn.orbtype = 'restricted'
    test_wfn.cache = {}
    test_wfn.d_cache = {}
    one_energy, coulomb, exchange = hamiltonian(test_wfn, 0b0101, 'restricted', deriv=None)
    assert one_energy == 1*1 + 1*1
    assert coulomb == 1*5 + 2*8
    assert exchange == 0

    one_energy, coulomb, exchange = hamiltonian(test_wfn, 0b1010, 'restricted', deriv=None)
    assert one_energy == 2*4 + 2*4
    assert coulomb == 1*17 + 2*20
    assert exchange == 0

    one_energy, coulomb, exchange = hamiltonian(test_wfn, 0b0110, 'restricted', deriv=None)
    assert one_energy == 1*3 + 2*2
    assert coulomb == 1*13 + 2*12
    assert exchange == 0

    one_energy, coulomb, exchange = hamiltonian(test_wfn, 0b1100, 'restricted', deriv=None)
    assert one_energy == 1*3 + 3*4
    assert coulomb == 3*10
    assert exchange == -3*11

def test_sen0_hamiltonian_2e():
    """ Tests wfns.proj.sen0_proj_hamiltonian.hamiltonian
    """
    test_wfn = TestProjectionWavefunction()
    test_wfn.one_int = (np.array([[1, 2], [3, 4]], dtype=float), )
    test_wfn.two_int = (np.array([[[[5, 6], [7, 8]],
                                   [[9, 10], [11, 12]]],
                                  [[[13, 14], [15, 16]],
                                   [[17, 18], [19, 20]]]], dtype=float), )
    test_wfn.orbtype = 'restricted'
    test_wfn.cache = {}
    test_wfn.d_cache = {}
    # check error
    assert_raises(ValueError, lambda: sen0_hamiltonian(test_wfn, 0b0001, 'restricted', deriv=None))
    assert_raises(ValueError, lambda: sen0_hamiltonian(test_wfn, 0b0011, 'restricted', deriv=None))
    assert_raises(TypeError, lambda: sen0_hamiltonian(test_wfn, 0b0101, 'generalized', deriv=None))

    one_energy, coulomb, exchange = sen0_hamiltonian(test_wfn, 0b0101, 'restricted', deriv=None)
    assert one_energy == 1*1 + 1*1
    assert coulomb == 1*5 + 2*8
    assert exchange == 0

    one_energy, coulomb, exchange = sen0_hamiltonian(test_wfn, 0b1010, 'restricted', deriv=None)
    assert one_energy == 2*4 + 2*4
    assert coulomb == 1*17 + 2*20
    assert exchange == 0

def test_sen0_hamiltonian_4e():
    """ Tests wfns.proj.sen0_proj_hamiltonian.hamiltonian
    """
    test_wfn = TestProjectionWavefunction()
    test_wfn.one_int = (np.array([[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]], dtype=float), )
    test_wfn.two_int = (np.array(np.arange(1,82).reshape(3, 3, 3, 3), dtype=float), )
    test_wfn.orbtype = 'restricted'
    test_wfn.cache = {}
    test_wfn.d_cache = {}
    # FIXME: ugly
    # overwrite the overlap function with 4 electron one
    def overlap(sd, deriv=None):
        if sd == 0b011011:
            return 1
        elif sd == 0b101101:
            return 2
        elif sd == 0b110110:
            return 3
        return 0
    test_wfn.overlap = overlap

    # check error
    assert_raises(ValueError, lambda: sen0_hamiltonian(test_wfn, 0b000001, 'restricted',
                                                       deriv=None))
    assert_raises(ValueError, lambda: sen0_hamiltonian(test_wfn, 0b001111, 'restricted',
                                                       deriv=None))
    assert_raises(TypeError, lambda: sen0_hamiltonian(test_wfn, 0b011011, 'generalized',
                                                      deriv=None))

    one_energy, coulomb, exchange = sen0_hamiltonian(test_wfn, 0b011011, 'restricted', deriv=None)
    assert one_energy == 2*1*1 + 2*1*5
    assert coulomb == 1*1 + 4*1*11 + 1*41 + 3*9 + 2*45
    assert exchange == -2*1*13

    one_energy, coulomb, exchange = sen0_hamiltonian(test_wfn, 0b110110, 'restricted', deriv=None)
    assert one_energy == 2*3*5 + 2*3*9
    assert coulomb == 3*41 + 4*3*51 + 3*81 + 2*37 +1*73
    assert exchange == -2*3*53
