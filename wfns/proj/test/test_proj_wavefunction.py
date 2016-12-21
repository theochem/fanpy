from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np

from wfns.proj.proj_wavefunction import ProjectionWavefunction


class TestProjectionWavefunction(ProjectionWavefunction):
    # overwrite to stop initialization
    def __init__(self):
        pass

    @property
    def template_coeffs(self):
        return np.empty(4)

    def compute_pspace(self, num_sd):
        return [0b1111, 0b10111, 0b11011, 0b11101][:num_sd]

    def compute_overlap(self, sd, deriv=None):
        return 3

    def compute_hamiltonian(self, sd, deriv=None):
        return 2

    def normalize(self):
        return 1.0


def test_assign_params():
    """
    Tests ProjectionWavefunction.assign_params
    """
    test = TestProjectionWavefunction()
    # Assign random float
    test.dtype = np.float64
    test.assign_params()
    assert np.all(np.abs(test.params[:-1]) < 1 / 4.0)
    # Assign random float
    test.dtype = np.complex128
    test.assign_params()
    print(np.real(test.params[:-1]))
    assert np.all(np.real(test.params[:-1]) < 1 / 4.0)
    assert np.all(np.imag(test.params[:-1]) < 1 / 4.0)
    # Check energy
    assert test.params[-1] == 0.0
    # Assign params
    test.assign_params(np.array([0, 1, 2, 3, 4], dtype=np.float64))
    assert np.allclose(test.params, np.arange(5))
    # Assign bad params
    assert_raises(TypeError, lambda: test.assign_params([1, 2, 3, 4, 5]))
    assert_raises(TypeError, lambda: test.assign_params('12345'))
    assert_raises(ValueError, lambda: test.assign_params(np.arange(6)))
    assert_raises(ValueError, lambda: test.assign_params(np.arange(5).reshape(1, 5)))
    assert_raises(TypeError, lambda: test.assign_params(np.arange(5)))


def test_assign_pspace():
    """
    Tests ProjectionWavefunction.assign_pspace
    """
    test = TestProjectionWavefunction()
    test.params = np.empty(4)
    # default assignment
    test.assign_pspace()
    assert test.pspace == (0b1111, 0b10111, 0b11011, 0b11101)[:3]
    # assign pspace
    test.assign_pspace([0b1111, 0b10111])
    assert test.pspace == (0b1111, 0b10111)
    test.assign_pspace((0b1111, 0b10111))
    assert test.pspace == (0b1111, 0b10111)
    # assign int
    test.assign_pspace(4)
    assert test.pspace == (0b1111, 0b10111, 0b11011, 0b11101)
    test.assign_pspace(3)
    assert test.pspace == (0b1111, 0b10111, 0b11011, 0b11101)[:3]
    test.assign_pspace(2)
    assert test.pspace == (0b1111, 0b10111, 0b11011, 0b11101)[:2]
    test.assign_pspace(1)
    assert test.pspace == (0b1111, 0b10111, 0b11011, 0b11101)[:1]
    # bad pspace
    assert_raises(TypeError, lambda: test.assign_pspace('123'))
    assert_raises(ValueError, lambda: test.assign_pspace(['1', '2']))


def test_overlap():
    """
    Tests ProjectionWavefunction.overlap
    """
    test = TestProjectionWavefunction()
    test.cache = {0b1111: 1, 0b10111: 2}
    # sd included in cache
    assert test.overlap(0b1111) == 1
    assert test.overlap(0b10111) == 2
    # sd not included in cache
    assert test.overlap(0b100111) == 3
    # derivative
    test.H = np.arange(9).reshape(3, 3)
    test.orbtype = 'restricted'
    test.d_cache = {(0b1111, 0): 4, (0b1111, 1): 5, (0b1111, 2): 6, (0b10111, 0): 7, (0b10111, 1): 8}
    assert test.overlap(0b1111, deriv=0) == 4
    assert test.overlap(0b1111, deriv=1) == 5
    assert test.overlap(0b1111, deriv=2) == 6
    assert test.overlap(0b10111, deriv=0) == 7
    assert test.overlap(0b10111, deriv=1) == 8


def test_compute_norm():
    """
    Tests ProjectionWavefunction.compute_norm
    """
    pass
    test = TestProjectionWavefunction()
    test.nuc_nuc = 2.0
    test.params = np.array([0, 1, 2, 3, 4])
    test.cache = {0b1111: 1.2, 0b10111: 2.1}
    test.pspace = [0b1111, 0b10111]
    # sd is none
    assert test.compute_norm() == 1.2**2
    # sd specified
    assert test.compute_norm(ref_sds=[0b1111]) == 1.2**2
    assert test.compute_norm(ref_sds=[0b10111]) == 2.1**2
    assert test.compute_norm(ref_sds=(0b1111, 0b10111)) == 1.2**2 + 2.1**2
    # bad ref_sds
    assert_raises(TypeError, lambda: test.compute_norm(ref_sds='11'))
    assert_raises(TypeError, lambda: test.compute_norm(ref_sds=('11',)))


def test_compute_energy():
    """
    Tests ProjectionWavefunction.compute_energy
    """
    test = TestProjectionWavefunction()
    test.nuc_nuc = 2.0
    test.params = np.array([0, 1, 2, 3, 4])
    test.cache = {0b1111: 1, 0b10111: 2}

    # energy is param
    assert test.compute_energy() == test.params[-1]
    assert test.compute_energy(include_nuc=False) == test.params[-1]
    assert test.compute_energy(deriv=0) == 0
    assert test.compute_energy(deriv=4) == 1
