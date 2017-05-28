"""Test wfns.wavefunction.wavefunctions.
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.wavefunction.base_wavefunction import BaseWavefunction


class TestWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abstract class structure."""
    def __init__(self):
        pass

    def get_overlap(self):
        pass

    @property
    def spin(self):
        return None

    @property
    def seniority(self):
        return None

    @property
    def template_params(self):
        return np.identity(10)


def test_assign_nelec():
    """Test BaseWavefunction.assign_nelec."""
    test = TestWavefunction()
    # check errors
    assert_raises(TypeError, test.assign_nelec, None)
    assert_raises(TypeError, test.assign_nelec, 2.0)
    assert_raises(TypeError, test.assign_nelec, '2')
    assert_raises(ValueError, test.assign_nelec, 0)
    assert_raises(ValueError, test.assign_nelec, -2)
    # int
    test.assign_nelec(2)
    assert test.nelec == 2


def test_nspin():
    """Test BaseWavefunction.nspin."""
    test = TestWavefunction()
    # check errors
    assert_raises(TypeError, test.assign_nspin, None)
    assert_raises(TypeError, test.assign_nspin, 2.0)
    assert_raises(TypeError, test.assign_nspin, '2')
    assert_raises(ValueError, test.assign_nspin, 0)
    assert_raises(ValueError, test.assign_nspin, -2)
    assert_raises(NotImplementedError, test.assign_nspin, 3)
    # int
    test.assign_nspin(2)
    assert test.nspin == 2


def test_assign_dtype():
    """Test BaseWavefunction.assign_dtype."""
    test = TestWavefunction()
    # check errors
    assert_raises(TypeError, test.assign_dtype, '')
    assert_raises(TypeError, test.assign_dtype, 'float64')
    assert_raises(TypeError, test.assign_dtype, int)
    assert_raises(TypeError, test.assign_dtype, np.float32)
    # None assigned
    test.assign_dtype(None)
    assert test.dtype == np.float64
    # other assignments
    for dtype in [float, np.float64]:
        test.assign_dtype(dtype)
        assert test.dtype == np.float64
    for dtype in [complex, np.complex128]:
        test.assign_dtype(dtype)
        assert test.dtype == np.complex128


def test_assign_memory():
    """Test BaseWavefunction.assign_memory."""
    test = TestWavefunction()
    test.assign_memory(None)
    assert test.memory == np.inf
    test.assign_memory(10)
    assert test.memory == 10
    test.assign_memory(20.0)
    assert test.memory == 20
    test.assign_memory('10mb')
    assert test.memory == 1e6 * 10
    test.assign_memory('20.1gb')
    assert test.memory == 1e9 * 20.1


def test_assign_params():
    """Test BaseWavefunction.assign_params."""
    # default
    test = TestWavefunction()
    test.assign_dtype(float)
    test.assign_params()
    assert np.allclose(test.params, np.identity(10))
    assert test.dtype == test.params.dtype

    test = TestWavefunction()
    test.assign_dtype(complex)
    test.assign_params()
    assert np.allclose(test.params, np.identity(10))
    assert test.dtype == test.params.dtype

    # check errors
    test = TestWavefunction()
    test.assign_dtype(float)
    assert_raises(TypeError, test.assign_params, 2)
    assert_raises(TypeError, test.assign_params, [2, 3])
    assert_raises(ValueError, test.assign_params, np.random.rand(10, 11))
    assert_raises(TypeError, test.assign_params, np.arange(100, dtype=int).reshape(10, 10))
    assert_raises(TypeError, test.assign_params, np.arange(100, dtype=complex).reshape(10, 10))

    # check noise
    test = TestWavefunction()
    test.assign_dtype(float)
    test.assign_params(add_noise=True)
    assert np.all(np.abs(test.params - np.identity(10)) <= 0.2/100)
    assert not np.allclose(test.params, np.identity(10))

    test = TestWavefunction()
    test.assign_dtype(complex)
    test.assign_params(add_noise=True)
    assert np.all(np.abs(np.real(test.params - np.identity(10))) <= 0.1/100)
    assert np.all(np.abs(np.imag(test.params - np.identity(10))) <= 0.01*0.1/100)
    assert not np.allclose(np.real(test.params), np.identity(10))
    assert not np.allclose(np.imag(test.params), np.zeros((10, 10)))


def test_nspatial():
    """Test BaseWavefunction.nspatial."""
    test = TestWavefunction()
    test.assign_nspin(10)
    assert test.nspatial == 5


def test_nparams():
    """Test BaseWavefunction.nparams."""
    test = TestWavefunction()
    assert test.nparams == 100


def test_params_shape():
    """Test BaseWavefunction.params_shape."""
    test = TestWavefunction()
    assert test.params_shape == (10, 10)
