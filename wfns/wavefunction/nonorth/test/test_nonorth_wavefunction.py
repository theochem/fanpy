"""Test wfns.wavefunction.nonorth.nonorth_wavefunction"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.wavefunction.base_wavefunction import BaseWavefunction
from wfns.wavefunction.nonorth.nonorth_wavefunction import NonorthWavefunction
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction


class TestNonorthWavefunction(NonorthWavefunction):
    """Class to test NonorthWavefunction."""
    def __init__(self):
        pass


class TestWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abstract class structure."""
    _spin = None
    _seniority = None

    def __init__(self):
        pass

    def get_overlap(self):
        pass

    @property
    def spin(self):
        return self._spin

    @property
    def seniority(self):
        return self._seniority

    @property
    def template_params(self):
        return np.identity(10)


def test_nonorth_assign_wfn():
    """Tests NonorthWavefunction.assign_wfn."""
    test = TestNonorthWavefunction()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.memory = 10

    # check default
    test.assign_wfn(None)
    assert isinstance(test.wfn, CIWavefunction)
    assert test.wfn.nelec == test.nelec
    assert test.wfn.nspin == test.nspin
    assert test.wfn.dtype == test.dtype
    assert test.wfn.memory == test.memory

    # check type
    assert_raises(ValueError, test.assign_wfn, int)
    test_wfn = TestWavefunction()
    test_wfn.nelec = 3
    assert_raises(ValueError, test.assign_wfn, test_wfn)
    test_wfn.nelec = 4
    test_wfn.dtype = np.complex128
    assert_raises(ValueError, test.assign_wfn, test_wfn)
    test_wfn.dtype = np.float64
    test_wfn.memory = 5
    assert_raises(ValueError, test.assign_wfn, test_wfn)


def test_nonorth_assign_params():
    """Tests NonorthWavefunction.assign_params."""
    test = TestNonorthWavefunction()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.memory = 10

    test_wfn = TestWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.dtype = np.float64
    test_wfn.memory = 10
    test_wfn._spin = 2
    test.assign_wfn(test_wfn)

    test.assign_params(None)
    assert isinstance(test.params, tuple)
    assert len(test.params) == 1
    assert np.allclose(test.params[0], np.eye(5, 6, dtype=float))

    test_params = np.random.rand(5, 6)
    assert_raises(TypeError, test.assign_params, (test_params, )*3)
    assert_raises(TypeError, test.assign_params, [])
    assert_raises(TypeError, test.assign_params, {0: test_params})
    assert_raises(TypeError, test.assign_params, [test_params.tolist()])
    assert_raises(TypeError, test.assign_params, [np.random.rand(5, 6, 1)])
    assert_raises(TypeError, test.assign_params, [test_params.astype(complex)])
    assert_raises(ValueError, test.assign_params, [np.random.rand(6, 5)])
    assert_raises(ValueError, test.assign_params, [np.random.rand(9, 9)])
    assert_raises(ValueError, test.assign_params, [np.random.rand(5, 6), np.random.rand(5, 5)])
    assert_raises(ValueError, test.assign_params, [np.random.rand(5, 5), np.random.rand(5, 6)])

    test.assign_params(test_params)
    assert isinstance(test.params, tuple)
    assert len(test.params) == 1
    assert np.allclose(test.params[0], test_params)
    test_params_alpha = np.random.rand(5, 6)
    test_params_beta = np.random.rand(5, 6)
    test.assign_params([test_params_alpha, test_params_beta])
    assert isinstance(test.params, tuple)
    assert len(test.params) == 2
    assert np.allclose(test.params[0], test_params_alpha)
    assert np.allclose(test.params[1], test_params_beta)

    test_params = np.random.rand(10, 12)
    test.assign_params(test_params)
    assert isinstance(test.params, tuple)
    assert len(test.params) == 1
    assert np.allclose(test.params[0], test_params)


def test_nonorth_spin():
    """Test NonorthWavefunction.spin"""
    test = TestNonorthWavefunction()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.memory = 10

    test_wfn = TestWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.dtype = np.float64
    test_wfn.memory = 10
    test_wfn._spin = 2
    test.assign_wfn(test_wfn)

    # restricted
    test.assign_params(np.random.rand(5, 6))
    assert test.spin == 2
    # restricted
    test.assign_params([np.random.rand(5, 6), np.random.rand(5, 6)])
    assert test.spin == 2
    # generalized
    test.assign_params(np.random.rand(10, 12))
    assert test.spin is None


def test_nonorth_seniority():
    """Test NonorthWavefunction.seniority"""
    test = TestNonorthWavefunction()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.memory = 10

    test_wfn = TestWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.dtype = np.float64
    test_wfn.memory = 10
    test_wfn._seniority = 2
    test.assign_wfn(test_wfn)

    # restricted
    test.assign_params(np.random.rand(5, 6))
    assert test.seniority == 2
    # restricted
    test.assign_params([np.random.rand(5, 6), np.random.rand(5, 6)])
    assert test.seniority == 2
    # generalized
    test.assign_params(np.random.rand(10, 12))
    assert test.seniority is None


def test_nonorth_template_params():
    """Test NonorthWavefunction.template_params"""
    test = TestNonorthWavefunction()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.memory = 10

    test_wfn = TestWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.dtype = np.float64
    test_wfn.memory = 10
    test.assign_wfn(test_wfn)

    assert isinstance(test.template_params, tuple)
    assert len(test.template_params) == 1
    assert np.allclose(test.template_params[0], np.eye(5, 6))


def test_nonorth_nparams():
    """Test NonorthWavefunction.nparams"""
    test = TestNonorthWavefunction()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.memory = 10

    test_wfn = TestWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.dtype = np.float64
    test_wfn.memory = 10
    test.assign_wfn(test_wfn)

    # restricted
    test.assign_params(np.random.rand(5, 6))
    assert test.nparams == (30, )
    # unrestricted
    test.assign_params([np.random.rand(5, 6), np.random.rand(5, 6)])
    assert test.nparams == (30, 30)
    # generalized
    test.assign_params(np.random.rand(10, 12))
    assert test.nparams == (120, )


def test_nonorth_params_shape():
    """Test NonorthWavefunction.params_shape"""
    test = TestNonorthWavefunction()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.memory = 10

    test_wfn = TestWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.dtype = np.float64
    test_wfn.memory = 10
    test.assign_wfn(test_wfn)

    # restricted
    test.assign_params(np.random.rand(5, 6))
    assert test.params_shape == ((5, 6), )
    # unrestricted
    test.assign_params([np.random.rand(5, 6), np.random.rand(5, 6)])
    assert test.params_shape == ((5, 6), (5, 6))
    # generalized
    test.assign_params(np.random.rand(10, 12))
    assert test.params_shape == ((10, 12), )


def test_nonorth_orbtype():
    """Test NonorthWavefunction.orbtype"""
    test = TestNonorthWavefunction()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.memory = 10

    test_wfn = TestWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.dtype = np.float64
    test_wfn.memory = 10
    test.assign_wfn(test_wfn)

    # restricted
    test.assign_params(np.random.rand(5, 6))
    assert test.orbtype == 'restricted'
    # unrestricted
    test.assign_params([np.random.rand(5, 6), np.random.rand(5, 6)])
    assert test.orbtype == 'unrestricted'
    # generalized
    test.assign_params(np.random.rand(10, 12))
    assert test.orbtype == 'generalized'
