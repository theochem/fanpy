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
    # else
    test.params = np.random.rand(10, 12)
    assert_raises(NotImplementedError, lambda: test.orbtype)


def test_nonorth_get_overlap():
    """Test NonorthWavefunction.get_overap."""
    test = TestNonorthWavefunction()
    test.nelec = 2
    test.nspin = 4
    test.dtype = np.float64
    test.memory = 10
    test.assign_wfn(None)
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}

    # generalized
    test.params = [np.arange(1, 17).reshape(4, 4)]
    # 0b0101 uses [[1, 2, 3, 4],
    #              [9, 10, 11, 12]]
    assert np.isclose(test.get_overlap(0b0101), ((1*10 - 2*9) * wfn_sd_coeff[0b0011] +
                                                 (1*11 - 3*9) * wfn_sd_coeff[0b0101] +
                                                 (1*12 - 4*9) * wfn_sd_coeff[0b1001] +
                                                 (2*11 - 3*10) * wfn_sd_coeff[0b0110] +
                                                 (2*12 - 4*10) * wfn_sd_coeff[0b1010] +
                                                 (3*12 - 4*11) * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)
    # 0b0110 uses [[5, 6, 7, 8],
    #              [9, 10, 11, 12]]
    assert np.isclose(test.get_overlap(0b0110), ((5*10 - 6*9) * wfn_sd_coeff[0b0011] +
                                                 (5*11 - 7*9) * wfn_sd_coeff[0b0101] +
                                                 (5*12 - 8*9) * wfn_sd_coeff[0b1001] +
                                                 (6*11 - 7*10) * wfn_sd_coeff[0b0110] +
                                                 (6*12 - 8*10) * wfn_sd_coeff[0b1010] +
                                                 (7*12 - 8*11) * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)
    # 0b1100 uses [[9, 10, 11, 12],
    #              [13, 14, 15, 16]]
    assert np.isclose(test.get_overlap(0b1100), ((9*14 - 10*13) * wfn_sd_coeff[0b0011] +
                                                 (9*15 - 11*13) * wfn_sd_coeff[0b0101] +
                                                 (9*16 - 12*13) * wfn_sd_coeff[0b1001] +
                                                 (10*15 - 11*14) * wfn_sd_coeff[0b0110] +
                                                 (10*16 - 12*14) * wfn_sd_coeff[0b1010] +
                                                 (11*16 - 12*15) * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)

    # unrestricted
    test.clear_cache()
    test.params = [np.array([[1, 2], [5, 6]]), np.array([[11, 12], [15, 16]])]
    # 0b0101 uses [[1, 2, 0, 0],
    #              [0, 0, 11, 12]]
    assert np.isclose(test.get_overlap(0b0101), ((1*0 - 2*0) * wfn_sd_coeff[0b0011] +
                                                 (1*11 - 0*0) * wfn_sd_coeff[0b0101] +
                                                 (1*12 - 0*0) * wfn_sd_coeff[0b1001] +
                                                 (2*11 - 0*0) * wfn_sd_coeff[0b0110] +
                                                 (2*12 - 0*0) * wfn_sd_coeff[0b1010] +
                                                 (0*12 - 0*11) * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)
    # 0b0110 uses [[5, 6, 0, 0],
    #              [0, 0, 11, 12]]
    assert np.isclose(test.get_overlap(0b0110), ((5*0 - 6*0) * wfn_sd_coeff[0b0011] +
                                                 (5*11 - 0*0) * wfn_sd_coeff[0b0101] +
                                                 (5*12 - 0*0) * wfn_sd_coeff[0b1001] +
                                                 (6*11 - 0*0) * wfn_sd_coeff[0b0110] +
                                                 (6*12 - 0*0) * wfn_sd_coeff[0b1010] +
                                                 (0*12 - 0*11) * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)
    # 0b1100 uses [[0, 0, 11, 12],
    #              [0, 0, 15, 16]]
    assert np.isclose(test.get_overlap(0b1100), ((0*0 - 0*0) * wfn_sd_coeff[0b0011] +
                                                 (0*15 - 11*0) * wfn_sd_coeff[0b0101] +
                                                 (0*16 - 12*0) * wfn_sd_coeff[0b1001] +
                                                 (0*15 - 11*0) * wfn_sd_coeff[0b0110] +
                                                 (0*16 - 12*0) * wfn_sd_coeff[0b1010] +
                                                 (11*16 - 12*15) * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)
    # restricted
    test.clear_cache()
    test.params = [np.array([[1, 2], [5, 6]])]
    # 0b0101 uses [[1, 2, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(test.get_overlap(0b0101), ((1*0 - 2*0) * wfn_sd_coeff[0b0011] +
                                                 (1*1 - 0*0) * wfn_sd_coeff[0b0101] +
                                                 (1*2 - 0*0) * wfn_sd_coeff[0b1001] +
                                                 (2*1 - 0*0) * wfn_sd_coeff[0b0110] +
                                                 (2*2 - 0*0) * wfn_sd_coeff[0b1010] +
                                                 (0*2 - 0*1) * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)
    # 0b0110 uses [[5, 6, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(test.get_overlap(0b0110), ((5*0 - 6*0) * wfn_sd_coeff[0b0011] +
                                                 (5*1 - 0*0) * wfn_sd_coeff[0b0101] +
                                                 (5*2 - 0*0) * wfn_sd_coeff[0b1001] +
                                                 (6*1 - 0*0) * wfn_sd_coeff[0b0110] +
                                                 (6*2 - 0*0) * wfn_sd_coeff[0b1010] +
                                                 (0*1 - 0*2) * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)
    # 0b1100 uses [[0, 0, 1, 2],
    #              [0, 0, 5, 6]]
    assert np.isclose(test.get_overlap(0b1100), ((0*0 - 0*0) * wfn_sd_coeff[0b0011] +
                                                 (0*1 - 0*5) * wfn_sd_coeff[0b0101] +
                                                 (0*2 - 0*6) * wfn_sd_coeff[0b1001] +
                                                 (0*1 - 0*5) * wfn_sd_coeff[0b0110] +
                                                 (0*2 - 0*6) * wfn_sd_coeff[0b1010] +
                                                 (1*6 - 5*2) * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)


def test_nonorth_get_overlap_deriv():
    """Test NonorthWavefunction.get_overap with derivatization option."""
    test = TestNonorthWavefunction()
    test.nelec = 2
    test.nspin = 4
    test.dtype = np.float64
    test.memory = 10
    test.assign_wfn(None)
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}

    # generalized
    test.params = [np.arange(1, 17).reshape(4, 4)]
    # 0b0101 uses [[1, 2, 3, 4],
    #              [9, 10, 11, 12]]
    assert np.isclose(test.get_overlap(0b0101, deriv=0), (10 * wfn_sd_coeff[0b0011] +
                                                          11 * wfn_sd_coeff[0b0101] +
                                                          12 * wfn_sd_coeff[0b1001]),
                      rtol=0, atol=1e-12)
    assert np.isclose(test.get_overlap(0b0101, deriv=6), 0, rtol=0, atol=1e-12)
    assert np.isclose(test.get_overlap(0b0101, deriv=8),  (-2 * wfn_sd_coeff[0b0011] +
                                                           -3 * wfn_sd_coeff[0b0101] +
                                                           -4 * wfn_sd_coeff[0b1001]),
                      rtol=0, atol=1e-12)

    # unrestricted
    test.clear_cache()
    test.params = [np.array([[1, 2], [5, 6]]), np.array([[11, 12], [15, 16]])]
    # 0b0101 uses [[1, 2, 0, 0],
    #              [0, 0, 11, 12]]
    assert np.isclose(test.get_overlap(0b0101, deriv=0), (0 * wfn_sd_coeff[0b0011] +
                                                          11 * wfn_sd_coeff[0b0101] +
                                                          12 * wfn_sd_coeff[0b1001]),
                      rtol=0, atol=1e-12)
    assert np.isclose(test.get_overlap(0b0101, deriv=5), (1 * wfn_sd_coeff[0b1001] +
                                                          2 * wfn_sd_coeff[0b1010] +
                                                          0 * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)
    assert np.isclose(test.get_overlap(0b0101, deriv=6), 0, rtol=0, atol=1e-12)
    assert np.isclose(test.get_overlap(0b0101, deriv=5), (1 * wfn_sd_coeff[0b1001] +
                                                          2 * wfn_sd_coeff[0b1010] +
                                                          0 * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)
    # 0b1100 uses [[0, 0, 11, 12],
    #              [0, 0, 15, 16]]
    assert np.isclose(test.get_overlap(0b1100, deriv=6), -12 * wfn_sd_coeff[0b1100],
                      rtol=0, atol=1e-12)

    # restricted (one block)
    test.clear_cache()
    test.params = [np.array([[1, 2], [5, 6]])]
    # 0b0110 uses [[5, 6, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(test.get_overlap(0b0110, deriv=0), (5 * wfn_sd_coeff[0b0101] +
                                                          6 * wfn_sd_coeff[0b0110] +
                                                          0 * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)

    # restricted (two block)
    # 0b0101 uses [[1, 2, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(test.get_overlap(0b0101, deriv=0), (0 * wfn_sd_coeff[0b0011] +
                                                          2*1 * wfn_sd_coeff[0b0101] +
                                                          2 * wfn_sd_coeff[0b1001] +
                                                          2 * wfn_sd_coeff[0b0110] +
                                                          0 * wfn_sd_coeff[0b1100]),
                      rtol=0, atol=1e-12)
