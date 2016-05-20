from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises

import sys
sys.path.append('../')
from wavefunction import Wavefunction
import numpy as np

class TestWavefunction(Wavefunction):
    # overwrite to stop initialization
    def __init__(self):
        pass
    @property
    def _methods(self):
        pass

def test_assign_dtype():
    """
    Tests Wavefunction.assign_dtype
    """
    test = TestWavefunction()
    # None assigned
    test.assign_dtype(None)
    assert test.dtype == np.float64
    # Invalid assignment
    assert_raises(TypeError, lambda:test.assign_dtype('sdf'))
    assert_raises(TypeError, lambda:test.assign_dtype(''))
    assert_raises(TypeError, lambda:test.assign_dtype(np.float32))
    # other assignments
    all_types = [float, complex, np.float64, np.complex128]
    for dtype in all_types:
        test.assign_dtype(dtype)
        assert test.dtype == dtype

def test_assign_integrals():
    """
    Tests Wavefunction.assign_integrals
    """
    test = TestWavefunction()
    # H and G are numpy arrays
    H = np.arange(100, dtype=float).reshape(10,10)
    G = np.arange(10000, dtype=float).reshape(10,10,10,10)
    test.assign_integrals(H, G)
    assert np.allclose(H, test.H)
    assert np.allclose(G, test.G)
    # H and G are tuples
    H = (np.arange(100, dtype=float).reshape(10,10),)*2
    G = (np.arange(10000, dtype=float).reshape(10,10,10,10),)*3
    test.assign_integrals(H, G)
    assert np.allclose(H, test.H)
    assert np.allclose(G, test.G)
    # bad H type
    H = np.arange(100, dtype=float).reshape(10,10).tolist()
    G = np.arange(10000, dtype=float).reshape(10,10,10,10)
    assert_raises(TypeError, lambda:test.assign_integrals(H, G))
    H = ()
    assert_raises(TypeError, lambda:test.assign_integrals(H, G))
    H = (1, 2)
    assert_raises(TypeError, lambda:test.assign_integrals(H, G))
    H = (np.arange(100, dtype=float).reshape(10,10),)*2
    assert_raises(TypeError, lambda:test.assign_integrals(H, G))
    H = (1, 2, 3)
    assert_raises(TypeError, lambda:test.assign_integrals(H, G))
    # bad G types
    H = np.arange(100, dtype=float).reshape(10,10)
    G = np.arange(10000, dtype=float).reshape(10,10,10,10)
    G = G.tolist()
    assert_raises(TypeError, lambda:test.assign_integrals(H, G))
    G = ()
    assert_raises(TypeError, lambda:test.assign_integrals(H, G))
    G = (np.arange(10000, dtype=float).reshape(10,10,10,10),)
    assert_raises(TypeError, lambda:test.assign_integrals(H, G))
    G = (np.arange(10000, dtype=float).reshape(10,10,10,10),)*2
    assert_raises(TypeError, lambda:test.assign_integrals(H, G))
    G = (np.arange(10000, dtype=float).reshape(10,10,10,10),)*3
    assert_raises(TypeError, lambda:test.assign_integrals(H, G))
    # check orb_type
    H = np.arange(100, dtype=float).reshape(10,10)
    G = np.arange(10000, dtype=float).reshape(10,10,10,10)
    test.assign_integrals(H, G)
    assert test.orb_type == 'restricted'
    test.assign_integrals(H, G, 'restricted')
    assert test.orb_type == 'restricted'
    assert_raises(TypeError, lambda:test.assign_integrals(H, G, 'unrestricted'))
    test.assign_integrals(H, G, 'generalized')
    assert test.orb_type == 'generalized'
    assert_raises(ValueError, lambda:test.assign_integrals(H, G, 'random'))

    H = (np.arange(100, dtype=float).reshape(10,10),)*2
    G = (np.arange(10000, dtype=float).reshape(10,10,10,10),)*3
    test.assign_integrals(H, G)
    assert test.orb_type == 'unrestricted'
    test.assign_integrals(H, G, 'unrestricted')
    assert test.orb_type == 'unrestricted'
    assert_raises(TypeError, lambda:test.assign_integrals(H, G, 'restricted'))
    assert_raises(TypeError, lambda:test.assign_integrals(H, G, 'generalized'))

def test_assign_nuc_nuc():
    """
    Tests Wavefunction.assign_nuc_nuc
    """
    test = TestWavefunction()
    test.assign_nuc_nuc(None)
    assert test.nuc_nuc == 0.0
    test.assign_nuc_nuc(123.0)
    assert test.nuc_nuc == 123.0
    assert_raises(TypeError, lambda:test.assign_nuc_nuc(123))
    assert_raises(TypeError, lambda:test.assign_nuc_nuc('123'))

def test_assign_nelec():
    """
    Tests Wavefunction.assign_nelec
    """
    test = TestWavefunction()
    test.assign_nelec(2)
    assert test.nelec == 2
    assert_raises(TypeError, lambda:test.assign_nelec(None))
    assert_raises(TypeError, lambda:test.assign_nelec(2.0))
    assert_raises(ValueError, lambda:test.assign_nelec(0))
    assert_raises(ValueError, lambda:test.assign_nelec(-2))

def test_compute_energy():
    """
    Tests Wavefunction.compute_energy
    """
    test = TestWavefunction()
    test.assign_nuc_nuc(3.0)
    test._energy = 4.0
    assert test.compute_energy(include_nuc=False) == 4.0
    assert test.compute_energy(include_nuc=True) == 7.0

def test_nspin():
    """
    Tests Wavefunction.nspin
    """
    test = TestWavefunction()
    # H and G are numpy arrays
    H = np.arange(100, dtype=float).reshape(10,10)
    G = np.arange(10000, dtype=float).reshape(10,10,10,10)
    test.assign_integrals(H, G, 'restricted')
    assert test.nspin == 20
    test.assign_integrals(H, G, 'generalized')
    assert test.nspin == 10
    # H and G are tuples
    H = (np.arange(100, dtype=float).reshape(10,10),)*2
    G = (np.arange(10000, dtype=float).reshape(10,10,10,10),)*3
    test.assign_integrals(H, G, 'unrestricted')
    assert test.nspin == 20

def test_nspatial():
    """
    Tests Wavefunction.nspatial
    """
    test = TestWavefunction()
    # H and G are numpy arrays
    H = np.arange(121, dtype=float).reshape(11,11)
    G = np.arange(14641, dtype=float).reshape(11,11,11,11)
    test.assign_integrals(H, G, 'restricted')
    assert test.nspatial == 11
    test.assign_integrals(H, G, 'generalized')
    assert test.nspatial == 5
    # H and G are tuples
    H = (np.arange(121, dtype=float).reshape(11,11),)*2
    G = (np.arange(14641, dtype=float).reshape(11,11,11,11),)*3
    test.assign_integrals(H, G, 'unrestricted')
    assert test.nspatial == 11

def test_npair():
    """
    Tests Wavefunction.npair
    """
    test = TestWavefunction()
    test.assign_nelec(2)
    assert test.npair == 1
    test.assign_nelec(3)
    assert test.npair == 1
