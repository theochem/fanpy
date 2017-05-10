""" Tests wfns.wavefunction.wavefunctions
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.wavefunction.wavefunction import Wavefunction


def test_assign_nelec():
    """
    Tests Wavefunction.assign_nelec
    """
    test = Wavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check errors
    assert_raises(TypeError, lambda: test.assign_nelec(None))
    assert_raises(TypeError, lambda: test.assign_nelec(2.0))
    assert_raises(TypeError, lambda: test.assign_nelec('2'))
    assert_raises(ValueError, lambda: test.assign_nelec(0))
    assert_raises(ValueError, lambda: test.assign_nelec(-2))
    # int
    test.assign_nelec(2)
    assert test.nelec == 2
    # long
    test.assign_nelec(long(2))
    assert test.nelec == 2

def test_assign_dtype():
    """
    Tests Wavefunction.assign_dtype
    """
    test = Wavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check errors
    assert_raises(TypeError, lambda: test.assign_dtype(''))
    assert_raises(TypeError, lambda: test.assign_dtype('float64'))
    assert_raises(TypeError, lambda: test.assign_dtype(int))
    assert_raises(TypeError, lambda: test.assign_dtype(np.float32))
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



def test_assign_nuc_nuc():
    """
    Tests Wavefunction.assign_nuc_nuc
    """
    test = Wavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check errors
    assert_raises(TypeError, lambda: test.assign_nuc_nuc(0))
    assert_raises(TypeError, lambda: test.assign_nuc_nuc(123))
    assert_raises(TypeError, lambda: test.assign_nuc_nuc('123'))
    # None
    test.assign_nuc_nuc(None)
    assert test.nuc_nuc == 0.0
    # float
    test.assign_nuc_nuc(123.0)
    assert test.nuc_nuc == 123.0

def test_assign_integrals_typecheck():
    """
    Tests the type checks in Wavefunction.assign_integrals
    """
    test = Wavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check errors
    # If one_int and two_int are not a numpy array or a tuple of numpy arrays
    assert_raises(TypeError, lambda: test.assign_integrals(12, np.ones((3, 3, 3, 3)), orbtype=None))
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)), ),
                                                           {np.ones((3, 3, 3, 3)):None},
                                                           orbtype=None))
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)), ),
                                                           (np.ones((3, 3, 3, 3)), ) * 4,
                                                           orbtype=None))
    assert_raises(TypeError, lambda: test.assign_integrals([], (np.ones((3, 3, 3, 3)), ) * 2,
                                                           orbtype=None))
    # If orbital type from one_int and orbital type from two_int are not the same
    assert_raises(TypeError, lambda: test.assign_integrals(np.ones((3, 3)),
                                                           (np.ones((3, 3, 3, 3)), ) * 3,
                                                           orbtype=None))
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)), ) * 2,
                                                           np.ones((3, 3, 3, 3)), orbtype=None))
    # If orbital type is inconsistent with the integrals given
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)), ) * 2,
                                                           (np.ones((3, 3, 3, 3))) * 3,
                                                           orbtype=None))
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)), ) * 2,
                                                           (np.ones((3, 3, 3, 3))) * 3,
                                                           orbtype='restricted'))
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)), ) * 2,
                                                           (np.ones((3, 3, 3, 3))) * 3,
                                                           orbtype='generalized'))
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)), ),
                                                           (np.ones((3, 3, 3, 3)), ),
                                                           orbtype='unrestricted'))
    assert_raises(TypeError, lambda: test.assign_integrals(np.ones((3, 3)),
                                                           np.ones((3, 3, 3, 3)),
                                                           orbtype='unrestricted'))
    # If one_int and two_int are tuples and its elements are not numpy arrays
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)).tolist(), ),
                                                           (np.ones((3, 3, 3, 3)), ),
                                                           orbtype='restricted'))
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)), ),
                                                           (np.ones((3, 3, 3, 3)).tolist(), ),
                                                           orbtype='generalized'))
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)),
                                                            np.ones((3, 3)).tolist()),
                                                           (np.ones((3, 3, 3, 3)), ) * 3,
                                                           orbtype='unrestricted'))
    # If one_int and two_int are tuples and numpy arrays do not have the consistent shapes
    assert_raises(TypeError, lambda: test.assign_integrals(np.ones((4, 4)),
                                                           np.ones((3, 3, 3, 3)),
                                                           orbtype='restricted'))
    assert_raises(TypeError, lambda: test.assign_integrals(np.ones((4, 3)),
                                                           np.ones((3, 3, 3, 3)),
                                                           orbtype='generalized'))
    assert_raises(TypeError, lambda: test.assign_integrals((np.ones((3, 3)), ),
                                                           (np.ones((3, 3, 3, 4)), ),
                                                           orbtype='unrestricted'))
    # If one_int and two_int are tuples and wavefunction data type is float and numpy arrays' data
    # type is not float
    test.dtype = np.float64
    assert_raises(TypeError, lambda: test.assign_integrals(np.ones((4, 4), dtype=int),
                                                           np.ones((3, 3, 3, 3)),
                                                           orbtype='restricted'))
    assert_raises(TypeError, lambda: test.assign_integrals(np.ones((4, 4)),
                                                           np.ones((3, 3, 3, 3), dtype=complex),
                                                           orbtype='restricted'))
    # If one_int and two_int are tuples and wavefunction data type is complex and numpy arrays' data
    # type is not float or complex
    test.dtype = np.complex128
    assert_raises(TypeError, lambda: test.assign_integrals(np.ones((4, 4), dtype=int),
                                                           np.ones((3, 3, 3, 3)),
                                                           orbtype='restricted'))
    assert_raises(TypeError, lambda: test.assign_integrals(np.ones((4, 4)),
                                                           np.ones((3, 3, 3, 3), dtype=bool),
                                                           orbtype='restricted'))
    # If generalized orbitals and odd number of spin orbitals
    assert_raises(NotImplementedError, lambda: test.assign_integrals(np.ones((11, 11)),
                                                                     np.ones((11, 11, 11, 11)),
                                                                     orbtype='generalized'))

def test_assign_integrals():
    """
    Tests Wavefunction.assign_integrals
    """
    test = Wavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # one_int and two_int are numpy arrays
    one_int = np.random.rand(10, 10)
    two_int = np.random.rand(10, 10, 10, 10)
    test.assign_integrals(one_int, two_int, orbtype=None)
    assert np.allclose(one_int, test.one_int)
    assert np.allclose(two_int, test.two_int)
    # one_int and two_int are tuples (restricted/generalized)
    test.assign_integrals((one_int, ), (two_int, ), orbtype=None)
    assert np.allclose(one_int, test.one_int)
    assert np.allclose(two_int, test.two_int)
    # one_int and two_int are tuples (unrestricted)
    test.assign_integrals((one_int, )*2, (two_int, )*3, orbtype=None)
    assert np.allclose(one_int, test.one_int)
    assert np.allclose(two_int, test.two_int)
    # check orbtype
    test.assign_integrals(one_int, two_int, orbtype=None)
    assert test.orbtype == 'restricted'
    test.assign_integrals(one_int, two_int, orbtype='restricted')
    assert test.orbtype == 'restricted'
    test.assign_integrals((one_int, ), (two_int, ), orbtype='restricted')
    assert test.orbtype == 'restricted'
    test.assign_integrals(one_int, two_int, orbtype='generalized')
    assert test.orbtype == 'generalized'

    test.assign_integrals((one_int, )*2, (two_int, )*3, orbtype=None)
    assert test.orbtype == 'unrestricted'
    test.assign_integrals((one_int, )*2, (two_int, )*3, orbtype='unrestricted')
    assert test.orbtype == 'unrestricted'


def test_nspin():
    """
    Tests Wavefunction.nspin
    """
    test = Wavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # one_int and two_int are numpy arrays
    one_int = np.random.rand(10, 10)
    two_int = np.random.rand(10, 10, 10, 10)
    test.assign_integrals(one_int, two_int, orbtype='restricted')
    assert test.nspin == 20
    test.assign_integrals(one_int, two_int, orbtype='generalized')
    assert test.nspin == 10
    # one_int and two_int are tuples
    test.assign_integrals((one_int, )*2, (two_int, )*3, orbtype='unrestricted')
    assert test.nspin == 20


def test_nspatial():
    """
    Tests Wavefunction.nspatial
    """
    test = Wavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # restricted
    one_int = np.random.rand(11, 11)
    two_int = np.random.rand(11, 11, 11, 11)
    test.assign_integrals(one_int, two_int, orbtype='restricted')
    assert test.nspatial == 11
    # unrestricted
    test.assign_integrals((one_int, )*2, (two_int, )*3, orbtype='unrestricted')
    assert test.nspatial == 11
    # generalized
    one_int = np.random.rand(10, 10)
    two_int = np.random.rand(10, 10, 10, 10)
    test.assign_integrals(one_int, two_int, orbtype='generalized')
    assert test.nspatial == 5
