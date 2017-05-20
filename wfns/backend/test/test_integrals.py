"""Test wfns.backend.integrals
"""
from nose.tools import assert_raises
import numpy as np
from wfns.backend.integrals import BaseIntegrals, OneElectronIntegrals, TwoElectronIntegrals


def test_base_init():
    """Test wfns.backend.integrals.BaseIntegrals.__init__."""
    # bad input
    assert_raises(TypeError, BaseIntegrals, {})
    assert_raises(TypeError, BaseIntegrals, 123)
    assert_raises(TypeError, BaseIntegrals, (i for i in range(4)))
    assert_raises(TypeError, BaseIntegrals, np.array([1, 2], dtype=str))
    assert_raises(TypeError, BaseIntegrals, np.array([1, 2], dtype=int))
    assert_raises(TypeError, BaseIntegrals, [np.array([1, 2], dtype=int),
                                             [1, 2]])
    assert_raises(TypeError, BaseIntegrals, [np.array([1, 2], dtype=int),
                                             np.array([1, 2], dtype=float)])
    assert_raises(TypeError, BaseIntegrals, [np.array([1, 2], dtype=complex),
                                             np.array([1, 2], dtype=float)])
    # good input
    test = np.array([1.0, 2.0])
    assert BaseIntegrals(test).integrals == (test, )
    assert BaseIntegrals([test, test]).integrals == (test, test)
    assert BaseIntegrals((test, test)).integrals == (test, test)
    assert BaseIntegrals(BaseIntegrals((test, test))).integrals == (test, test)


def test_one_init():
    """Test wfns.backend.integrals.OneElectronIntegrals.__init__."""
    assert_raises(TypeError, OneElectronIntegrals, 3*(np.random.rand(4, 4), ))
    assert_raises(TypeError, OneElectronIntegrals, np.random.rand(4, 4, 4))
    assert_raises(TypeError, OneElectronIntegrals, np.random.rand(4, 3))
    assert_raises(TypeError, OneElectronIntegrals, (np.random.rand(4, 4), np.random.rand(3, 3)))


def test_one_possible_orbtypes():
    """Test wfns.backend.integrals.OneElectronIntegrals.possible_orbtypes."""
    test = np.random.rand(4, 4)
    assert OneElectronIntegrals(test).possible_orbtypes == ('restricted', 'generalized')
    assert OneElectronIntegrals((test, test)).possible_orbtypes == ('unrestricted', )
    # forcing change
    temp = OneElectronIntegrals(test)
    temp.integrals = (test, test, test)
    assert_raises(NotImplementedError, lambda: temp.possible_orbtypes)


def test_one_num_orbs():
    """Test wfns.backend.integrals.OneElectronIntegrals.num_orbs."""
    assert OneElectronIntegrals(np.random.rand(4, 4)).num_orbs == 4
    assert OneElectronIntegrals([np.random.rand(3, 3), np.random.rand(3, 3)]).num_orbs == 3


def test_one_dtype():
    """Test wfns.backend.integrals.OneElectronIntegrals.dtype."""
    assert OneElectronIntegrals(np.random.rand(4, 4).astype(float)).dtype == float
    assert OneElectronIntegrals(np.random.rand(4, 4).astype(complex)).dtype == complex


def test_one_get_value():
    """Test wfns.backend.integrals.OneElectronIntegrals.get_value."""
    one_int = np.random.rand(4, 4)
    test = OneElectronIntegrals(one_int)
    # check errors
    assert_raises(ValueError, test.get_value, -1, 0, 'restricted')
    assert_raises(ValueError, test.get_value, 0, -1, 'unrestricted')
    assert_raises(ValueError, test.get_value, 0, -1, 'generalized')
    assert_raises(ValueError, test.get_value, 8, 0, 'restricted')
    assert_raises(ValueError, test.get_value, 0, 8, 'unrestricted')
    assert_raises(ValueError, test.get_value, 0, 4, 'generalized')
    assert_raises(TypeError, test.get_value, 0, 0, 'random type')
    # restricted
    one_int = np.arange(16, dtype=float).reshape(4, 4)
    test = OneElectronIntegrals(one_int)
    assert test.get_value(0, 0, 'restricted') == 0.0
    assert test.get_value(0, 1, 'restricted') == 1.0
    assert test.get_value(0, 4, 'restricted') == 0.0
    assert test.get_value(0, 5, 'restricted') == 0.0
    assert test.get_value(1, 0, 'restricted') == 4.0
    assert test.get_value(1, 1, 'restricted') == 5.0
    assert test.get_value(1, 4, 'restricted') == 0.0
    assert test.get_value(1, 5, 'restricted') == 0.0
    assert test.get_value(4, 0, 'restricted') == 0.0
    assert test.get_value(4, 1, 'restricted') == 0.0
    assert test.get_value(4, 4, 'restricted') == 0.0
    assert test.get_value(4, 5, 'restricted') == 1.0
    assert test.get_value(5, 0, 'restricted') == 0.0
    assert test.get_value(5, 1, 'restricted') == 0.0
    assert test.get_value(5, 4, 'restricted') == 4.0
    assert test.get_value(5, 5, 'restricted') == 5.0
    # unrestricted
    one_int = (np.arange(16, dtype=float).reshape(4, 4),
               np.arange(16, 32, dtype=float).reshape(4, 4))
    test = OneElectronIntegrals(one_int)
    assert test.get_value(0, 0, 'unrestricted') == 0.0
    assert test.get_value(0, 1, 'unrestricted') == 1.0
    assert test.get_value(0, 4, 'unrestricted') == 0.0
    assert test.get_value(0, 5, 'unrestricted') == 0.0
    assert test.get_value(1, 0, 'unrestricted') == 4.0
    assert test.get_value(1, 1, 'unrestricted') == 5.0
    assert test.get_value(1, 4, 'unrestricted') == 0.0
    assert test.get_value(1, 5, 'unrestricted') == 0.0
    assert test.get_value(4, 0, 'unrestricted') == 0.0
    assert test.get_value(4, 1, 'unrestricted') == 0.0
    assert test.get_value(4, 4, 'unrestricted') == 16.0
    assert test.get_value(4, 5, 'unrestricted') == 17.0
    assert test.get_value(5, 0, 'unrestricted') == 0.0
    assert test.get_value(5, 1, 'unrestricted') == 0.0
    assert test.get_value(5, 4, 'unrestricted') == 20.0
    assert test.get_value(5, 5, 'unrestricted') == 21.0
    # generalized
    one_int = (np.arange(64, dtype=float).reshape(8, 8), )
    test = OneElectronIntegrals(one_int)
    assert test.get_value(0, 0, 'generalized') == 0.0
    assert test.get_value(0, 1, 'generalized') == 1.0
    assert test.get_value(0, 4, 'generalized') == 4.0
    assert test.get_value(0, 5, 'generalized') == 5.0
    assert test.get_value(1, 0, 'generalized') == 8.0
    assert test.get_value(1, 1, 'generalized') == 9.0
    assert test.get_value(1, 4, 'generalized') == 12.0
    assert test.get_value(1, 5, 'generalized') == 13.0
    assert test.get_value(4, 0, 'generalized') == 32.0
    assert test.get_value(4, 1, 'generalized') == 33.0
    assert test.get_value(4, 4, 'generalized') == 36.0
    assert test.get_value(4, 5, 'generalized') == 37.0
    assert test.get_value(5, 0, 'generalized') == 40.0
    assert test.get_value(5, 1, 'generalized') == 41.0
    assert test.get_value(5, 4, 'generalized') == 44.0
    assert test.get_value(5, 5, 'generalized') == 45.0


def test_two_init():
    """Test wfns.backend.integrals.TwoElectronIntegrals.__init__."""
    assert_raises(TypeError, TwoElectronIntegrals, 2*(np.random.rand(4, 4, 4, 4), ))
    assert_raises(TypeError, TwoElectronIntegrals, 4*(np.random.rand(4, 4, 4, 4), ))
    assert_raises(TypeError, TwoElectronIntegrals, np.random.rand(4, 4, 4))
    assert_raises(TypeError, TwoElectronIntegrals, np.random.rand(4, 4, 3, 4))
    assert_raises(TypeError, TwoElectronIntegrals, (np.random.rand(3, 3, 3, 3),
                                                    np.random.rand(3, 3, 4, 3),
                                                    np.random.rand(3, 3, 3, 3)))
    assert_raises(ValueError, TwoElectronIntegrals, np.random.rand(2, 2, 2, 2), 'physicalchemist')


def test_two_possible_orbtypes():
    """Test wfns.backend.integrals.TwoElectronIntegrals.possible_orbtypes."""
    test = np.random.rand(4, 4, 4, 4)
    assert TwoElectronIntegrals(test).possible_orbtypes == ('restricted', 'generalized')
    assert TwoElectronIntegrals((test, test, test)).possible_orbtypes == ('unrestricted', )
    # forcing change
    temp = TwoElectronIntegrals(test)
    temp.integrals = (test, test)
    assert_raises(NotImplementedError, lambda: temp.possible_orbtypes)


def test_two_num_orbs():
    """Test wfns.backend.integrals.TwoElectronIntegrals.num_orbs."""
    assert TwoElectronIntegrals(np.random.rand(4, 4, 4, 4)).num_orbs == 4
    assert TwoElectronIntegrals([np.random.rand(3, 3, 3, 3),
                                 np.random.rand(3, 3, 3, 3),
                                 np.random.rand(3, 3, 3, 3)]).num_orbs == 3


def test_two_dtype():
    """Test wfns.backend.integrals.TwoElectronIntegrals.dtype."""
    assert TwoElectronIntegrals(np.random.rand(4, 4, 4, 4).astype(float)).dtype == float
    assert TwoElectronIntegrals(np.random.rand(4, 4, 4, 4).astype(complex)).dtype == complex


def test_two_get_value():
    """Test wfns.backend.integrals.TwoElectronIntegrals.get_value."""
    two_int = (np.arange(256, dtype=float).reshape(4, 4, 4, 4),)
    test = TwoElectronIntegrals(two_int)
    # check errors
    assert_raises(ValueError, test.get_value, -1, 0, 0, 0, 'restricted')
    assert_raises(ValueError, test.get_value, 0, -1, 0, 0, 'unrestricted')
    assert_raises(ValueError, test.get_value, 0, 0, -1, 0, 'generalized')
    assert_raises(ValueError, test.get_value, 8, 0, 0, 0, 'restricted')
    assert_raises(ValueError, test.get_value, 0, 8, 0, 0, 'unrestricted')
    assert_raises(ValueError, test.get_value, 0, 0, 4, 0, 'generalized')
    assert_raises(TypeError, test.get_value, 0, 0, 0, 0, 'random type')
    # restricted
    two_int = (np.arange(256, dtype=float).reshape(4, 4, 4, 4),)
    test = TwoElectronIntegrals(two_int)
    assert test.get_value(0, 0, 0, 1, 'restricted') == 1.0
    assert test.get_value(0, 0, 4, 1, 'restricted') == 0.0
    assert test.get_value(0, 4, 0, 1, 'restricted') == 0.0
    assert test.get_value(4, 0, 0, 1, 'restricted') == 0.0
    assert test.get_value(0, 4, 4, 1, 'restricted') == 0.0
    assert test.get_value(4, 0, 4, 1, 'restricted') == 1.0
    assert test.get_value(4, 4, 0, 1, 'restricted') == 0.0
    assert test.get_value(4, 4, 4, 1, 'restricted') == 0.0
    assert test.get_value(0, 0, 0, 5, 'restricted') == 0.0
    assert test.get_value(0, 0, 4, 5, 'restricted') == 0.0
    assert test.get_value(0, 4, 0, 5, 'restricted') == 1.0
    assert test.get_value(4, 0, 0, 5, 'restricted') == 0.0
    assert test.get_value(0, 4, 4, 5, 'restricted') == 0.0
    assert test.get_value(4, 0, 4, 5, 'restricted') == 0.0
    assert test.get_value(4, 4, 0, 5, 'restricted') == 0.0
    assert test.get_value(4, 4, 4, 5, 'restricted') == 1.0
    # unrestricted
    two_int = (np.arange(256, dtype=float).reshape(4, 4, 4, 4),
               np.arange(256, 512, dtype=float).reshape(4, 4, 4, 4),
               np.arange(512, 768, dtype=float).reshape(4, 4, 4, 4))
    test = TwoElectronIntegrals(two_int)
    assert test.get_value(0, 0, 0, 1, 'unrestricted') == 1.0
    assert test.get_value(0, 0, 4, 1, 'unrestricted') == 0.0
    assert test.get_value(0, 4, 0, 1, 'unrestricted') == 0.0
    assert test.get_value(4, 0, 0, 1, 'unrestricted') == 0.0
    assert test.get_value(0, 4, 4, 1, 'unrestricted') == 0.0
    assert test.get_value(4, 0, 4, 1, 'unrestricted') == 260.0
    assert test.get_value(4, 4, 0, 1, 'unrestricted') == 0.0
    assert test.get_value(4, 4, 4, 1, 'unrestricted') == 0.0
    assert test.get_value(0, 0, 0, 5, 'unrestricted') == 0.0
    assert test.get_value(0, 0, 4, 5, 'unrestricted') == 0.0
    assert test.get_value(0, 4, 0, 5, 'unrestricted') == 257.0
    assert test.get_value(4, 0, 0, 5, 'unrestricted') == 0.0
    assert test.get_value(0, 4, 4, 5, 'unrestricted') == 0.0
    assert test.get_value(4, 0, 4, 5, 'unrestricted') == 0.0
    assert test.get_value(4, 4, 0, 5, 'unrestricted') == 0.0
    assert test.get_value(4, 4, 4, 5, 'unrestricted') == 513.0
    # generalized
    two_int = (np.arange(4096, dtype=float).reshape(8, 8, 8, 8), )
    test = TwoElectronIntegrals(two_int)
    assert test.get_value(0, 0, 0, 1, 'generalized') == 1.0
    assert test.get_value(0, 0, 4, 1, 'generalized') == 33.0
    assert test.get_value(0, 4, 0, 1, 'generalized') == 257.0
    assert test.get_value(4, 0, 0, 1, 'generalized') == 2049.0
    assert test.get_value(0, 4, 4, 1, 'generalized') == 289.0
    assert test.get_value(4, 0, 4, 1, 'generalized') == 2081.0
    assert test.get_value(4, 4, 0, 1, 'generalized') == 2305.0
    assert test.get_value(4, 4, 4, 1, 'generalized') == 2337.0
    assert test.get_value(0, 0, 0, 5, 'generalized') == 5.0
    assert test.get_value(0, 0, 4, 5, 'generalized') == 37.0
    assert test.get_value(0, 4, 0, 5, 'generalized') == 261.0
    assert test.get_value(4, 0, 0, 5, 'generalized') == 2053.0
    assert test.get_value(0, 4, 4, 5, 'generalized') == 293.0
    assert test.get_value(4, 0, 4, 5, 'generalized') == 2085.0
    assert test.get_value(4, 4, 0, 5, 'generalized') == 2309.0
    assert test.get_value(4, 4, 4, 5, 'generalized') == 2341.0
