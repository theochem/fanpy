"""Test wfns.param."""
import numpy as np
from nose.tools import assert_raises
from wfns.param import ParamContainer


def test_assign_param():
    """Test ParamContainer.assign_param."""
    TestClass = ParamContainer
    TestClass.__init__ = lambda params: None
    test = TestClass()
    # int
    test.assign_params(1)
    assert np.array_equal(test.params, np.array(1))
    assert test.params.dtype == int
    # float
    test.assign_params(1.0)
    assert np.array_equal(test.params, np.array(1.0))
    assert test.params.dtype == float
    # complex
    test.assign_params(1.0j)
    assert np.array_equal(test.params, np.array(1.0j))
    assert test.params.dtype == complex
    # numpy array of int
    test.assign_params(np.array([1, 2]))
    assert np.array_equal(test.params, np.array([1, 2]))
    assert test.params.dtype == int
    # numpy array of float
    test.assign_params(np.array([1.0, 2.0]))
    assert np.array_equal(test.params, np.array([1.0, 2.0]))
    assert test.params.dtype == float
    # numpy array of complex
    test.assign_params(np.array([1.0j, 2.0j]))
    assert np.array_equal(test.params, np.array([1.0j, 2.0j]))
    assert test.params.dtype == complex

    # other iterables
    assert_raises(TypeError, test.assign_params, [1, 2])
    assert_raises(TypeError, test.assign_params, (1, 2))
    # bad dtypes
    assert_raises(TypeError, test.assign_params, np.array([1, 2], dtype=str))


def test_nparams():
    """Test ParamContainer.nparams."""
    test = ParamContainer(np.random.rand(10))
    assert test.nparams == 10
    test = ParamContainer(np.random.rand(10, 10))
    assert test.nparams == 100
