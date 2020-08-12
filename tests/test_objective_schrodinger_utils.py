"""Test fanpy.eqn.utils."""
import collections

import numpy as np
import pytest
from utils import skip_init
from fanpy.eqn.utils import ParamContainer, ComponentParameterIndices


def test_assign_param():
    """Test ParamContainer.assign_param."""
    test = skip_init(ParamContainer)
    # int
    test.assign_params(1)
    assert np.array_equal(test.params, np.array([1]))
    assert test.params.dtype == int
    # float
    test.assign_params(1.0)
    assert np.array_equal(test.params, np.array([1.0]))
    assert test.params.dtype == float
    # complex
    test.assign_params(1.0j)
    assert np.array_equal(test.params, np.array([1.0j]))
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

    # bad dtypes
    with pytest.raises(TypeError):
        test.assign_params(np.array([[1, 2]]))
    with pytest.raises(TypeError):
        test.assign_params(np.array([1, 2], dtype=str))
    with pytest.raises(ValueError):
        test.assign_params(np.array([]))


def test_nparams():
    """Test ParamContainer.nparams."""
    test = ParamContainer(np.random.rand(10))
    assert test.nparams == 10


def test_indices_setitem():
    """Test ComponentParameterIndices.__setitem__."""
    test = ComponentParameterIndices()
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test[param1] = []
    assert isinstance(test[param1], np.ndarray)
    assert test[param1].size == 0
    assert test[param1].dtype == int
    test[param2] = (0, )
    assert np.allclose(test[param2], np.array([0]))
    test[param3] = np.array([2, 1])
    assert np.allclose(test[param3], np.array([1, 2]))
    test[param3] = np.array([True, False, False, True])
    assert np.allclose(test[param3], np.array([0, 3]))

    with pytest.raises(TypeError):
        test[2] = []
    with pytest.raises(TypeError):
        test[param3] = set([3])
    with pytest.raises(TypeError):
        test[param3] = np.array([[0, 1]])
    with pytest.raises(TypeError):
        test[param3] = np.array([0, 1], dtype=float)
    with pytest.raises(ValueError):
        test[param3] = np.array([True, True, False, False, False])
    with pytest.raises(ValueError):
        test[param3] = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        test[param3] = np.array([0, 1, -1])
    with pytest.raises(ValueError):
        test[param3] = np.array([0, 1, 4])


def test_indices_getitem():
    """Test ComponentParameterIndices.__getitem__."""
    test = ComponentParameterIndices()
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test[param2] = np.array([0])
    test[param3] = np.array([2, 1])
    assert isinstance(test[param1], np.ndarray)
    assert test[param1].size == 0
    assert test[param1].dtype == int
    assert np.allclose(test[param2], np.array([0]))
    assert np.allclose(test[param3], np.array([1, 2]))


def test_indices_eq():
    """Test ComponentParameterIndices.__eq__ and __ne__."""
    test = ComponentParameterIndices()
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test[param1] = []
    test[param2] = (0, )
    test[param3] = np.array([True, False, False, True])

    test2 = ComponentParameterIndices()
    test2[param1] = np.array([])
    test2[param2] = (True, False)
    test2[param3] = np.array([3, 0])

    assert not test == 1
    assert test != 1
    assert test == test
    assert not test != test
    assert test == test2
    assert not test != test2

    test[param2] = [1]
    assert not test == test2
    assert test != test2
