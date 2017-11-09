"""Test wfns.param."""
import collections
import numpy as np
from nose.tools import assert_raises
from wfns.param import ParamContainer, ParamMask


class TestParamContainer(ParamContainer):
    def __init__(self):
        pass


class TestParamMask(ParamMask):
    def __init__(self):
        pass


def test_assign_param():
    """Test ParamContainer.assign_param."""
    test = TestParamContainer()
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


def test_parammask_load_mask_container_params():
    """Test ParamMask.load_mask_container_params."""
    test = TestParamMask()
    test._masks_container_params = collections.OrderedDict()
    assert_raises(TypeError, test.load_mask_container_params, 1, np.array([2]))
    container = ParamContainer(np.arange(10))
    assert_raises(TypeError, test.load_mask_container_params, container, range(10))
    assert_raises(TypeError, test.load_mask_container_params, container, np.arange(10, dtype=float))
    assert_raises(ValueError, test.load_mask_container_params, container, np.array([-1]))
    assert_raises(ValueError, test.load_mask_container_params, container, np.array([10]))
    assert_raises(ValueError, test.load_mask_container_params, container, np.zeros(11, dtype=bool))

    sel = np.array([0, 1])
    test.load_mask_container_params(container, sel)
    assert np.allclose(test._masks_container_params[container], sel)
    sel = np.zeros(10, dtype=bool)
    sel[np.array([1, 3, 5])] = True
    test.load_mask_container_params(container, sel)
    assert np.allclose(test._masks_container_params[container], np.array([1, 3, 5]))
    sel = np.zeros(10, dtype=bool)
    sel[np.array([5, 3, 1])] = True
    test.load_mask_container_params(container, sel)
    assert np.allclose(test._masks_container_params[container], np.array([1, 3, 5]))
    sel = np.zeros(10, dtype=bool)
    sel[np.array([5, 3, 5])] = True
    test.load_mask_container_params(container, sel)
    assert np.allclose(test._masks_container_params[container], np.array([3, 5]))


def test_parammask_load_mask_objective_params():
    """Test ParamMask.load_mask_objective_params."""
    test = TestParamMask()
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test._masks_container_params = {param1: np.array([0]),
                                    param2: np.array([1]),
                                    param3: np.array([2, 3])}
    test.load_masks_objective_params()
    assert np.allclose(test._masks_objective_params[param1], np.array([True, False, False, False]))
    assert np.allclose(test._masks_objective_params[param2], np.array([False, True, False, False]))
    assert np.allclose(test._masks_objective_params[param3], np.array([False, False, True, True]))


def test_parammask_all_params():
    """Test ParamMask.all_params."""
    test = ParamMask((ParamContainer(1), False), (ParamContainer(np.array([2, 3])), np.array(0)),
                     (ParamContainer(np.array([4, 5, 6, 7])), np.array([True, False, False, True])))
    assert np.allclose(test.all_params, np.array([1, 2, 3, 4, 5, 6, 7]))


def test_parammask_active_params():
    """Test ParamMask.active_params."""
    test = ParamMask((ParamContainer(1), False), (ParamContainer(np.array([2, 3])), np.array(0)),
                     (ParamContainer(np.array([4, 5, 6, 7])), np.array([True, False, False, True])))
    assert np.allclose(test.active_params, np.array([2, 4, 7]))


def test_parammask_load_params():
    """Test ParamMask.load_params."""
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test = ParamMask((param1, False), (param2, np.array(0)),
                     (param3, np.array([True, False, False, True])))
    assert_raises(TypeError, test.load_params, [9, 10, 11])
    assert_raises(TypeError, test.load_params, np.array([[9, 10, 11]]))
    assert_raises(ValueError, test.load_params, np.array([9, 10, 11, 12]))
    test.load_params(np.array([9, 10, 11]))
    assert np.allclose(param1.params, 1)
    assert np.allclose(param2.params, np.array([9, 3]))
    assert np.allclose(param3.params, np.array([10, 5, 6, 11]))


def test_parammask_derivative_index():
    """Test ParamMask.derivative_index."""
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test = ParamMask((param1, False), (param2, np.array(1)),
                     (param3, np.array([True, False, False, True])))
    assert_raises(TypeError, test.derivative_index, (1, 0))
    assert test.derivative_index(ParamContainer(2), 0) is None
    assert test.derivative_index(param1, 0) is None
    assert test.derivative_index(param2, 0) == 1
    assert test.derivative_index(param2, 1) is None
    assert test.derivative_index(param3, 0) is None
    assert test.derivative_index(param3, 1) == 0
    assert test.derivative_index(param3, 2) == 3
