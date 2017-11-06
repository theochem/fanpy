"""Test wfns.objective.base_objective."""
from nose.tools import assert_raises
import numpy as np
from wfns.param import ParamContainer
from wfns.objective.base_objective import ParamMask


class TestParamMask(ParamMask):
    def __init__(self):
        pass


def test_load_mask_container_params():
    """Test ParamMask.load_mask_container_params."""
    test = TestParamMask()
    test._masks_container_params = {}
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
