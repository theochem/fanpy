"""Test wfns.objective.base_objective."""
from nose.tools import assert_raises
import numpy as np
from wfns.param import ParamContainer, ParamMask
from wfns.objective.base_objective import BaseObjective


class TestBaseObjective(BaseObjective):
    @property
    def num_eqns(self):
        pass

    def objective(self, params):
        pass


def test_baseobjective_init():
    """Test BaseObjective.__init__."""
    test = TestBaseObjective(tmpfile='tmpfile.npy')
    assert test.tmpfile == 'tmpfile.npy'
    assert_raises(TypeError, TestBaseObjective, tmpfile=23)


def test_baseobjective_assign_param_selection():
    """Test BaseObjective.assign_param_selection."""
    test = TestBaseObjective()

    test.assign_param_selection(())
    assert isinstance(test.param_selection, ParamContainer)

    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    param_selection = [(param1, False), (param2, np.array(0)),
                       (param3, np.array([True, False, False, True]))]
    for mask in [param_selection, ParamMask(*param_selection)]:
        test.assign_param_selection(mask)
        assert len(test.param_selection._masks_container_params) == 3
        container, sel = test.param_selection._masks_container_params.popitem()
        assert container == param3
        assert np.allclose(sel, np.array([0, 3]))
        container, sel = test.param_selection._masks_container_params.popitem()
        assert container == param2
        assert np.allclose(sel, np.array([0]))
        container, sel = test.param_selection._masks_container_params.popitem()
        assert container == param1
        assert np.allclose(sel, np.array([]))

    assert_raises(TypeError, test.assign_param_selection, np.array([(param1, False)]))


def test_baseobjective_params():
    """Test BaseObjective.params."""
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test = TestBaseObjective([(param1, False), (param2, np.array(0)),
                              (param3, np.array([True, False, False, True]))])
    assert np.allclose(test.params, np.array([2, 4, 7]))


def test_baseobjective_assign_params():
    """Test BaseObjective.assign_params."""
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test = TestBaseObjective([(param1, False), (param2, np.array(0)),
                              (param3, np.array([True, False, False, True]))])
    test.assign_params(np.array([99, 98, 97]))
    assert np.allclose(param1.params, [1])
    assert np.allclose(param2.params, [99, 3])
    assert np.allclose(param3.params, [98, 5, 6, 97])
