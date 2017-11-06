"""Test wfns.objective.base_objective."""
from nose.tools import assert_raises
import collections
import numpy as np
from wfns.param import ParamContainer
from wfns.objective.base_objective import ParamMask, BaseObjective
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.hamiltonian.base_hamiltonian import BaseHamiltonian


class TestParamMask(ParamMask):
    def __init__(self):
        pass


class TestBaseHamiltonian(BaseHamiltonian):
    def integrate_wfn_sd(self, wfn, sd, deriv=None):
        pass

    def integrate_sd_sd(self, sd1, sd2, deriv=None):
        pass


class TestBaseObjective(BaseObjective):
    def objective(self, params):
        pass


def test_load_mask_container_params():
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


def test_load_mask_objective_params():
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


def test_all_params():
    """Test ParamMask.all_params."""
    test = ParamMask((ParamContainer(1), False), (ParamContainer(np.array([2, 3])), np.array(0)),
                     (ParamContainer(np.array([4, 5, 6, 7])), np.array([True, False, False, True])))
    assert np.allclose(test.all_params, np.array([1, 2, 3, 4, 5, 6, 7]))


def test_active_params():
    """Test ParamMask.active_params."""
    test = ParamMask((ParamContainer(1), False), (ParamContainer(np.array([2, 3])), np.array(0)),
                     (ParamContainer(np.array([4, 5, 6, 7])), np.array([True, False, False, True])))
    assert np.allclose(test.active_params, np.array([2, 4, 7]))


def test_load_params():
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


def test_derivative_index():
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


def test_baseobjective_init():
    """Test BaseObjective.__init__."""
    wfn = CIWavefunction(2, 4)
    ham = TestBaseHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    assert_raises(TypeError, TestBaseObjective, ham, ham)
    assert_raises(TypeError, TestBaseObjective, wfn, wfn)
    wfn = CIWavefunction(2, 4, dtype=complex)
    assert_raises(ValueError, TestBaseObjective, wfn, ham)
    wfn = CIWavefunction(2, 6)
    assert_raises(ValueError, TestBaseObjective, wfn, ham)
    wfn = CIWavefunction(2, 4)
    assert_raises(TypeError, TestBaseObjective, wfn, ham, tmpfile=2)

    test = TestBaseObjective(wfn, ham, tmpfile='tmpfile.npy')
    assert test.wfn == wfn
    assert test.ham == ham
    assert test.tmpfile == 'tmpfile.npy'
    assert np.allclose(test.param_selection.all_params, wfn.params)
    assert np.allclose(test.param_selection.active_params, wfn.params)


def test_baseobjective_assign_param_selection():
    """Test BaseObjective.assign_param_selection."""
    wfn = CIWavefunction(2, 4)
    ham = TestBaseHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = TestBaseObjective(wfn, ham)

    test.assign_param_selection(None)
    assert len(test.param_selection._masks_container_params) == 1
    container, sel = test.param_selection._masks_container_params.popitem()
    assert container == wfn
    assert np.allclose(sel, np.arange(wfn.nparams))

    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test.assign_param_selection([(param1, False), (param2, np.array(0)),
                                 (param3, np.array([True, False, False, True]))])
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


def test_baseobjective_params():
    """Test BaseObjective.params."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = TestBaseHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = TestBaseObjective(wfn, ham, param_selection=[(wfn, np.array([0, 3, 5]))])
    assert np.allclose(test.params, wfn.params[np.array([0, 3, 5])])


def test_baseobjective_assign_params():
    """Test BaseObjective.assign_params."""
    wfn = CIWavefunction(2, 4)
    params = np.random.rand(wfn.nparams)
    wfn.assign_params(params)
    ham = TestBaseHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = TestBaseObjective(wfn, ham, param_selection=[(wfn, np.array([0, 3, 5]))])
    test.assign_params(np.array([99, 98, 97]))
    params[np.array([0, 3, 5])] = [99, 98, 97]
    assert np.allclose(params, wfn.params)
