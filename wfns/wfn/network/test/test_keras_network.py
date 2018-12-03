"Tests for wfns.wfn.network.keras_network.KerasNetwork."
import numpy as np
import keras
from nose.tools import assert_raises
from wfns.wfn.network.keras_network import KerasNetwork
import wfns.backend.slater as slater
from wfns.backend.sd_list import sd_list


class TestKerasNetwork(KerasNetwork):
    """KerasNetwork that skips initialization."""
    def __init__(self):
        pass


def test_assign_dtype():
    "Test KerasNetwork.assign_dtype."
    test = TestKerasNetwork()
    test.assign_dtype(float)
    assert test.dtype == np.float64
    assert_raises(ValueError, test.assign_dtype, complex)


def test_assign_model():
    "Test KerasNetwork.assign_model."
    test = TestKerasNetwork()
    test.nspin = 10
    # default
    test.assign_model()
    assert isinstance(test.model, keras.engine.training.Model)
    assert len(test.model.inputs) == 1
    assert test.model.inputs[0].shape[1] == 10
    assert len(test.model.layers[0].weights) == 1
    assert test.model.layers[0].weights[0].shape == (10, 10)
    assert len(test.model.layers[1].weights) == 1
    assert test.model.layers[1].weights[0].shape == (10, 10)
    assert len(test.model.layers[2].weights) == 1
    assert test.model.layers[2].weights[0].shape == (10, 1)
    assert len(test.model.outputs) == 1
    assert test.model.outputs[0].shape[1] == 1
    # custom model
    test.model = None
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(100, input_dim=10, use_bias=True))
    model.add(keras.layers.core.Dense(1, input_dim=100, use_bias=True))
    test.assign_model(model)
    assert isinstance(test.model, keras.engine.training.Model)
    assert len(test.model.inputs) == 1
    assert test.model.inputs[0].shape[1] == 10
    assert len(test.model.layers[0].weights) == 2
    assert test.model.layers[0].weights[0].shape == (10, 100)
    assert test.model.layers[0].weights[1].shape == (100,)
    assert len(test.model.layers[1].weights) == 2
    assert test.model.layers[1].weights[0].shape == (100, 1)
    assert test.model.layers[1].weights[1].shape == (1,)
    assert len(test.model.outputs) == 1
    assert test.model.outputs[0].shape[1] == 1
    # raises
    assert_raises(TypeError, test.assign_model, 1)
    assert_raises(TypeError, test.assign_model, keras.engine.network.Network())
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(1, input_dim=9, use_bias=True))
    assert_raises(ValueError, test.assign_model, model)
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(2, input_dim=10, use_bias=True))
    assert_raises(ValueError, test.assign_model, model)
    # FIXME: does not test networks with different number of sets of inputs and outputs


def test_nparams():
    "Test KerasNetwork.nparams."
    test = TestKerasNetwork()
    test.nspin = 10
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(100, input_dim=10, use_bias=True))
    model.add(keras.layers.core.Dense(1, input_dim=100, use_bias=True))
    test.assign_model(model)
    assert test.nparams == 1100 + 101


def test_params_shape():
    "Test KerasNetwork.params_shape."
    test = TestKerasNetwork()
    test.nspin = 10
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(100, input_dim=10, use_bias=True))
    model.add(keras.layers.core.Dense(1, input_dim=100, use_bias=True))
    test.assign_model(model)
    assert test.params_shape == (1100 + 101, )


def test_template_params():
    "Test KerasNetwork.template_params."
    test = TestKerasNetwork()
    params = np.random.rand(100)
    test._template_params = params
    assert np.allclose(test.template_params, params)


def test_assign_template_params():
    "Test KerasNetwork.assign_template_params."
    test = TestKerasNetwork()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.assign_model()
    test.assign_template_params()
    assert np.allclose(np.identity(10), test._template_params[:100].reshape(10, 10))
    assert np.allclose(np.identity(10), test._template_params[100:200].reshape(10, 10))
    # FIXME: need a better test to check the quality of the template params/initial guess
    hidden_sds = [slater.occ_indices(sd) for sd in sd_list(4, 5, exc_orders=[1, 2])]
    hidden_units = np.zeros((len(hidden_sds), 10))
    for i, hidden_sd in enumerate(hidden_sds):
        hidden_units[i, hidden_sd] = 1
    assert all(test._template_params[200:].dot(z) < test._template_params[200:].dot(hidden_units[0])
               for z in hidden_units[1:])
    # bad model
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(100, input_dim=10, use_bias=True))
    model.add(keras.layers.core.Dense(1, input_dim=100, use_bias=True))
    test.assign_model(model)
    assert_raises(ValueError, test.assign_template_params)
    test.nelec = 12
    assert_raises(ValueError, test.assign_template_params)
    test.nelec = 9
    assert_raises(ValueError, test.assign_template_params)


def test_assign_params():
    "Test KerasNetwork.assign_params."
    test = TestKerasNetwork()
    test.nspin = 10
    test.dtype = np.float64
    test.assign_model()
    # default
    test._template_params = np.ones(test.nparams)
    test.assign_params()
    weights = test.model.get_weights()
    assert np.allclose(weights[0], np.ones(100).reshape(10, 10))
    assert np.allclose(weights[1], np.ones(100).reshape(10, 10))
    assert np.allclose(weights[2], np.ones(10).reshape(10, 1))
    # user specified
    answer = np.random.rand(210)
    test.assign_params(answer)
    weights = test.model.get_weights()
    assert np.allclose(weights[0], answer[:100].reshape(10, 10))
    assert np.allclose(weights[1], answer[100:200].reshape(10, 10))
    assert np.allclose(weights[2], answer[200:].reshape(10, 1))


def test_get_overlap():
    "Test KerasNetwork.get_overlap."
    test = TestKerasNetwork()
    test.nspin = 4
    test.dtype = np.float64
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(4, input_dim=4, activation=keras.activations.relu,
                                      use_bias=False))
    model.add(keras.layers.core.Dense(1, input_dim=4, activation=None, use_bias=False))
    test.assign_model(model)
    matrix1 = np.random.rand(4, 4)
    matrix2 = np.random.rand(4)
    test.assign_params(np.hstack([matrix1.flatten(), matrix2]))
    # overlap
    input_vec = np.zeros(4)
    input_vec[[0, 2]] = 1
    hidden_units = input_vec[None, :].dot(matrix1)
    hidden_units[hidden_units < 0] = 0
    assert np.allclose(test.get_overlap(0b0101), hidden_units.dot(matrix2))
    input_vec = np.zeros(4)
    input_vec[[1, 2, 3]] = 1
    hidden_units = input_vec[None, :].dot(matrix1)
    hidden_units[hidden_units < 0] = 0
    assert np.allclose(test.get_overlap(0b1110), hidden_units.dot(matrix2))
    # derivative
    assert_raises(TypeError, test.get_overlap, 0b0101, 0.0)
    assert test.get_overlap(0b0101, 20) == 0
    input_vec = np.zeros(4)
    input_vec[[0, 2]] = 1
    hidden_unit = input_vec[None, :].dot(matrix1[:, 0])
    assert np.allclose(test.get_overlap(0b0101, 16), hidden_unit if hidden_unit > 0 else 0)
    der_activation = input_vec[None, :].dot(matrix1[:, 2])
    der_activation = der_activation if der_activation > 0 else 0
    assert np.allclose(test.get_overlap(0b0101, 6),
                       input_vec[1] * matrix1[1, 2] * der_activation)
