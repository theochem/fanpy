"""Test wfns.wavefunction.wavefunctions."""
import cachetools

import numpy as np
import pytest
from utils import disable_abstract, skip_init
from wfns.wfn.base import BaseWavefunction


def test_assign_nelec():
    """Test BaseWavefunction.assign_nelec."""
    test = skip_init(disable_abstract(BaseWavefunction))
    # check errors
    with pytest.raises(TypeError):
        test.assign_nelec(None)
    with pytest.raises(TypeError):
        test.assign_nelec(2.0)
    with pytest.raises(TypeError):
        test.assign_nelec("2")
    with pytest.raises(ValueError):
        test.assign_nelec(0)
    with pytest.raises(ValueError):
        test.assign_nelec(-2)
    # int
    test.assign_nelec(2)
    assert test.nelec == 2


def test_nspin():
    """Test BaseWavefunction.nspin."""
    test = skip_init(disable_abstract(BaseWavefunction))
    # check errors
    with pytest.raises(TypeError):
        test.assign_nspin(None)
    with pytest.raises(TypeError):
        test.assign_nspin(2.0)
    with pytest.raises(TypeError):
        test.assign_nspin("2")
    with pytest.raises(ValueError):
        test.assign_nspin(0)
    with pytest.raises(ValueError):
        test.assign_nspin(-2)
    with pytest.raises(NotImplementedError):
        test.assign_nspin(3)
    # int
    test.assign_nspin(2)
    assert test.nspin == 2


def test_assign_memory():
    """Test BaseWavefunction.assign_memory."""
    test = skip_init(disable_abstract(BaseWavefunction))
    test.assign_memory(None)
    assert test.memory == np.inf
    test.assign_memory(10)
    assert test.memory == 10
    test.assign_memory(20.0)
    assert test.memory == 20
    test.assign_memory("10mb")
    assert test.memory == 1e6 * 10
    test.assign_memory("20.1gb")
    assert test.memory == 1e9 * 20.1
    with pytest.raises(TypeError):
        test.assign_memory([])
    with pytest.raises(ValueError):
        test.assign_memory("20.1kb")


def temp_assign_params(self, params=None, add_noise=False, default_params=np.identity(10)):
    """Assign the parameters of the wavefunction."""
    if params is None:
        params = default_params

    super().assign_params(params=params, add_noise=add_noise)


def test_assign_params():
    """Test BaseWavefunction.assign_params."""
    # default
    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "assign_params": temp_assign_params,
            },
        )
    )
    test.assign_params()
    assert np.allclose(test.params, np.identity(10))

    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "assign_params": temp_assign_params,
            },
        )
    )
    test.assign_params()
    assert np.allclose(test.params, np.identity(10))

    # check errors
    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "assign_params": temp_assign_params,
            },
        )
    )

    # check noise
    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "assign_params": temp_assign_params,
            },
        )
    )
    test.assign_params(add_noise=True)
    assert np.all(np.abs(test.params - np.identity(10)) <= 0.2 / 100)
    assert not np.allclose(test.params, np.identity(10))

    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "assign_params": temp_assign_params,
            },
        )
    )
    test.assign_params(params=np.identity(10).astype(complex), add_noise=True)
    assert np.all(np.abs(np.real(test.params - np.identity(10))) <= 0.1 / 100)
    assert np.all(np.abs(np.imag(test.params - np.identity(10))) <= 0.01 * 0.1 / 100)
    assert not np.allclose(np.real(test.params), np.identity(10))
    assert not np.allclose(np.imag(test.params), np.zeros((10, 10)))

    # for testing one line of code
    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "assign_params": lambda self, params, add_noise=False:
                temp_assign_params(self, params, add_noise, np.zeros((1, 1, 1))),
            },
        )
    )
    test.assign_params(2.0)
    assert test.params.shape == (1, 1, 1)


def test_init():
    """Test BaseWavefunction.__init__."""
    test = skip_init(disable_abstract(BaseWavefunction))
    BaseWavefunction.__init__(test, 2, 10, memory=20)
    assert test.nelec == 2
    assert test.nspin == 10
    assert test.memory == 20


def test_olp():
    """Test BaseWavefunction._olp."""
    test = skip_init(disable_abstract(BaseWavefunction))
    with pytest.raises(NotImplementedError):
        test._olp(0b0101)


def test_olp_deriv():
    """Test BaseWavefunction._olp_deriv."""
    test = skip_init(disable_abstract(BaseWavefunction))
    with pytest.raises(NotImplementedError):
        test._olp_deriv(0b0101, 0)


def test_load_cache():
    """Test BaseWavefunction.load_cache."""
    test = skip_init(
        disable_abstract(BaseWavefunction)
    )
    test.memory = 1000
    test.params = np.array([1, 2, 3])
    test._cache_fns = {}
    test.load_cache()
    assert hasattr(test, "_cache_fns")
    assert isinstance(test._cache_fns["overlap"], cachetools.LRUCache)
    assert isinstance(test._cache_fns["overlap derivative"], cachetools.LRUCache)

    test.memory = np.inf
    test.load_cache()
    assert test._cache_fns["overlap"].maxsize == 2**30


def test_clear_cache():
    """Test BaseWavefunction.clear_cache."""
    test = skip_init(disable_abstract(BaseWavefunction))
    with pytest.raises(AttributeError):
        test.clear_cache()

    cache_one = cachetools.LRUCache(2)
    cache_two = cachetools.LRUCache(2)

    @cachetools.cachedmethod(cache=lambda obj: cache_one)
    def olp(self, sd):
        """Overlap of wavefunction."""
        return 0.0

    test._cache_fns = {}
    test._cache_fns["overlap"] = cache_one
    with pytest.raises(KeyError):
        test.clear_cache("overlap derivative")

    @cachetools.cachedmethod(cache=lambda obj: cache_two)
    def olp_deriv(self, sd, deriv):
        """Return the derivative of the overlap of wavefunction."""
        return 0.0

    test._cache_fns["overlap derivative"] = cache_two

    olp(None, 2)
    olp(None, 3)
    olp_deriv(None, 2, 0)
    olp_deriv(None, 3, 0)
    assert cache_one.currsize == 2
    assert cache_two.currsize == 2
    test.clear_cache("overlap")
    assert cache_one.currsize == 0
    assert cache_two.currsize == 2
    test.clear_cache()
    assert cache_one.currsize == 0
    assert cache_two.currsize == 0


def test_nspatial():
    """Test BaseWavefunction.nspatial."""
    test = skip_init(disable_abstract(BaseWavefunction))
    test.assign_nspin(10)
    assert test.nspatial == 5


def test_nparams():
    """Test BaseWavefunction.nparams."""
    test = skip_init(
        disable_abstract(BaseWavefunction)
    )
    test.params = np.arange(100)
    assert test.nparams == 100


def test_spin():
    """Test BaseWavefunction.spin."""
    test = skip_init(disable_abstract(BaseWavefunction))
    assert test.spin is None


def test_seniority():
    """Test BaseWavefunction.seniority."""
    test = skip_init(disable_abstract(BaseWavefunction))
    assert test.seniority is None
