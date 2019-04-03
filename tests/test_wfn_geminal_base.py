"""Test wfns.wavefunction.geminal.gem_wavefunction."""
import pytest
import numpy as np
from wfns.wfn.geminal.base import BaseGeminal
from utils import skip_init, disable_abstract


class TempBaseGeminal(BaseGeminal):
    """GeminalWavefunction that skips initialization."""

    def __init__(self):
        self._cache_fns = {}

    def generate_possible_orbpairs(self, occ_indices):
        if occ_indices == (0, 1, 2, 3):
            yield ((0, 1), (2, 3)), 1
        else:
            yield (), 1


def test_gem_assign_nelec():
    """Test GeminalWavefunction.assign_nelec."""
    test = skip_init(disable_abstract(BaseGeminal))
    # int
    test.assign_nelec(2)
    assert test.nelec == 2
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
    with pytest.raises(ValueError):
        test.assign_nelec(1)
    with pytest.raises(ValueError):
        test.assign_nelec(3)


def test_gem_spin():
    """Test GeminalWavefunction.spin."""
    test = skip_init(disable_abstract(BaseGeminal))
    assert test.spin is None


def test_gem_seniority():
    """Test GeminalWavefunction.seniority."""
    test = skip_init(disable_abstract(BaseGeminal))
    assert test.seniority is None


def test_gem_npair():
    """Test GeminalWavefunction.npair."""
    test = skip_init(disable_abstract(BaseGeminal))
    test.assign_nelec(4)
    assert test.npair == 2


def test_gem_assign_ngem():
    """Test GeminalWavefunction.assign_ngem."""
    test = skip_init(disable_abstract(BaseGeminal))
    test.assign_nelec(4)
    test.assign_ngem(2)
    assert test.ngem == 2
    test.assign_ngem(3)
    assert test.ngem == 3
    # check errors
    with pytest.raises(TypeError):
        test.assign_ngem(2.0)
    with pytest.raises(TypeError):
        test.assign_ngem("2")
    with pytest.raises(ValueError):
        test.assign_ngem(0)
    with pytest.raises(ValueError):
        test.assign_ngem(1)
    with pytest.raises(ValueError):
        test.assign_ngem(-2)


def test_gem_assign_orbpair():
    """Test BaseGeminal.assign_orbpair."""
    test = skip_init(disable_abstract(BaseGeminal))
    # default
    test.assign_nspin(6)
    test.assign_orbpairs()
    assert test.dict_orbpair_ind == {
        (0, 1): 0,
        (0, 2): 1,
        (0, 3): 2,
        (0, 4): 3,
        (0, 5): 4,
        (1, 2): 5,
        (1, 3): 6,
        (1, 4): 7,
        (1, 5): 8,
        (2, 3): 9,
        (2, 4): 10,
        (2, 5): 11,
        (3, 4): 12,
        (3, 5): 13,
        (4, 5): 14,
    }

    # not iterable
    with pytest.raises(TypeError):
        test.assign_orbpairs(3)
    with pytest.raises(TypeError):
        test.assign_orbpairs(True)
    with pytest.raises(TypeError):
        test.assign_orbpairs(("1,2", (1, 2)))
    with pytest.raises(TypeError):
        test.assign_orbpairs(((0, 1), (1, 2, 3)))
    with pytest.raises(TypeError):
        test.assign_orbpairs(((0, 1), (1, 2.0)))
    with pytest.raises(ValueError):
        test.assign_orbpairs(((0, 1), (1, 1)))
    with pytest.raises(ValueError):
        test.assign_orbpairs(((0, 1), (1, 0)))
    # generator of tuple
    test.assign_orbpairs(((i, i + 3) for i in range(3)))
    assert test.dict_orbpair_ind == {(0, 3): 0, (1, 4): 1, (2, 5): 2}
    # list of tuple
    test.assign_orbpairs([(i, i + 3) for i in range(3)])
    assert test.dict_orbpair_ind == {(0, 3): 0, (1, 4): 1, (2, 5): 2}
    # tuple of tuple
    test.assign_orbpairs(tuple((i, i + 3) for i in range(3)))
    assert test.dict_orbpair_ind == {(0, 3): 0, (1, 4): 1, (2, 5): 2}
    # generator of list
    test.assign_orbpairs(([i, i + 3] for i in range(3)))
    assert test.dict_orbpair_ind == {(0, 3): 0, (1, 4): 1, (2, 5): 2}
    # list of list
    test.assign_orbpairs([[i, i + 3] for i in range(3)])
    assert test.dict_orbpair_ind == {(0, 3): 0, (1, 4): 1, (2, 5): 2}
    # tuple of list
    test.assign_orbpairs(tuple([i, i + 3] for i in range(3)))
    assert test.dict_orbpair_ind == {(0, 3): 0, (1, 4): 1, (2, 5): 2}
    # generator of tuple unordered
    test.assign_orbpairs(((i + 3, i) for i in range(3)))
    assert test.dict_orbpair_ind == {(0, 3): 0, (1, 4): 1, (2, 5): 2}
    # list of tuple unordered
    test.assign_orbpairs([(i + 3, i) for i in range(3)])
    assert test.dict_orbpair_ind == {(0, 3): 0, (1, 4): 1, (2, 5): 2}
    # tuple of tuple unordered
    test.assign_orbpairs(tuple((i + 3, i) for i in range(3)))
    assert test.dict_orbpair_ind == {(0, 3): 0, (1, 4): 1, (2, 5): 2}


def test_gem_norbpair():
    """Test BaseGeminal.norbpair."""
    test = skip_init(disable_abstract(BaseGeminal))
    test.assign_nspin(6)
    test.assign_orbpairs()
    assert test.norbpair == 15


def test_gem_template_params():
    """Test BaseGeminal.template_params."""
    test = skip_init(disable_abstract(BaseGeminal))
    test.assign_dtype(float)
    test.assign_nelec(6)
    test.assign_nspin(6)
    test.assign_orbpairs()
    test.assign_ngem(3)
    np.allclose(
        test.template_params,
        np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ),
    )


def test_gem_assign_params():
    """Test BaseGeminal.assign_params."""
    test = skip_init(disable_abstract(BaseGeminal))
    test.assign_dtype(float)
    test.assign_nelec(6)
    test.assign_nspin(6)
    test.assign_orbpairs()
    test.assign_ngem(3)
    # default
    test.assign_params()
    np.allclose(
        test.params,
        np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ),
    )

    test2 = skip_init(disable_abstract(BaseGeminal))
    test2.assign_dtype(float)
    test2.assign_nelec(4)
    with pytest.raises(ValueError):
        test.assign_params(test2)
    test2.assign_nelec(6)
    test2.assign_nspin(8)
    with pytest.raises(ValueError):
        test.assign_params(test2)
    test2.assign_nelec(6)
    test2.assign_nspin(6)
    test2.assign_ngem(4)
    with pytest.raises(ValueError):
        test.assign_params(test2)
    test2.assign_nelec(6)
    test2.assign_nspin(6)
    test2.assign_ngem(3)
    test2.dict_ind_orbpair = {0: (0, 4), 1: (1, 3)}
    test2.params = np.arange(6).reshape(3, 2)
    test.assign_params(test2)
    np.allclose(
        test.params,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ),
    )

    test.assign_orbpairs(
        {
            (0, 1): 0,
            (0, 2): 1,
            (0, 3): 2,
            (0, 4): 3,
            (0, 5): 4,
            (1, 2): 5,
            (1, 4): 6,
            (1, 5): 7,
            (2, 3): 8,
            (2, 4): 9,
            (2, 5): 10,
            (3, 4): 11,
            (3, 5): 12,
            (4, 5): 13,
        }
    )
    test.assign_params()
    test.assign_params(test2)
    np.allclose(
        test.params,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ),
    )


def test_gem_get_col_ind():
    """Test BaseGeminal.get_col_ind."""
    test = skip_init(disable_abstract(BaseGeminal))
    test.dict_orbpair_ind = {(0, 1): 0, (1, 5): 6}
    assert test.get_col_ind((0, 1)) == 0
    assert test.get_col_ind((1, 5)) == 6
    with pytest.raises(ValueError):
        test.get_col_ind([0, 1])
    with pytest.raises(ValueError):
        test.get_col_ind((0, 2))


def test_gem_get_orbpair():
    """Test BaseGeminal.get_orbpair."""
    test = skip_init(disable_abstract(BaseGeminal))
    test.dict_ind_orbpair = {0: (0, 1), 6: (1, 5)}
    assert test.get_orbpair(0) == (0, 1)
    assert test.get_orbpair(6) == (1, 5)
    with pytest.raises(ValueError):
        test.get_orbpair(1)
    with pytest.raises(ValueError):
        test.get_orbpair("0")


def test_gem_compute_permanent():
    """Test BaseGeminal.compute_permanent."""
    test = skip_init(disable_abstract(BaseGeminal))
    test.assign_dtype(float)
    test.assign_nelec(6)
    test.assign_nspin(6)
    test.assign_orbpairs()
    test.assign_ngem(3)
    test.assign_params(np.arange(45, dtype=float).reshape(3, 15))
    assert np.equal(
        test.compute_permanent([0, 1, 2]),
        0 * (16 * 32 + 17 * 31) + 1 * (15 * 32 + 17 * 30) + 2 * (15 * 31 + 16 * 30),
    )
    assert np.equal(test.compute_permanent((0, 1, 2), deriv=0), 16 * 32 + 17 * 31)
    assert np.equal(test.compute_permanent((0, 1, 2), deriv=16), 0 * 32 + 2 * 30)
    assert np.equal(test.compute_permanent((0, 1, 2), deriv=34), 0)
    assert np.equal(test.compute_permanent((0, 1, 2), deriv=9999), 0)
    assert np.equal(
        test.compute_permanent((3, 4, 5)),
        3 * (19 * 35 + 34 * 20) + 4 * (18 * 35 + 33 * 20) + 5 * (18 * 34 + 33 * 19),
    )
    assert np.equal(test.compute_permanent((3, 4, 5), deriv=35), 3 * 19 + 18 * 4)
    # row inds
    assert np.equal(test.compute_permanent(col_inds=(1, 2), row_inds=[0, 1]), 1 * 17 + 16 * 2)
    # ryser
    assert np.equal(
        test.compute_permanent(col_inds=(0, 1, 2, 3), row_inds=[0, 1]),
        (0 * 16 + 15 * 1)
        + (0 * 17 + 15 * 2)
        + (0 * 18 + 15 * 3)
        + (1 * 17 + 2 * 16)
        + (1 * 18 + 3 * 16)
        + (2 * 18 + 17 * 3),
    )
    # one by one matrix derivatized
    assert np.equal(test.compute_permanent(col_inds=(0,), row_inds=(0,), deriv=0), 1)


def test_gem_compute_permanent_deriv():
    """Test derivatives of BaseGeminal.compute_permanent using finite difference."""
    test = skip_init(disable_abstract(BaseGeminal))
    test.assign_dtype(float)
    test.assign_nelec(6)
    test.assign_nspin(6)
    test.assign_memory(1000)
    test.assign_orbpairs()
    test.assign_ngem(3)
    test.assign_params(np.random.rand(3, 15) * 10)
    test._cache_fns = {}
    test.load_cache()

    original_params = test.params[:]
    delta = np.zeros(test.params_shape)
    step = 1e-3
    for i in range(test.params_shape[0]):
        for j in range(test.params_shape[1]):
            delta *= 0
            delta[i, j] = step
            test.assign_params(original_params)
            test.clear_cache()
            one = test.compute_permanent(range(15))
            test.assign_params(test.params + delta)
            test.clear_cache()
            two = test.compute_permanent(range(15))
            # FIXME: not quite sure why the difference between permanent derivative and finite
            #        difference increases as the step size decreases
            assert np.allclose(
                test.compute_permanent(range(15), deriv=15 * i + j),
                (two - one) / step,
                rtol=0,
                atol=1e-5,
            )


def test_gem_get_overlap():
    """Test BaseGeminal.get_overlap."""
    test = skip_init(
        disable_abstract(
            BaseGeminal,
            dict_overwrite={
                "generate_possible_orbpairs": lambda self, occ_indices: [
                    (((0, 1), (2, 3)), 1) if occ_indices == (0, 1, 2, 3) else ((), 1)
                ]
            },
        )
    )
    test.assign_dtype(float)
    test.assign_nelec(4)
    test.assign_nspin(6)
    test.assign_memory()
    test.assign_orbpairs()
    test.assign_ngem(3)
    test.assign_params(np.arange(45, dtype=float).reshape(3, 15))
    test._cache_fns = {}
    test.load_cache()
    assert test.get_overlap(0b001111) == 9 * (15 * 1 + 30 * 1) + 1 * (15 * 39 + 30 * 24)
    assert test.get_overlap(0b000111) == 0
    # check derivatives
    test.assign_params(np.arange(45, dtype=float).reshape(3, 15))
    assert test.get_overlap(0b001111, deriv=0) == 24 * 1 + 39 * 1
    assert test.get_overlap(0b001111, deriv=1) == 0
    assert test.get_overlap(0b001111, deriv=9) == 15 * 1 + 30 * 1
    assert test.get_overlap(0b001111, deriv=15) == 9 * 1 + 39 * 1
    assert test.get_overlap(0b001111, deriv=39) == 0 * 1 + 15 * 1
    assert test.get_overlap(0b000111, deriv=0) == 0
    assert test.get_overlap(0b001111, deriv=45) == 0
    assert test.get_overlap(0b001111, deriv=3) == 0
    with pytest.raises(TypeError):
        test.get_overlap(0b001111, "1")
