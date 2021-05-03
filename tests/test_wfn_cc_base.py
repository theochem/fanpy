"""Test fanpy.wavefunction.cc.cc_wavefunction."""
import numdifftools as nd

import numpy as np

import pytest

from fanpy.tools import slater
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.cc.base import BaseCC


class TempBaseCC(BaseCC):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_ranks():
    """Test CCWavefunction.assign_ranks."""
    test = TempBaseCC()
    # check method
    test.assign_nelec(3)
    test.assign_ranks()
    assert test.ranks == [1, 2, 3]
    test.assign_ranks(2)
    assert test.ranks == [1, 2]
    test.assign_ranks([1, 2, 3])
    assert test.ranks == [1, 2, 3]
    # check errors
    with pytest.raises(TypeError):
        test.assign_ranks("Not an int or list of ints")
    with pytest.raises(TypeError):
        test.assign_ranks([1.0, 2.0])
    with pytest.raises(ValueError):
        test.assign_ranks(-4)
    with pytest.raises(ValueError):
        test.assign_ranks(10)


def test_assign_exops():
    """Test CCWavefunction.assign_exops."""
    test = TempBaseCC()
    # check errors
    with pytest.raises(TypeError):
        test.assign_exops("This is not a list")
    with pytest.raises(TypeError):
        test.assign_exops([[1, 2], "This is not a list"])
    with pytest.raises(TypeError):
        test.assign_exops([[1, 2], [3, 4, 5], [6, 7, 8, 9]])
    with pytest.raises(TypeError):
        test.assign_exops([[1, 2], [3.0, 4]])
    with pytest.raises(ValueError):
        test.assign_exops([[1, 2], [3, -4]])
    # check method
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_ranks()
    test.assign_exops()
    exops = [
        [0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1],
        [3, 2], [0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2], [1, 2, 0, 3], [1, 3, 0, 2], [2, 3, 0, 1],
    ]
    assert test.exops == {tuple(exop) : ind for ind, exop in enumerate(exops)}
    test.assign_exops([[0, 2], [1, 3]])
    exops = [[0, 1], [0, 3], [2, 1], [2, 3], [0, 2, 1, 3]]
    assert test.exops == {tuple(exop) : ind for ind, exop in enumerate(exops)}


def test_assign_refwfn():
    """Test CCWavefunction.assign_refwfn."""
    # check method
    test = TempBaseCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_refwfn()
    assert test.refwfn == slater.ground(nocc=test.nelec, norbs=test.nspin)
    ci_test = CIWavefunction(nelec=2, nspin=4, spin=0, seniority=0)
    test.assign_refwfn(ci_test)
    assert test.refwfn.nelec == 2
    assert test.refwfn.nspin == 4
    assert test.refwfn.spin == 0
    assert test.refwfn.seniority == 0
    # FIXME: check if sds is correct
    slater_test = 0b0101
    test.assign_refwfn(slater_test)
    assert test.refwfn == 0b0101
    # check errors
    with pytest.raises(TypeError):
        test.assign_refwfn("This is not a gmpy2 or CIwfn object")
    # with pytest.raises(AttributeError):
    #     test.assign_refwfn("This doesn't have a sd_vec attribute")
    with pytest.raises(ValueError):
        test.assign_refwfn(0b1111)
    # with pytest.raises(ValueError):
    #     test.assign_refwfn(0b001001)


def test_template_params():
    """Test CCWavefunction.template_params."""
    test = TempBaseCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_ranks()
    test.assign_exops()
    np.allclose(test.template_params, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))


def test_assign_params():
    """Test CCWavefunction.assign_params."""
    test = TempBaseCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_ranks()
    test.assign_exops()
    test.assign_params()
    np.allclose(test.params, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    test2 = TempBaseCC()
    test2.assign_nelec(4)
    with pytest.raises(ValueError):
        test.assign_params(test2)
    test2.assign_nelec(2)
    test2.assign_nspin(6)
    with pytest.raises(ValueError):
        test.assign_params(test2)
    test2.assign_nspin(4)
    test2.assign_ranks()
    test2.assign_exops([[0, 1], [2, 3]])
    with pytest.raises(ValueError):
        test.assign_params(test2)
    test2.assign_nelec(2)
    test2.assign_nspin(4)
    test2.assign_ranks()
    test2.assign_exops()
    test2.assign_params()
    test2.params = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    test.assign_params(test2)
    np.allclose(test.params, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]))


def test_get_ind():
    """Test CCWavefunction.get_ind."""
    test = TempBaseCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_ranks()
    test.assign_exops()
    assert test.get_ind([0, 1]) == 0
    # check errors
    with pytest.raises(ValueError):
        test.get_ind([0, 1, 2, 3, 4, 5])


def test_get_exop():
    """Test CCWavefunction.get_exop."""
    test = TempBaseCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_ranks()
    test.assign_exops()
    assert test.get_exop(0) == (0, 1)


def test_product_amplitudes():
    """Test CCWavefunction.product_amplitudes."""
    test = TempBaseCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_ranks()
    test.assign_exops()
    test.assign_params(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]))
    assert test.product_amplitudes([0, 1, 2]) == 6
    answer = np.zeros(18)
    answer[0] = 2 * 3
    answer[1] = 1 * 3
    answer[2] = 1 * 2
    assert np.allclose(test.product_amplitudes([0, 1, 2], True), answer)


def test_olp_deriv():
    """Test BaseCC._olp_deriv."""
    test = BaseCC(4, 10, )
    test.assign_params(np.random.rand(test.nparams))
    for sd in np.random.choice(sd_list(4, 10), 5):
        def func(x):
            test.assign_params(x)
            return test._olp(sd, slater.ground(4, 10))

        grad = nd.Gradient(func, step=1e-4)(test.params)
        assert np.allclose(test._olp_deriv(sd, slater.ground(4, 10)), grad, atol=1e-4)


def test_get_overlap():
    """Test CCWavefunction.get_overlap."""
    test = TempBaseCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_ranks()
    test.assign_exops()
    test.assign_refwfn()
    test.assign_memory("1gb")
    test.assign_params(np.arange(1, 19))
    test.load_cache()
    # FIXME: WRONG NUMBERS
    # assert test.get_overlap(0b1010) == 1*4 + 2*3 + 5
    # assert test.get_overlap(0b1010, 4) == 1*4 + 2*3
    assert test.get_overlap(0b1010) == 1*9 - 3*8 - 14
    assert np.allclose(
        test.get_overlap(0b1010, True),
        np.array([9, 0, -8, 0, 0, 0, 0, -3, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0]),
    )


def test_generate_possible_exops():
    """Test CCWavefunction.generate_possible_exops."""
    test = TempBaseCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_ranks()
    test.assign_exops()
    test.generate_possible_exops([0, 2], [1, 3])
    assert test.exop_combinations[(0, 2, 1, 3)] == [([0, 1], [2, 3]), ([0, 3], [2, 1]),
                                                    ([0, 2, 1, 3], )]
