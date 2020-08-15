"""Test fanpy.wavefunction.base."""
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np

import pytest

from utils import skip_init


def test_assign_spin():
    """Test CIWavefunction.assign_spin."""
    test = skip_init(CIWavefunction)
    # check error
    with pytest.raises(TypeError):
        test.assign_spin("1")
    with pytest.raises(TypeError):
        test.assign_spin([1])
    with pytest.raises(ValueError):
        test.assign_spin(1.33)
    # None
    test.assign_spin(None)
    assert test._spin is None
    # Int assigned
    test.assign_spin(10)
    assert test._spin == 10.0
    assert isinstance(test._spin, float)
    # float assigned
    test.assign_spin(0.5)
    assert test._spin == 0.5


def test_spin():
    """Test CIWavefunction.spin."""
    test = skip_init(CIWavefunction)
    test.assign_spin(2)
    assert test.spin == 2


def test_assign_seniority():
    """Test CIWavefunction.assign_seniority."""
    test = skip_init(CIWavefunction)
    # check error
    with pytest.raises(TypeError):
        test.assign_seniority("1")
    with pytest.raises(TypeError):
        test.assign_seniority(1.0)
    with pytest.raises(ValueError):
        test.assign_seniority(-1)
    # None
    test.assign_seniority(None)
    assert test.seniority is None
    # Int assigned
    test.assign_seniority(10)
    assert test.seniority == 10


def test_seniority():
    """Test CIWavefunction.seniority."""
    test = skip_init(CIWavefunction)
    test.assign_seniority(2)
    assert test.seniority == 2


def test_assign_sds():
    """Test CIWavefunction.assign_sds."""
    test = skip_init(CIWavefunction)
    test.assign_nelec(2)
    test.assign_nspin(6)
    test.assign_spin(0)
    test.assign_seniority(0)
    # check error
    #  not iterable
    with pytest.raises(TypeError):
        test.assign_sds(2)
    #  iterable of not ints
    with pytest.raises(TypeError):
        test.assign_sds((str(i) for i in range(2)))
    with pytest.raises(TypeError):
        test.assign_sds([float(i) for i in range(2)])
    #  bad electron number
    with pytest.raises(ValueError):
        test.assign_sds([0b1, 0b111])
    #  bad spin
    test.assign_spin(0.5)
    test.assign_seniority(None)
    with pytest.raises(ValueError):
        test.assign_sds([0b000011, 0b000110])
    #  bad seniority
    test.assign_spin(None)
    test.assign_seniority(0)
    with pytest.raises(ValueError):
        test.assign_sds([0b000011, 0b000110])
    #  bad spin and seniority
    test.assign_spin(1)
    test.assign_seniority(0)
    with pytest.raises(ValueError):
        test.assign_sds([0b000011, 0b000110, 0b110000, 0b001001, 0b000101])
    #  empty list
    test.assign_spin(None)
    test.assign_seniority(None)
    with pytest.raises(ValueError):
        test.assign_sds([])

    test = skip_init(CIWavefunction)
    test.assign_nelec(2)
    test.assign_nspin(6)
    test.assign_spin(None)
    test.assign_seniority(None)
    # None assigned
    test.assign_sds()
    assert test.sds == (
        0b001001,
        0b001010,
        0b011000,
        0b001100,
        0b101000,
        0b000011,
        0b010001,
        0b000101,
        0b100001,
        0b010010,
        0b000110,
        0b100010,
        0b010100,
        0b110000,
        0b100100,
    )
    del test.sds
    test.assign_sds(None)
    assert test.sds == (
        0b001001,
        0b001010,
        0b011000,
        0b001100,
        0b101000,
        0b000011,
        0b010001,
        0b000101,
        0b100001,
        0b010010,
        0b000110,
        0b100010,
        0b010100,
        0b110000,
        0b100100,
    )
    # tuple assigned
    del test.sds
    test.assign_sds((0b0011,))
    assert test.sds == (0b0011,)
    # list assigned
    del test.sds
    test.assign_sds([0b1100])
    assert test.sds == (0b1100,)
    # generator assigned
    del test.sds
    test.assign_sds((i for i in [0b1001]))
    assert test.sds == (0b1001,)
    # repeated elements
    del test.sds
    test.assign_sds([0b0101] * 20)
    assert test.sds == (0b0101,) * 20


def test_get_overlap():
    """Test CIWavefunction.get_overlap."""
    test = CIWavefunction(2, 6, params=np.arange(15, dtype=float))
    assert test.get_overlap(0b001001, deriv=None) == 0
    assert test.get_overlap(0b001010, deriv=None) == 1
    assert test.get_overlap(0b011000, deriv=None) == 2
    assert test.get_overlap(0b001100, deriv=None) == 3
    assert test.get_overlap(0b101000, deriv=None) == 4
    assert test.get_overlap(0b000011, deriv=None) == 5
    assert test.get_overlap(0b010001, deriv=None) == 6
    assert test.get_overlap(0b000101, deriv=None) == 7
    assert test.get_overlap(0b100001, deriv=None) == 8
    assert test.get_overlap(0b010010, deriv=None) == 9
    assert test.get_overlap(0b000110, deriv=None) == 10
    assert test.get_overlap(0b100010, deriv=None) == 11
    assert test.get_overlap(0b010100, deriv=None) == 12
    assert test.get_overlap(0b110000, deriv=None) == 13
    assert test.get_overlap(0b100100, deriv=None) == 14
    for j, sd in enumerate(test.sds):
        output = np.zeros(len(test.sds))
        output[j] = 1
        assert np.allclose(test.get_overlap(sd, deriv=np.arange(15)), output)
    assert test.get_overlap(0b111111, deriv=None) == 0

    with pytest.raises(TypeError):
        test.get_overlap("1")
