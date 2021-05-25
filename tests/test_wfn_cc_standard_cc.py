"""Test fanpy.wavefunction.cc.standard_cc."""
import pytest
from fanpy.tools import slater
from fanpy.wfn.cc.standard_cc import StandardCC


class TempStandardCC(StandardCC):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_exops():
    """Test StandardCC.assign_exops."""
    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_ranks([2])
    test.assign_exops()
    assert test.exops == {(0, 1, 2, 3): 0, (0, 1, 2, 6): 1, (0, 1, 2, 7): 2, (0, 1, 3, 6): 3,
                          (0, 1, 3, 7): 4, (0, 1, 6, 7): 5, (0, 4, 2, 3): 6, (0, 4, 2, 6): 7,
                          (0, 4, 2, 7): 8, (0, 4, 3, 6): 9, (0, 4, 3, 7): 10, (0, 4, 6, 7): 11,
                          (0, 5, 2, 3): 12, (0, 5, 2, 6): 13, (0, 5, 2, 7): 14, (0, 5, 3, 6): 15,
                          (0, 5, 3, 7): 16, (0, 5, 6, 7): 17, (1, 4, 2, 3): 18, (1, 4, 2, 6): 19,
                          (1, 4, 2, 7): 20, (1, 4, 3, 6): 21, (1, 4, 3, 7): 22, (1, 4, 6, 7): 23,
                          (1, 5, 2, 3): 24, (1, 5, 2, 6): 25, (1, 5, 2, 7): 26, (1, 5, 3, 6): 27,
                          (1, 5, 3, 7): 28, (1, 5, 6, 7): 29, (4, 5, 2, 3): 30, (4, 5, 2, 6): 31,
                          (4, 5, 2, 7): 32, (4, 5, 3, 6): 33, (4, 5, 3, 7): 34, (4, 5, 6, 7): 35} 


def test_assign_refwfn():
    """Test StandardCC.assign_refwfn."""
    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    with pytest.raises(TypeError):
        test.assign_refwfn("This is not a gmpy2 instance")
    with pytest.raises(ValueError):
        test.assign_refwfn(0b00010001)
    test.assign_refwfn()
    assert test.refwfn == 0b00110011


def test_generate_possible_exops():
    """Test StandardCC.generate_possible_exops."""
    test = TempStandardCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_refwfn()
    test.assign_ranks([1])
    test.assign_exops()
    test.generate_possible_exops([0, 2], [1, 3])
    assert test.exop_combinations[(0, 2, 1, 3)] == [([0, 1], [2, 3]), ([0, 3], [2, 1])]

    test.assign_ranks([1, 2])
    test.assign_exops()
    test.generate_possible_exops([0, 2], [1, 3])
    assert test.exop_combinations[(0, 2, 1, 3)] == [([0, 1], [2, 3]), ([0, 3], [2, 1]),
                                                    ([0, 2, 1, 3], )]

    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn(0b00110011)
    test.assign_ranks([1])
    test.assign_exops()
    test.generate_possible_exops([0, 1], [2, 3])
    assert test.exop_combinations[(0, 1, 2, 3)] == [([0, 2], [1, 3]), ([0, 3], [1, 2])]
    test.generate_possible_exops([0, 1, 4], [2, 3, 6])
    assert test.exop_combinations == {
        (0, 1, 2, 3): [([0, 2], [1, 3]), ([0, 3], [1, 2])],
        (0, 1, 4, 2, 3, 6): [
            ([0, 2], [1, 3], [4, 6]), ([0, 2], [1, 6], [4, 3]), ([0, 3], [1, 2], [4, 6]),
            ([0, 3], [1, 6], [4, 2]), ([0, 6], [1, 2], [4, 3]), ([0, 6], [1, 3], [4, 2]),
        ]
    }

    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn(0b00110011)
    test.assign_ranks([1, 2])
    test.assign_exops()
    test.generate_possible_exops([0, 1], [2, 3])
    assert test.exop_combinations == {
        (0, 1, 2, 3): [([0, 2], [1, 3]), ([0, 3], [1, 2]), ([0, 1, 2, 3],)]
    }
    test.generate_possible_exops([0, 1, 4], [2, 3, 6])
    assert test.exop_combinations == {
        (0, 1, 2, 3): [([0, 2], [1, 3]), ([0, 3], [1, 2]), ([0, 1, 2, 3],)],
        (0, 1, 4, 2, 3, 6): [
            ([0, 2], [1, 3], [4, 6]), ([0, 2], [1, 6], [4, 3]), ([0, 3], [1, 2], [4, 6]),
            ([0, 3], [1, 6], [4, 2]), ([0, 6], [1, 2], [4, 3]), ([0, 6], [1, 3], [4, 2]),
            ([0, 2], [1, 4, 3, 6]), ([0, 3], [1, 4, 2, 6]), ([0, 6], [1, 4, 2, 3]),
            ([1, 2], [0, 4, 3, 6]), ([1, 3], [0, 4, 2, 6]), ([1, 6], [0, 4, 2, 3]),
            ([4, 2], [0, 1, 3, 6]), ([4, 3], [0, 1, 2, 6]), ([4, 6], [0, 1, 2, 3]),
        ]
    }

    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn(0b00110011)
    test.assign_ranks([1, 2, 3])
    test.assign_exops()
    test.generate_possible_exops([0, 1], [2, 3])
    assert test.exop_combinations == {
        (0, 1, 2, 3): [([0, 2], [1, 3]), ([0, 3], [1, 2]), ([0, 1, 2, 3],)]
    }
    test.generate_possible_exops([0, 1, 4], [2, 3, 6])
    assert test.exop_combinations == {
        (0, 1, 2, 3): [([0, 2], [1, 3]), ([0, 3], [1, 2]), ([0, 1, 2, 3],)],
        (0, 1, 4, 2, 3, 6): [
            ([0, 2], [1, 3], [4, 6]), ([0, 2], [1, 6], [4, 3]), ([0, 3], [1, 2], [4, 6]),
            ([0, 3], [1, 6], [4, 2]), ([0, 6], [1, 2], [4, 3]), ([0, 6], [1, 3], [4, 2]),
            ([0, 2], [1, 4, 3, 6]), ([0, 3], [1, 4, 2, 6]), ([0, 6], [1, 4, 2, 3]),
            ([1, 2], [0, 4, 3, 6]), ([1, 3], [0, 4, 2, 6]), ([1, 6], [0, 4, 2, 3]),
            ([4, 2], [0, 1, 3, 6]), ([4, 3], [0, 1, 2, 6]), ([4, 6], [0, 1, 2, 3]),
            ([0, 1, 4, 2, 3, 6],),
        ]
    }
