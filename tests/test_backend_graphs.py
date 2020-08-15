"""Test fors wfn.graphs."""
import numpy as np

from fanpy.tools.graphs import (
    generate_biclique_pmatch,
    generate_complete_pmatch,
    generate_general_pmatch,
    generate_unordered_partition,
    int_partition_recursive,
)
from fanpy.tools.slater import sign_perm


def test_generate_complete_pmatch():
    """Test wfn.tools.graphs.generate_complete_pmatch."""
    # bad input
    assert list(generate_complete_pmatch([])) == [((), 1)]
    assert list(generate_complete_pmatch([1])) == [((), 1)]
    assert list(generate_complete_pmatch([1, 2, 3])) == [((), 1)]

    # 4 vertices
    occ_indices = [0, 1, 3, 4]
    answer = [(((0, 1), (3, 4)), 1), (((0, 3), (1, 4)), -1), (((0, 4), (1, 3)), 1)]
    assert answer == list(generate_complete_pmatch(occ_indices))

    # 6 vertices
    occ_indices = [0, 1, 3, 4, 6, 7]
    answer = [
        (((0, 1), (3, 4), (6, 7)), 1),
        (((0, 1), (3, 6), (4, 7)), -1),
        (((0, 1), (3, 7), (4, 6)), 1),
        (((0, 3), (1, 4), (6, 7)), -1),
        (((0, 3), (1, 6), (4, 7)), 1),
        (((0, 3), (1, 7), (4, 6)), -1),
        (((0, 4), (1, 3), (6, 7)), 1),
        (((0, 4), (1, 6), (3, 7)), -1),
        (((0, 4), (1, 7), (3, 6)), 1),
        (((0, 6), (1, 7), (3, 4)), -1),
        (((0, 6), (1, 4), (3, 7)), 1),
        (((0, 6), (1, 3), (4, 7)), -1),
        (((0, 7), (1, 6), (3, 4)), 1),
        (((0, 7), (1, 4), (3, 6)), -1),
        (((0, 7), (1, 3), (4, 6)), 1),
    ]
    assert answer == [
        (tuple(sorted(i, key=lambda x: x[0])), sign)
        for i, sign in sorted(generate_complete_pmatch(occ_indices), key=lambda x: x[0])
    ]

    # check sign
    occ_indices = [0, 1, 3, 4, 6, 7, 9, 10]
    for pairing_scheme, sign in generate_complete_pmatch(occ_indices):
        jumbled_indices = [j for i in pairing_scheme for j in i]
        assert sign == sign_perm(jumbled_indices, occ_indices, is_decreasing=False)
    occ_indices = [0, 1, 3, 4, 6, 7]
    for pairing_scheme, sign in generate_complete_pmatch(occ_indices):
        jumbled_indices = [j for i in pairing_scheme for j in i]
        assert sign == sign_perm(jumbled_indices, occ_indices, is_decreasing=False)


def test_generate_biclique_pmatch():
    """Test wfn.tools.graphs.generate_biclique_pmatch."""
    # bad input
    assert list(generate_biclique_pmatch([], [])) == [((), 1)]
    assert list(generate_biclique_pmatch([], [1, 2])) == [((), 1)]
    assert list(generate_biclique_pmatch([1, 2], [2, 3])) == [((), 1)]
    assert list(generate_biclique_pmatch([1, 2], [3])) == [((), 1)]

    # 4 vertices
    indices_one = [0, 1]
    indices_two = [3, 4]
    answer = [(((0, 3), (1, 4)), -1), (((0, 4), (1, 3)), 1)]
    assert answer == list(generate_biclique_pmatch(indices_one, indices_two))
    ordered_indices = sorted(indices_one + indices_two)
    for pairing_scheme, sign in generate_biclique_pmatch(indices_one, indices_two):
        jumbled_indices = [j for i in pairing_scheme for j in i]
        assert sign == sign_perm(jumbled_indices, ordered_indices, is_decreasing=False)

    # 6 vertices
    indices_one = [0, 1, 3]
    indices_two = [4, 6, 7]
    answer = [
        (((0, 4), (1, 6), (3, 7)), -1),
        (((0, 4), (1, 7), (3, 6)), 1),
        (((0, 6), (1, 4), (3, 7)), 1),
        (((0, 6), (1, 7), (3, 4)), -1),
        (((0, 7), (1, 4), (3, 6)), -1),
        (((0, 7), (1, 6), (3, 4)), 1),
    ]
    assert answer == list(generate_biclique_pmatch(indices_one, indices_two))
    ordered_indices = sorted(indices_one + indices_two)
    for pairing_scheme, sign in generate_biclique_pmatch(indices_one, indices_two):
        jumbled_indices = [j for i in pairing_scheme for j in i]
        assert sign == sign_perm(jumbled_indices, ordered_indices, is_decreasing=False)

    # 8 vertices
    indices_one = [0, 1, 3, 5]
    indices_two = [4, 6, 7, 8]
    answer = [
        (((0, 4), (1, 6), (3, 7), (5, 8)), -1),
        (((0, 4), (1, 6), (3, 8), (5, 7)), 1),
        (((0, 4), (1, 7), (3, 6), (5, 8)), 1),
        (((0, 4), (1, 7), (3, 8), (5, 6)), -1),
        (((0, 4), (1, 8), (3, 6), (5, 7)), -1),
        (((0, 4), (1, 8), (3, 7), (5, 6)), 1),
        (((0, 6), (1, 4), (3, 7), (5, 8)), 1),
        (((0, 6), (1, 4), (3, 8), (5, 7)), -1),
        (((0, 6), (1, 7), (3, 4), (5, 8)), -1),
        (((0, 6), (1, 7), (3, 8), (5, 4)), 1),
        (((0, 6), (1, 8), (3, 4), (5, 7)), 1),
        (((0, 6), (1, 8), (3, 7), (5, 4)), -1),
        (((0, 7), (1, 4), (3, 6), (5, 8)), -1),
        (((0, 7), (1, 4), (3, 8), (5, 6)), 1),
        (((0, 7), (1, 6), (3, 4), (5, 8)), 1),
        (((0, 7), (1, 6), (3, 8), (5, 4)), -1),
        (((0, 7), (1, 8), (3, 4), (5, 6)), -1),
        (((0, 7), (1, 8), (3, 6), (5, 4)), 1),
        (((0, 8), (1, 4), (3, 6), (5, 7)), 1),
        (((0, 8), (1, 4), (3, 7), (5, 6)), -1),
        (((0, 8), (1, 6), (3, 4), (5, 7)), -1),
        (((0, 8), (1, 6), (3, 7), (5, 4)), 1),
        (((0, 8), (1, 7), (3, 4), (5, 6)), 1),
        (((0, 8), (1, 7), (3, 6), (5, 4)), -1),
    ]
    assert answer == list(generate_biclique_pmatch(indices_one, indices_two))
    ordered_indices = sorted(indices_one + indices_two)
    for pairing_scheme, sign in generate_biclique_pmatch(indices_one, indices_two):
        jumbled_indices = [j for i in pairing_scheme for j in i]
        assert sign == sign_perm(jumbled_indices, ordered_indices, is_decreasing=False)


def test_unorderd_partition():
    """Test wfn.tools.graphs.generate_unordered_partition."""
    assert list(generate_unordered_partition([1], [(1, 1)])) == [[[1]]]

    assert list(generate_unordered_partition([1, 2], [(2, 1)])) == [[[1, 2]]]
    assert list(generate_unordered_partition([1, 2], [(1, 2)])) == [[[1], [2]]]

    assert list(generate_unordered_partition([1, 2, 3], [(3, 1)])) == [[[1, 2, 3]]]
    assert list(generate_unordered_partition([1, 2, 3], [(2, 1), (1, 1)])) == [
        [[1, 2], [3]],
        [[1, 3], [2]],
        [[2, 3], [1]],
    ]
    assert list(generate_unordered_partition([1, 2, 3], [(1, 3)])) == [[[1], [2], [3]]]

    assert list(generate_unordered_partition([1, 2, 3, 4], [(4, 1)])) == [[[1, 2, 3, 4]]]
    assert list(generate_unordered_partition([1, 2, 3, 4], [(3, 1), (1, 1)])) == [
        [[1, 2, 3], [4]],
        [[1, 2, 4], [3]],
        [[1, 3, 4], [2]],
        [[2, 3, 4], [1]],
    ]
    assert list(generate_unordered_partition([1, 2, 3, 4], [(2, 2)])) == [
        [[1, 2], [3, 4]],
        [[1, 3], [2, 4]],
        [[1, 4], [2, 3]],
    ]
    assert list(generate_unordered_partition([1, 2, 3, 4], [(1, 4)])) == [[[1], [2], [3], [4]]]

    assert list(generate_unordered_partition([1], [(2, 1), (1, 2)])) == [
        [[1], [], []],
        [[], [1], []],
    ]
    assert list(generate_unordered_partition([1], [(1, 2), (2, 1)])) == [
        [[1], [], []],
        [[], [], [1]],
    ]

    assert list(generate_unordered_partition([1, 2], [(2, 1), (1, 2)])) == [
        [[1, 2], [], []],
        [[1], [2], []],
        [[2], [1], []],
        [[], [1], [2]],
    ]
    assert list(generate_unordered_partition([1, 2], [(1, 2), (2, 1)])) == [
        [[1], [2], []],
        [[1], [], [2]],
        [[2], [], [1]],
        [[], [], [1, 2]],
    ]

    assert list(generate_unordered_partition([1, 2, 3], [(2, 1), (1, 2)])) == [
        [[1, 2], [3], []],
        [[1, 3], [2], []],
        [[1], [2], [3]],
        [[2, 3], [1], []],
        [[2], [1], [3]],
        [[3], [1], [2]],
    ]
    assert list(generate_unordered_partition([1, 2, 3], [(1, 2), (2, 1)])) == [
        [[1], [2], [3]],
        [[1], [3], [2]],
        [[1], [], [2, 3]],
        [[2], [3], [1]],
        [[2], [], [1, 3]],
        [[3], [], [1, 2]],
    ]

    assert list(generate_unordered_partition([1, 2, 3, 4], [(2, 1), (1, 2)])) == [
        [[1, 2], [3], [4]],
        [[1, 3], [2], [4]],
        [[1, 4], [2], [3]],
        [[2, 3], [1], [4]],
        [[2, 4], [1], [3]],
        [[3, 4], [1], [2]],
    ]
    assert list(generate_unordered_partition([1, 2, 3, 4], [(1, 2), (2, 1)])) == [
        [[1], [2], [3, 4]],
        [[1], [3], [2, 4]],
        [[1], [4], [2, 3]],
        [[2], [3], [1, 4]],
        [[2], [4], [1, 3]],
        [[3], [4], [1, 2]],
    ]


def test_unorderd_partition_perfect_matching():
    """Test wfn.tools.graphs.generate_unordered_partition for perfect matching."""
    # 4 vertices
    occ_indices = [0, 1, 3, 4]
    answer = [[[0, 1], [3, 4]], [[0, 3], [1, 4]], [[0, 4], [1, 3]]]
    assert answer == list(generate_unordered_partition(occ_indices, [(2, 2)]))

    # 6 vertices
    occ_indices = [0, 1, 3, 4, 6, 7]
    answer = [
        [[0, 1], [3, 4], [6, 7]],
        [[0, 1], [3, 6], [4, 7]],
        [[0, 1], [3, 7], [4, 6]],
        [[0, 3], [1, 4], [6, 7]],
        [[0, 3], [1, 6], [4, 7]],
        [[0, 3], [1, 7], [4, 6]],
        [[0, 4], [1, 3], [6, 7]],
        [[0, 6], [1, 3], [4, 7]],
        [[0, 7], [1, 3], [4, 6]],
        [[0, 4], [1, 6], [3, 7]],
        [[0, 4], [1, 7], [3, 6]],
        [[0, 6], [1, 4], [3, 7]],
        [[0, 7], [1, 4], [3, 6]],
        [[0, 6], [1, 7], [3, 4]],
        [[0, 7], [1, 6], [3, 4]],
    ]

    assert answer == list(generate_unordered_partition(occ_indices, [(2, 3)]))


def test_int_partition_recursive():
    """Test wfn.tools.graphs.int_partition_recursive."""
    assert list(int_partition_recursive([1], 0, 1)) == []
    assert list(int_partition_recursive([1], -1, 1)) == []

    assert list(int_partition_recursive([1], 1, 0)) == [[]]
    assert list(int_partition_recursive([1], 0, 0)) == [[]]
    assert list(int_partition_recursive([1], -1, 0)) == [[]]

    assert list(int_partition_recursive([1], 1, 1)) == [[1]]
    assert list(int_partition_recursive([1], 1, 2)) == [[1, 1]]

    assert list(int_partition_recursive([1, 2], 2, 2)) == [[2], [1, 1]]
    assert list(int_partition_recursive([1, 2], 1, 2)) == [[1, 1]]
    assert list(int_partition_recursive([1, 2], 0, 2)) == []

    assert list(int_partition_recursive([1, 2, 3], 3, 3)) == [[3], [2, 1], [1, 1, 1]]
    assert list(int_partition_recursive([2, 3], 2, 3)) == [[3]]
    assert list(int_partition_recursive([1, 3], 2, 3)) == [[3], [1, 1, 1]]
    assert list(int_partition_recursive([1, 2], 2, 3)) == [[2, 1], [1, 1, 1]]

    assert list(int_partition_recursive([1, 2, 3, 4], 4, 4)) == [
        [4],
        [3, 1],
        [2, 2],
        [2, 1, 1],
        [1, 1, 1, 1],
    ]
    assert list(int_partition_recursive([1, 3, 4], 3, 4)) == [[4], [3, 1], [1, 1, 1, 1]]

    # messing with the order of the coins results in different ordering
    assert list(int_partition_recursive([3, 2, 1], 3, 3)) == [[1, 1, 1], [1, 2], [3]]
    assert list(int_partition_recursive([4, 1, 2, 3], 4, 4)) == [
        [3, 1],
        [2, 2],
        [2, 1, 1],
        [1, 1, 1, 1],
        [4],
    ]
    assert list(int_partition_recursive([4, 3, 2, 1], 4, 4)) == [
        [1, 1, 1, 1],
        [1, 1, 2],
        [1, 3],
        [2, 2],
        [4],
    ]


def test_generate_general_pmatch():
    """Test wfn.tools.graphs.generate_general_pmatch."""
    # bad input
    assert list(generate_general_pmatch([], np.ones((0, 0)))) == []
    assert list(generate_general_pmatch([1], np.ones((1, 1)))) == []
    assert list(generate_general_pmatch([1, 2, 3], np.ones((3, 3)))) == []

    # 4 vertices
    occ_indices = [0, 1, 3, 4]
    answer = [([(0, 1), (3, 4)], 1), ([(0, 3), (1, 4)], -1), ([(0, 4), (1, 3)], 1)]
    assert answer == list(generate_general_pmatch(occ_indices, np.ones((4, 4))))

    # 6 vertices
    occ_indices = [0, 1, 3, 4, 6, 7]
    answer = [
        (((0, 1), (3, 4), (6, 7)), 1),
        (((0, 1), (3, 6), (4, 7)), -1),
        (((0, 1), (3, 7), (4, 6)), 1),
        (((0, 3), (1, 4), (6, 7)), -1),
        (((0, 3), (1, 6), (4, 7)), 1),
        (((0, 3), (1, 7), (4, 6)), -1),
        (((0, 4), (1, 3), (6, 7)), 1),
        (((0, 4), (1, 6), (3, 7)), -1),
        (((0, 4), (1, 7), (3, 6)), 1),
        (((0, 6), (1, 7), (3, 4)), -1),
        (((0, 6), (1, 4), (3, 7)), 1),
        (((0, 6), (1, 3), (4, 7)), -1),
        (((0, 7), (1, 6), (3, 4)), 1),
        (((0, 7), (1, 4), (3, 6)), -1),
        (((0, 7), (1, 3), (4, 6)), 1),
    ]
    assert answer == [
        (tuple(sorted(i, key=lambda x: x[0])), sign)
        for i, sign in sorted(generate_complete_pmatch(occ_indices), key=lambda x: x[0])
    ]
