""" Tests wfn.graphs
"""
from nose.tools import assert_raises
from wfns.graphs import generate_complete_pmatch, generate_biclique_pmatch

def test_generate_complete_pmatch():
    """ Tests the generator for the perfect matchings of a complete graph
    """
    # check for assert statements
    assert_raises(ValueError, lambda: list(generate_complete_pmatch([])))
    assert_raises(ValueError, lambda: list(generate_complete_pmatch([1])))
    assert_raises(ValueError, lambda: list(generate_complete_pmatch([1, 2, 3])))

    # 4 vertices
    occ_indices = [0, 1, 3, 4]
    answer = [((0, 1), (3, 4)),
              ((0, 3), (1, 4)),
              ((0, 4), (1, 3))]
    assert answer == list(generate_complete_pmatch(occ_indices))

    # 8 vertices
    occ_indices = [0, 1, 3, 4, 6, 7]
    answer = [((0, 1), (3, 4), (6, 7)),
              ((0, 1), (3, 6), (4, 7)),
              ((0, 1), (3, 7), (4, 6)),
              ((0, 3), (1, 4), (6, 7)),
              ((0, 3), (1, 6), (4, 7)),
              ((0, 3), (1, 7), (4, 6)),
              ((0, 4), (1, 3), (6, 7)),
              ((0, 4), (1, 6), (3, 7)),
              ((0, 4), (1, 7), (3, 6)),
              ((0, 6), (1, 7), (3, 4)),
              ((0, 6), (1, 4), (3, 7)),
              ((0, 6), (1, 3), (4, 7)),
              ((0, 7), (1, 6), (3, 4)),
              ((0, 7), (1, 4), (3, 6)),
              ((0, 7), (1, 3), (4, 6)),]
    assert answer == [tuple(sorted(i, key=lambda x: x[0])) for i in
                      sorted(generate_complete_pmatch(occ_indices), key=lambda x: x[0])]


def test_generate_biclique_pmatch():
    """ Tests the generator for the perfect matchings of a complete bipartite graph
    """
    # check for assert statements
    assert_raises(ValueError, lambda: list(generate_biclique_pmatch([], [])))
    assert_raises(ValueError, lambda: list(generate_biclique_pmatch([], [1, 2])))
    assert_raises(ValueError, lambda: list(generate_biclique_pmatch([1], [2, 3])))
    assert_raises(ValueError, lambda: list(generate_biclique_pmatch([1, 2], [2, 3])))

    # 4 vertices
    indices_one = [0, 1]
    indices_two = [3, 4]
    answer = [((0, 3), (1, 4)),
              ((0, 4), (1, 3))]
    assert answer == list(generate_biclique_pmatch(indices_one, indices_two))

    # 6 vertices
    indices_one = [0, 1, 3]
    indices_two = [4, 6, 7]
    answer = [((0, 4), (1, 6), (3, 7)),
              ((0, 4), (1, 7), (3, 6)),
              ((0, 6), (1, 4), (3, 7)),
              ((0, 6), (1, 7), (3, 4)),
              ((0, 7), (1, 4), (3, 6)),
              ((0, 7), (1, 6), (3, 4)),]
    assert answer == list(generate_biclique_pmatch(indices_one, indices_two))

    # 8 vertices
    indices_one = [0, 1, 3, 5]
    indices_two = [4, 6, 7, 8]
    answer = [((0, 4), (1, 6), (3, 7), (5, 8)),
              ((0, 4), (1, 6), (3, 8), (5, 7)),
              ((0, 4), (1, 7), (3, 6), (5, 8)),
              ((0, 4), (1, 7), (3, 8), (5, 6)),
              ((0, 4), (1, 8), (3, 6), (5, 7)),
              ((0, 4), (1, 8), (3, 7), (5, 6)),
              ((0, 6), (1, 4), (3, 7), (5, 8)),
              ((0, 6), (1, 4), (3, 8), (5, 7)),
              ((0, 6), (1, 7), (3, 4), (5, 8)),
              ((0, 6), (1, 7), (3, 8), (5, 4)),
              ((0, 6), (1, 8), (3, 4), (5, 7)),
              ((0, 6), (1, 8), (3, 7), (5, 4)),
              ((0, 7), (1, 4), (3, 6), (5, 8)),
              ((0, 7), (1, 4), (3, 8), (5, 6)),
              ((0, 7), (1, 6), (3, 4), (5, 8)),
              ((0, 7), (1, 6), (3, 8), (5, 4)),
              ((0, 7), (1, 8), (3, 4), (5, 6)),
              ((0, 7), (1, 8), (3, 6), (5, 4)),
              ((0, 8), (1, 4), (3, 6), (5, 7)),
              ((0, 8), (1, 4), (3, 7), (5, 6)),
              ((0, 8), (1, 6), (3, 4), (5, 7)),
              ((0, 8), (1, 6), (3, 7), (5, 4)),
              ((0, 8), (1, 7), (3, 4), (5, 6)),
              ((0, 8), (1, 7), (3, 6), (5, 4)),]
    assert answer == list(generate_biclique_pmatch(indices_one, indices_two))
