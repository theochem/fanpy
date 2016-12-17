""" Methods for constructing perfect matchings of a given graph

Functions
---------
generate_complete_pmatch(indices)
    Yields a perfect matching of a complete graph with given vertices, `indices`
generate_biclique_pmatch(indices_one, indices_two)
    Yields a perfect matching of a complete bipartite graph with given the partitions, `indices_one`
    and `indices_two`
"""
import itertools as it


def generate_complete_pmatch(indices):
    """ Generates all of the perfect matches of a complete (sub)graph

    Parameters
    ----------
    indices : list of int
         List of indices of the vertices used to create the complete graph

    Yields
    ------
    pairing_scheme : tuple of tuple of 2 ints
        Contains the edges needed to make a perfect match
    ValueError
        If the number of indices is less than two
        If the number of indices is odd

    Note
    ----
    The generator must be iterated to raise error
    """
    indices = tuple(indices)
    n = len(indices)
    if n % 2 == 1 or n < 2:
        raise ValueError('Given indices cannot produce a perfect matching (odd number of occupied'
                         ' indices)')
    if n == 2:
        yield ((indices[0], indices[1]), )
    else:
        # smaller subset (all pairs without the last two indices)
        Sn_2 = generate_complete_pmatch(indices[:-2])
        for scheme in Sn_2:
            # add in the last two indices
            yield scheme + (indices[-2:],)
            # swap the last two indices wth an existing pair
            # remove the ith pair and shuffle the indices with the last pair
            for i in reversed(range(n//2 - 1)):
                # this part of the scheme is kept constant
                yield (scheme[:i] +
                       ((scheme[i][0], indices[-2]), (scheme[i][1], indices[-1])) +
                       scheme[i+1:])
                yield (scheme[:i] +
                       ((scheme[i][0], indices[-1]), (scheme[i][1], indices[-2])) +
                       scheme[i+1:])


def generate_biclique_pmatch(indices_one, indices_two):
    """ Generates all of the perfect matches of a complete bipartite (sub)graph

    Parameters
    ----------
    indices_one : list of int
        List of indices of the vertices used to create the first half of the complete bipartite
        graph
    indices_two : list of int
        List of indices of the vertices used to create the second half of the complete bipartite
        graph

    Yields
    ------
    pairing_scheme : tuple of tuple of 2 ints
        Contains the edges needed to make a perfect match
    ValueError
        If the either one of the the two partitions have zero vertices
        If the number of vertices in the two partitions are not equal
        If the two partitions share vertices

    Note
    ----
    The generator must be iterated to raise error
    """
    indices_one = tuple(sorted(indices_one))
    indices_two = tuple(sorted(indices_two))
    if len(indices_one) == 0 or len(indices_one) == 0:
        raise ValueError('Cannot find the perfect matches of a disconnected bipartite graph')
    if len(indices_one) != len(indices_two):
        raise ValueError('Cannot make perfect matchings unless the number of vertices in each set'
                         ' is equal')
    if (len(set(indices_one).symmetric_difference(set(indices_two)))
            < len(indices_one + indices_two)):
        raise ValueError('A Bipartite graph cannot share vertices between the two sets')
    for new_indices in it.permutations(indices_two):
        yield tuple(zip(indices_one, new_indices))
