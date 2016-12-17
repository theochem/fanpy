import numpy as np
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
    """
    indices = tuple(indices)
    n = len(indices)
    if (n%2 == 1 or n < 2):
        raise ValueError, 'Given indices cannot produce a perfect matching (odd number of occupied indices)'
    if (n == 2):
        yield( ( (indices[0], indices[1]), ) )
    else:
        # smaller subset (all pairs without the last two indices)
        Sn_2 = generate_complete_pmatch(indices[:-2])
        for scheme in Sn_2:
            # add in the last two indices
            yield( scheme + (indices[-2:],) )
            # swap the last two indices wth an existing pair
            # remove the ith pair and shuffle the indices with the last pair
            for i in reversed(range(n//2 - 1)):
                # this part of the scheme is kept constant
                yield( scheme[:i] + ( (scheme[i][0], indices[-2]), (scheme[i][1], indices[-1]) ) + scheme[i+1:])
                yield( scheme[:i] + ( (scheme[i][0], indices[-1]), (scheme[i][1], indices[-2]) ) + scheme[i+1:])

def generate_biclique_pmatch(indices_one, indices_two):
    """ Generates all of the perfect matches of a complete bipartite (sub)graph

    Parameters
    ----------
    indices_one : list of int
         List of indices of the vertices used to create the first half of the complete bipartite graph
    indices_two : list of int
         List of indices of the vertices used to create the second half of the complete bipartite graph

    Yields
    ------
    pairing_scheme : tuple of tuple of 2 ints
        Contains the edges needed to make a perfect match
    """
    indices_one = tuple(sorted(indices_one))
    indices_two = tuple(sorted(indices_two))
    if len(indices_one) != len(indices_two):
        raise ValueError('Cannot make perfect matchings unless the number of vertices in each set is equal')
    if len(set(indices_one).symmetric_difference(set(indices_two))) < len(indices_one + indices_two):
        raise ValueError('A Bipartite graph cannot share vertices between the two sets')
    for new_indices in it.permutations(indices_two):
        yield tuple(zip(indices_one, new_indices))
