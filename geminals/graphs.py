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
            for i in range(n//2 - 1):
                # this part of the scheme is kept constant
                base_scheme = scheme[:i] + scheme[i+1:]
                yield( base_scheme + ( (scheme[i][0], indices[-2]), (scheme[i][1], indices[-1]) ) )
                yield( base_scheme + ( (scheme[i][0], indices[-1]), (scheme[i][1], indices[-2]) ) )
