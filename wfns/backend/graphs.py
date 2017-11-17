r"""Functions for constructing perfect matchings of a given graph.

Functions
---------
generate_complete_pmatch(indices)
    Yields a perfect matching of a complete graph with given vertices, `indices`.
generate_biclique_pmatch(indices_one, indices_two)
    Yields a perfect matching of a complete bipartite graph with given the partitions, `indices_one`
    and `indices_two`.

"""
from wfns.backend.slater import sign_perm


def generate_complete_pmatch(indices, sign=1):
    r"""Generate all of the perfect matches of a complete (sub)graph.

    Generated perfect matches correspond to a pairing scheme in geminal wavefunctions and the
    signature is needed to find the sign of the Slater determinant after "unpacking" the pairing
    scheme.

    Parameters
    ----------
    indices : list of int
        List of indices of the vertices used to create the complete graph.

    Yields
    ------
    pairing_scheme : tuple of tuple of 2 ints
        Contains the edges needed to make a perfect match.
    sign : {1, -1}
        Signature of the transpositions required to shuffle the `pairing_scheme` back into the
        original order in `indices`.

    Notes
    -----
    The `sign` gives the signature with respect to the original `indices`. If the `indices` are not
    ordered, then signatures won't really mean anything.

    """
    indices = tuple(indices)
    n = len(indices)
    if n % 2 == 1 or n < 2:
        yield tuple(), sign
    elif n == 2:
        yield ((indices[0], indices[1]), ), sign
    else:
        # smaller subset (all pairs without the last two indices)
        Sn_2 = generate_complete_pmatch(indices[:-2], sign=sign)
        for scheme, inner_sign in Sn_2:
            # add in the last two indices
            yield scheme + (indices[-2:],), inner_sign
            # starting from the last
            for i in reversed(range(n//2 - 1)):
                # replace ith pair in the scheme with last pair
                yield (scheme[:i] +
                       ((scheme[i][0], indices[-2]), (scheme[i][1], indices[-1])) +
                       scheme[i+1:]), -inner_sign
                yield (scheme[:i] +
                       ((scheme[i][0], indices[-1]), (scheme[i][1], indices[-2])) +
                       scheme[i+1:]), inner_sign


def generate_biclique_pmatch(indices_one, indices_two, ordered_set=None, is_decreasing=False):
    r"""Generate all of the perfect matches of a complete bipartite (sub)graph.

    Parameters
    ----------
    indices_one : list of int
        List of indices of the vertices used to create the first half of the complete bipartite
        graph.
    indices_two : list of int
        List of indices of the vertices used to create the second half of the complete bipartite
        graph.
    ordered_set : {tuple, list}
        Set of indices ordered in strictly increasing order.
        If not provided, then the given indices are ordered.
    is_decreasing : bool
        If True, indices are ordered so that they are decreasing. (Sometimes creators are ordered
        from greatest to smallest).
        Default is False.

    Yields
    ------
    pairing_scheme : tuple of tuple of 2 ints
        Contains the edges needed to make a perfect match.
    sign : {1, -1}
        Signature of the transpositions required to shuffle the `pairing_scheme` back into the
        original order in `indices`.

    Notes
    -----
    The generator must be iterated to raise error.

    """
    # assume indices_one and indices_two are sorted
    sign = 1
    orig_sign = sign_perm([i for pair in zip(indices_one, indices_two) for i in pair],
                          ordered_set=ordered_set, is_decreasing=is_decreasing)

    if len(indices_one) == 0 or len(indices_two) == 0:
        yield tuple(), sign
    elif len(indices_one) != len(indices_two):
        yield tuple(), sign
    elif (len(set(indices_one).symmetric_difference(set(indices_two)))
          < len(indices_one + indices_two)):
        yield tuple(), sign
    else:
        # for new_indices in it.permutations(indices_two):
        #     yield tuple(zip(indices_one, new_indices))
        # NOTE: This is the code when life was simpler. Now, permutations code need to be
        #       implemented to account for the signature changes
        # Following code was adapted from the example permutations code in itertools.permutations
        # documentations
        pool = tuple(indices_two)
        # pool is the set of object from which you will be selecting
        n = len(pool)
        indices = list(range(n))
        # indices select the specific ordering
        cycles = list(reversed(range(1, n+1)))
        # cycles keeps track of the number of swaps and the positions of elements that are swapped
        yield tuple(zip(indices_one, (pool[i] for i in indices))), sign * orig_sign
        # NOTE: to obtain the signature, the jumbld pair structure must be unzipped, then sorted
        #       from largest to smallest. orig_sign accounts for this transposition/permutation
        while n:
            for i in reversed(range(n)):
                cycles[i] -= 1
                if cycles[i] == 0:
                    # move ith index to the end
                    indices[i:] = indices[i+1:] + indices[i:i+1]
                    # in order to move ith element to the end, it must jump over n-i-1 elements
                    # (because i starts from 0)
                    sign *= (-1)**(n - i - 1)
                    # reset cycles (back to its original number)
                    cycles[i] = n - i
                else:
                    j = cycles[i]
                    # swap
                    indices[i], indices[-j] = indices[-j], indices[i]
                    # change sign because swapping any two elements with x elements in between will
                    # require x+(x+1)=2x+1 swaps
                    sign *= -1
                    yield tuple(zip(indices_one, (pool[i] for i in indices))), sign * orig_sign
                    break
            else:
                return
