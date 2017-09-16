"""Collection of functions used to construct and manipulate Slater determinants.

Slater determinants are represented with a bitstring that describes their occupation. The :math:`0`
would correspond to an unoccupied orbital and the :math:`1` would correspond to the occupied
orbital. For example, `0b00110011` will have the occupied orbitals with indices :math:`0, 1, 4, 5`.

For most of the time, the orbitals are spin orbitals, and their spin is designated by splitting the
orbitals into two blocks. If there are :math:`K` spatial orbitals, then the first :math:`K` spin
orbitals are the alpha orbitals, and the second :math:`K` spin orbitals are the beta orbitals. The
spin orbitals can be equivalently described with alternating alpha and beta spin orbitals, but in
the current module, the Slater determinant will be assumed to be organized in the "block" format.

Though Python integers (in the binary format) can be used as a representation of the occupation
vector, the `gmpy2.mpz` object is used by default. The `gmpy2` is a module that efficiently handles
the bitwise operation of arbitrary length bitstrings. Note that all of these methods can work with
both integers and `gmpy2.mpz` objects. However, the two objects, e.g. `2` and `gmpy2.mpz(2)` are
different objects and may cause conflict when storing and finding them from a list/dictionary/set.

Functions
---------
is_internal_sd(sd) : bool
    Check if given Slater determinant is the same type as the one used internally in this module.
internal_sd(identifier) : gmpy2.mpz
    Create a Slater detrminant as a `gmpy2.mpz` object.
occ(sd, i) : bool
    Check if a given Slater determinant has orbital `i` occupied.
occ_indices(sd) : tuple of ints
    Return indices of the occupied orbitals.
vir_indices(sd, norbs) : tuple of ints
    Return the indices of all of the virtual orbitals.
total_occ(sd) : int
    Return the total number of occupied orbitals in a Slater determinant.
is_alpha(i, nspatial) : bool
    Check if index `i` is an alpha spin orbital.
spatial_index(i, nspatial) : int
    Return the spatial orbital index that corresponds to the spin orbital `i`.
annihilate(sd, *indices) : {gmpy2.mpz, None}
    Annihilate the occupied orbital `i` from a Slater determinant.
create(sd, *indices) : {gmpy2.mpz, None}
    Create electrons in the orbitals that correspond to the given indices.
excite(sd, *indices) : {gmpy2.mpz, None}
    Excite electrons from occupied orbitals to virtual orbitals.
ground(nocc, norbs) : {gmpy2.mpz}
    Create a ground state Slater determinant.
shared(sd1, sd2) : gmpy2.mpz
    Return similarity between two Slater determinants.
diff(sd1, sd2) : 2-tuple of tuple of ints
    Return the difference between two Slater determinants.
combine_spin(alpha_bits, beta_bits, nspatial) : gmpy2.mpz
    Construct a Slater determinant from the occupation of alpha and beta spin-orbitals.

"""
import gmpy2
import numpy as np
from wfns.wrapper.pydocstring import docstring


# FIXME: necessary?
@docstring(indent_level=1)
def is_internal_sd(sd):
    """Check if given Slater determinant is a `gmpy2.mpz` object.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Some representation of a Slater determinant.

    Returns
    -------
    True if it is the right type.
    False if it is not the right type.

    """
    return isinstance(sd, type(gmpy2.mpz()))


# FIXME: necessary?
@docstring(indent_level=1)
def internal_sd(identifier):
    """Create a Slater detrminant as a `gmpy2.mpz` object.

    Parameters
    ----------
    identifier : {int, gmpy2.mpz}
        Occupation vector of a Slater determinant.
        Binary form of the integer describes the occupation of each orbital.

    Returns
    -------
    sd : gmpy2.mpz
        Representation of Slater determinant within this module.

    Raises
    ------
    TypeError
        If `identifier` not an integer or `gmpy2.mpz` object.

    """
    if isinstance(identifier, int):
        return gmpy2.mpz(identifier)
    elif is_internal_sd(identifier):
        return identifier
    else:
        raise TypeError('Unsupported Slater determinant form, {0}'.format(type(identifier)))


@docstring(indent_level=1)
def occ(sd, i):
    """Check if a given Slater determinant has orbital `i` occupied.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
    i : int
        Index of an orbital.

    Returns
    -------
    is_occ : bool
        True if orbital is occupied.
        False if orbital is not occupied.

    """
    return sd is not None and gmpy2.bit_test(sd, i)


@docstring(indent_level=1)
def occ_indices(sd):
    """Return indices of the occupied orbitals.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Returns
    -------
    occ_indices : tuple of int
        Tuple of occupied orbitals indices.

    """
    if sd is None:
        return ()
    output = [gmpy2.bit_scan1(sd, 0)]
    while output[-1] is not None:
        output.append(gmpy2.bit_scan1(sd, output[-1] + 1))
    return tuple(output[:-1])


@docstring(indent_level=1)
def vir_indices(sd, norbs):
    """Return the indices of all of the virtual orbitals.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
    norbs : int
        Total number of orbitals.

    Returns
    -------
    occ_indices : tuple of int
        Tuple of virtual orbital indices.

    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if sd is None or norbs <= 0:
        return ()
    output = [gmpy2.bit_scan0(sd, 0)]
    while output[-1] < norbs:
        output.append(gmpy2.bit_scan0(sd, output[-1] + 1))
    return tuple(output[:-1])


@docstring(indent_level=1)
def total_occ(sd):
    """Return the total number of occupied orbitals in a Slater determinant.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Returns
    -------
    total _occ : int
        Number of occupied orbitals.

    """
    if sd is None:
        return 0
    else:
        return gmpy2.popcount(sd)


@docstring(indent_level=1)
def is_alpha(i, nspatial):
    """Check if index `i` is an alpha spin orbital.

    Parameters
    ----------
    i : int
        Index of the spin orbital in the Slater determinant.
    nspatial : int
        Number of spatial orbitals.

    Returns
    -------
    is_alpha : bool
        True if alpha orbital.
        False if beta orbital.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.
        If `i < 0`.
        If `i > 2*nspatial`.

    """
    if nspatial <= 0 or i < 0 or i > 2*nspatial:
        raise ValueError('If `nspatial <= 0` or `i < 0` or `i > 2*nspatial`.')
    return i < nspatial


@docstring(indent_level=1)
def spatial_index(i, nspatial):
    """Return the spatial orbital index that corresponds to the spin orbital `i`.

    Parameters
    ----------
    i : int
        Spin orbital index in the Slater determinant.
    nspatial : int
        Number of spatial orbitals.

    Returns
    -------
    ind_spatial : int
        Spatial orbital index that corresponds to the spin orbital `i`.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.
        If `i < 0`.
        If `i > 2*nspatial`.

    """
    if nspatial <= 0 or i < 0 or i > 2*nspatial:
        raise ValueError('If `nspatial <= 0` or `i < 0` or `i > 2*nspatial`.')

    if is_alpha(i, nspatial):
        return i
    else:
        return i - nspatial


@docstring(indent_level=1)
def annihilate(sd, *indices):
    """Annihilate the occupied orbital `i` from a Slater determinant.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
    indices : int
        The indices of the orbitals to annihilate.

    Returns
    -------
    sd : {gmpy2.mpz, None}
        Integer that describes the occupation of a Slater determinant as a bitstring.
        If an electron is annihilated in a virtual orbital, `None` is returned.

    """
    for i in indices:
        # if orbital is not occupied
        if not occ(sd, i):
            return None
        # if orbital is not occupied
        sd = gmpy2.bit_flip(sd, i)
    return sd


@docstring(indent_level=1)
def create(sd, *indices):
    """Create electrons in the orbitals that correspond to the given indices.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
    indices : int
        The indices of the orbitals to create.

    Returns
    -------
    sd : {gmpy2.mpz, None}
        Integer that describes the occupation of a Slater determinant as a bitstring.
        If an electron is created in an occupied orbital, `None` is returned.

    """
    for i in indices:
        if not occ(sd, i) and sd is not None:
            sd = gmpy2.bit_flip(sd, i)
        else:
            return None
    return sd


@docstring(indent_level=1)
def excite(sd, *indices):
    """Excite electrons from occupied orbitals to virtual orbitals.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
    indices : int
        The indices of the orbitals to annihilate and create.
        The first half contain indices of orbitals to annihilate.
        The second half contain indices of orbitals to create.

    Returns
    -------
    excited_sd : {gmpy2.mpz, None}
        Integer that describes the occupation of a Slater determinant as a bitstring.
        If an occupied orbital is unoccupied or a virtual orbital is occupied, `None` is returned.

    Raises
    ------
    ValueError
        If the length of indices is not even (cannot evenly split up the indices in two).

    """
    if (len(indices) % 2) != 0:
        raise ValueError("Unqual number of creators and annihilators")
    sd = annihilate(sd, *indices[:len(indices) // 2])
    sd = create(sd, *indices[len(indices) // 2:])
    return sd


@docstring(indent_level=1)
def ground(nocc, norbs):
    """Create a ground state Slater determinant.

    If the number of electrons is odd, then the last electron is put into an alpha orbital.

    Parameters
    ----------
    nocc : int
        Number of occupied spin-orbitals.
    norbs : int
        Total number of spin-orbitals.

    Returns
    -------
    ground_sd : gmpy2.mpz
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Raises
    ------
    ValueError
        If `nocc > norbs`.
        If `norbs` is odd.

    Notes
    -----
    Assumes that the spin-orbitals are ordered by energy from lowest to greatest. The occupation is
    assumed to have the alpha block frist, then the beta block.

    """
    if nocc > norbs:
        raise ValueError('Number of occupied spin-orbitals must be less than the total number of'
                         ' spin-orbitals')
    if norbs % 2 != 0:
        raise ValueError('Total number of spin-orbitals must be even')
    alpha_bits = gmpy2.bit_mask(nocc // 2 + nocc % 2)
    beta_bits = gmpy2.bit_mask(nocc // 2) << (norbs // 2)
    return alpha_bits | beta_bits


# FIXME: API for shared and diff are too different.
@docstring(indent_level=1)
def shared(sd1, sd2):
    """Return similarity between two Slater determinants.

    Parameters
    ----------
    sd1 : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
    sd2 : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Returns
    -------
    shared : gmpy2.mpz
        Integer that describes the occupied orbitals that are shared between two Slater
        determinants.

    """
    return sd1 & sd2


@docstring(indent_level=1)
def diff(sd1, sd2):
    """Return the difference between two Slater determinants.

    Parameters
    ----------
    sd1 : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
    sd2 : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Returns
    -------
    diff : 2-tuple of tuple of ints
        First tuple are the occupied orbital indices of `sd1` that are not occupied in `sd2`.
        Second tuple are the occupied orbital indices of `sd2` that are not occupied in `sd1`.

    """
    sd_diff = sd1 ^ sd2
    sd1_diff = sd_diff & sd1
    sd2_diff = sd_diff & sd2
    return (occ_indices(sd1_diff), occ_indices(sd2_diff))


@docstring(indent_level=1)
def combine_spin(alpha_bits, beta_bits, nspatial):
    """Construct a Slater determinant from the occupation of alpha and beta spin-orbitals.

    Parameters
    ----------
    alpha_bits : {int, gmpy2.mpz}
        Integer that describes the occupation of alpha spin orbitals as a bitstring.
    beta_bits : {int, gmpy2.mpz}
        Integer that describes the occupation of beta spin orbitals as a bitstring.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    block_sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Indices less than `nspatial` correspond to the alpha spin orbitals.
        Indices greater than or equal to `nspatial` correspond to the beta spin orbitals.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.

    Notes
    -----
    Erratic behaviour if the total number of spatial orbitals is less than the the actual number
    (i.e. if there are any occupied orbitals with indices greater than nspatial).

    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if nspatial <= 0:
        raise ValueError('Number of spatial orbitals must be greater than 0.')
    return alpha_bits | (beta_bits << nspatial)


@docstring(indent_level=1)
def split_spin(block_sd, nspatial):
    """Split a Slater determinant into the alpha and beta parts.

    Parameters
    ----------
    block_sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Indices less than `nspatial` correspond to the alpha spin orbitals.
        Indices greater than or equal to `nspatial` correspond to the beta spin orbitals.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    alpha_bits : int
        Integer that describes the occupation of alpha spin-orbitals as a bitstring.
    beta_bits : int
        Integer that describes the occupation of beta spin-orbitals as a bitstring.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.

    Notes
    -----
    Erratic behaviour if the total number of spatial orbitals is less than the the actual number
    (i.e. if there are any occupied orbitals with indices greater than nspatial).

    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if nspatial <= 0:
        raise ValueError('Number of spatial orbitals must be greater than 0')
    alpha_bits = gmpy2.t_mod_2exp(block_sd, nspatial)
    beta_bits = block_sd >> nspatial
    return (alpha_bits, beta_bits)


@docstring(indent_level=1)
def interleave_index(i, nspatial):
    """Convert orbital index in block-sd notation to that of interleaved-sd notation.

    Parameters
    ----------
    i : int
        Index of orbital in block-sd notation.
    nspatial : int
        Number of spatial orbitals.

    Returns
    -------
    interleave_index : int
        Index of the same orbital in interleaved notation.

    Raises
    ------
    ValueError
        If the index is less than zero.
        If the index is greater than or equal to the number of spin orbitals.

    """
    if i < 0:
        raise ValueError('Index must be greater than or equal to zero.')
    elif i >= 2*nspatial:
        raise ValueError('Index must be less than the number of spin orbitals.')
    if i < nspatial:
        return 2*i
    else:
        return 2*(i - nspatial) + 1


@docstring(indent_level=1)
def deinterleave_index(i, nspatial):
    """Convert an orbital index in interleaved-sd notation to that of block-sd notation.

    Parameters
    ----------
    i : int
        Index of orbital in interleaved-sd notation.
    nspatial : int
        Number of spatial orbitals.

    Returns
    -------
    block_index : int
        Index of the same orbital in block notation.

    Raises
    ------
    ValueError
        If the index is less than zero.
        If the index is greater than or equal to the number of spin orbitals.

    """
    if i < 0:
        raise ValueError('Index must be greater than or equal to zero.')
    elif i >= 2*nspatial:
        raise ValueError('Index must be less than the number of spin orbitals.')
    if i % 2 == 0:
        return i//2
    else:
        return i//2 + nspatial


@docstring(indent_level=1)
def interleave(block_sd, nspatial):
    """Convert block-sd to the interleaved-sd form.

    Block form: alpha1, alpha2, ..., beta1, beta2, ...

    Shuffled form: alpha1, beta1, alpha2, beta2, ...

    Parameters
    ----------
    block_sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Indices less than nspatial correspond to the alpha spin orbitals.
        Indices greater than or equal to nspatial correspond to the beta spin orbitals.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    shuffled_sd : gmpy2.mpz
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Odd indices correspond to the alpha spin orbitals.
        Even indices correspond to the beta spin orbitals.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.

    Notes
    -----
    Erratic behaviour if the total number of spatial orbitals is less than the the actual number
    (i.e. if there are any occupied orbitals with indices greater than nspatial).

    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if nspatial <= 0:
        raise ValueError('Number of spatial orbitals must be greater than 0')

    # OPTION 1
    # shuffled_sd = gmpy2.mpz(0)
    # for i in range(nspatial):
    #     if gmpy2.bit_test(block_sd, i):
    #         shuffled_sd |= 1 << 2 * i
    #     if gmpy2.bit_test(block_sd, i + nspatial):
    #         shuffled_sd |= 1 << 2 * i + 1

    # OPTION 2
    sd_bit = bin(block_sd)[2:]
    sd_bit = '0'*(nspatial*2-len(sd_bit)) + sd_bit
    alpha_bit, beta_bit = sd_bit[nspatial:], sd_bit[:nspatial]
    shuffled_bit = '0b'+''.join(''.join(i) for i in zip(beta_bit, alpha_bit))
    shuffled_sd = gmpy2.mpz(shuffled_bit)
    return shuffled_sd

    # OPTION 3
    # shuffled_sd = gmpy2.mpz(0)
    # i = gmpy2.bit_scan1(block_sd, 0)
    # while i is not None:
    #     shuffled_sd = create(shuffled_sd, interleave_index(i, nspatial))
    # return shuffled_sd


@docstring(indent_level=1)
def deinterleave(shuffled_sd, nspatial):
    """Turn sd from shuffled form to the block form

    Block form: alpha1, alpha2, ..., beta1, beta2, ...

    Shuffled form: alpha1, beta1, alpha2, beta2, ...

    Parameters
    ----------
    shuffled_sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Odd indices correspond to the alpha spin orbitals.
        Even indices correspond to the beta spin orbitals.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    block_sd : gmpy2.mpz
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Indices less than nspatial correspond to the alpha spin orbitals.
        Indices greater than or equal to nspatial correspond to the beta spin orbitals.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.

    Notes
    -----
    Erratic behaviour if the total number of spatial orbitals is less than the the actual number
    (i.e. if there are any occupied orbitals with indices greater than nspatial).

    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if nspatial <= 0:
        raise ValueError('Number of spatial orbitals must be greater than 0')

    # OPTION 1
    # block_sd = gmpy2.mpz(0)
    # for i in range(nspatial):
    #     if gmpy2.bit_test(shuffled_sd, 2 * i):
    #         block_sd |= 1 << i
    #     if gmpy2.bit_test(shuffled_sd, 2 * i + 1):
    #         block_sd |= 1 << i + nspatial

    # OPTION 2
    sd_bit = bin(shuffled_sd)[2:]
    alpha_bit, beta_bit = sd_bit[-1::-2][::-1], sd_bit[-2::-2][::-1]
    alpha_bit = '0'*(nspatial - len(alpha_bit)) + alpha_bit
    beta_bit = '0'*(nspatial - len(beta_bit)) + beta_bit
    block_bit = '0b' + beta_bit + alpha_bit
    block_sd = gmpy2.mpz(block_bit)
    return block_sd

    # OPTION 3
    # shuffled_sd = gmpy2.mpz(0)
    # i = gmpy2.bit_scan1(block_sd, 0)
    # while i is not None:
    #     shuffled_sd = create(shuffled_sd, deinterleave_index(i, nspatial))
    # return shuffled_sd


@docstring(indent_level=1)
def get_spin(sd, nspatial):
    """Return the spin of the given Slater determinant.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    spin : float
        Spin of the given Slater determinant.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.

    """
    alpha_bits, beta_bits = split_spin(sd, nspatial)
    return (0.5)*(total_occ(alpha_bits) - total_occ(beta_bits))


@docstring(indent_level=1)
def get_seniority(sd, nspatial):
    """Return the seniority of the given Slater determinant.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    seniority : int
        Seniority of the given Slater determinant.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.

    """
    alpha_bits, beta_bits = split_spin(sd, nspatial)
    return total_occ(alpha_bits ^ beta_bits)


# FIXME: bad location
@docstring(indent_level=1)
def find_num_trans(jumbled_set, ordered_set=None, is_decreasing=False):
    """Find the number of transpositions needed to order a set of annihilators.

    Basically, we count the number of times each annihilator needs to hop over another to sort it.
    If the indices of the annihilators to its left are greater than its index, then it hops over
    those annihilators. Once the annihilator with the lowest index hops over all annihilators with a
    lower index to its left, the annihilator with the second lowest index hops, and so on, until the
    annihilators are ordered from smallest to largest indices.

    Parameters
    ----------
    jumbled_set : {tuple, list}
        Set of indices of the annihilators.
    ordered_set : {tuple, list}
        Set of indices ordered in increasing order.
        If not provided, then the given indices are ordered.
    is_decreasing : bool
        If True, then the number of transpositions required to get strictly decreasing list is
        returned. Note that the `ordered_set` must still be given in an increasing order.
        Default is False.

    Returns
    -------
    num_trans : int
        Number of adjacent transpositions needed to sort the `jumbled_set`.

    Raises
    ------
    ValueError
        If `ordered_set` is not strictly increasing.

    See Also
    --------
    * `wfns.slater.find_num_trans_dumb`

    Notes
    -----
    Though only adjacent elements are swapped, the order of the permutation that orders the given
    set should be the same.

    """
    jumbled_set = np.array(jumbled_set)
    if is_decreasing:
        jumbled_set = jumbled_set[::-1]

    # Given that the first index corresponds to the smallest number, and the
    # last index corresponds to the largest number,
    if ordered_set is None:
        # location of each number in the jumbled set
        loc_jumbled_num = np.argsort(jumbled_set)
        # ordered set
        ordered_set = jumbled_set[loc_jumbled_num]
    elif not all(i < j for i, j in zip(ordered_set, ordered_set[1:])):
        raise ValueError('ordered_set must be strictly increasing order.')
    else:
        # FIXME: ordered_set not checked
        ordered_set = np.array(ordered_set)
        # location of each number in the jumbled set
        loc_jumbled_num = np.where(jumbled_set == ordered_set[:, np.newaxis])[1]

    # Find all the numbers that are greater than itself
    truths_nums_greater = jumbled_set > ordered_set[:, np.newaxis]
    # Find all the numbers to the right of itself
    truths_nums_right = np.arange(jumbled_set.size) > loc_jumbled_num[:, np.newaxis]
    # Find all the numbers that are greater than itself on the left
    truths_nums_greater_left = truths_nums_greater.astype(int) - truths_nums_right.astype(int) > 0
    return np.sum(truths_nums_greater_left)


# FIXME: bad location
# FIXME: remove?
@docstring(indent_level=1)
def find_num_trans_dumb(jumbled_set, ordered_set=None, is_decreasing=True):
    """Find the number of transpositions needed to order a set of annihilators in increasing order.

    This method has the same functionality as `find_num_trans`, execept the number of swaps is
    counted explicitly for each index. This was used for debugging purposes.

    Parameters
    ----------
    jumbled_set : {tuple, list}
        Set of indices of the annihilators.
    ordered_set : {tuple, list}
        Set of indices ordered in increasing order.
        If not provided, then the given indices are ordered.
    is_decreasing : bool
        If True, then the number of transpositions required to get strictly decreasing list is
        returned. Note that the `ordered_set` must still be given in an increasing order.
        Default is False.

    Returns
    -------
    num_trans : int
        Number of hops needed to sort the `jumbled_set`.

    Raises
    ------
    ValueError
        If `ordered_set` is not strictly increasing.

    """
    if is_decreasing:
        jumbled_set = jumbled_set[::-1]

    # get ordered set
    if ordered_set is None:
        ordered_set = sorted(jumbled_set)
    elif not all(i < j for i, j in zip(ordered_set, ordered_set[1:])):
        raise ValueError('ordered_set must be strictly increasing order.')

    num_trans = 0
    # for each ordered number
    for i in ordered_set:
        for j in jumbled_set:
            # count the number of numbers that are greater than it
            if j > i:
                num_trans += 1
            # skip over numbers to the right
            elif j == i:
                break
    return num_trans


# FIXME: location? and pretty similar to occ_indices
# FIXME: API different from other `find_num_trans`
@docstring(indent_level=1)
def find_num_trans_swap(sd, pos_current, pos_future):
    """Find the number of swaps needed to move an orbital from one position to another.

    Parameters
    ----------
    sd : {int, gmpy2.mpz}
        Integer that describes the occupation of a Slater determinant as a bitstring.
    pos_current : int
        Position of the orbital that needs to be moved.
    pos_future : int
        Position to which the orbital is moved.

    Returns
    -------
    num_trans : int
        Number of hops needed to move the orbital.

    Raises
    ------
    ValueError
        If Slater determinant is None.
        If position is not a positive integer.
        If current orbital position is not occupied.
        If future orbital position is occupied.

    """
    if sd is None:
        raise ValueError('Bad Slater determinant is given.')
    if not (isinstance(pos_current, int) and 0 <= pos_current):
        raise ValueError('The current orbital position must be a positive integer.')
    if not (isinstance(pos_future, int) and 0 <= pos_future):
        raise ValueError('The future orbital position must be a positive integer.')
    if not occ(sd, pos_current):
        raise ValueError('Given orbital is not occupied in the given Slater determinant.')
    if occ(sd, pos_future):
        raise ValueError('Given future orbital is occupied in the given Slater determinant.')

    if pos_current > pos_future:
        pos_current, pos_future = pos_future, pos_current

    output = 0
    pos_last = pos_current
    while True:
        pos_last = gmpy2.bit_scan1(sd, pos_last + 1)
        if pos_last is None or pos_last >= pos_future:
            break
        else:
            output += 1
    return output
