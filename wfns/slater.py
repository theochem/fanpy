""" Collection of functions used to construct and manipulate Slater determinants

Functions
---------
occ(sd, i)
    Check if the orbital, `i`, is occupied in Slater determinant, `sd`
is_alpha(i, nspatial)
    Checks if spin orbital index `i` belongs to an alpha spin orbital
spatial_index(i, nspatial)
    Converts the spin orbital index, `i`, to the spatial orbital index
total_occ(sd)
    Returns the total number of occupied orbitals in a Slater determinant
annihilate(sd, *indices)
    Annihilates occupied orbitals of a Slater determinant
create(sd, *indices)
    Creates orbitals in a Slater determinant
excite(sd, *indices)
    Excites electrons from one set of orbitals to another in a Slater determinant
ground(nocc, norbs)
    Creates the ground state Slater determinant with `nocc` occupied orbitals and `norbs` spin
    orbitals
is_internal_sd(sd)
    Checks if given Slater determinant is consistent with the internal Slater determinant
    representation
internal_sd(identifier)
    Creates a Slater determinant that is consistent with the inner workings of this module
occ_indices(sd)
    Returns indices of all of the occupied orbitals
vir_indices(sd, norbs)
    Returns indices of all of the virtual orbitals
shared(sd1, sd2)
    Returns indices of all orbitals shared between two Slater determinants, `sd1` and `sd2`
diff(sd1, sd2)
    Returns the difference between two Slater determinants, `sd1` and `sd2`
combine_spin(alpha_bits, beta_bits, nspatial)
    Constructs a Slater determinant in block form from alpha and beta occupations
split_spin(block_sd, nspatial)
    Splits a Slater determinant in block form to the alpha and beta parts
interleave_index(i, nspatial)
    Converts orbital index, `i`, of Slater determinant in block form to shuffled form
deinterleave_index(i, nspatial)
    Converts orbital index, `i`, of Slater determinant in shuffled form to block form
interleave(block_sd, nspatial)
    Converts Slater determinants from block form to shuffled form
deinterleave(shuffled_sd, nspatial)
    Converts Slater determinants from shuffled form to block form
get_spin
    Returns the spin of the given slater determinant
find_num_trans(jumbled_set, ordered_set=None, is_creator=True)
    Returns the number of adjacent swaps necessary to convert a set of indices into increasing order
find_num_trans_dumb(jumbled_set, ordered_set=None, is_creator=True)
    Returns the number of transpostions necessary to convert a set of indices into increasing order
    using brute force
"""
from itertools import tee
import gmpy2
import numpy as np


def occ(sd, i):
    """
    Checks if a given Slater determinant has orbital `i` occupied.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    i : int
        Index for an occupied orbital

    Returns
    -------
    bool
        True if occupied
        False if not occupied
    """
    if sd is None:
        return False
    else:
        return gmpy2.bit_test(sd, i)

def is_alpha(i, nspatial):
    """ Checks if index `i` belongs to an alpha spin orbital

    Parameter
    ---------
    i : int
        Index of the spin orbital in the Slater determinant
    nspatial : int
        Number of spatial orbitals

    Returns
    -------
    True if alpha orbital
    False if beta orbital

    Note
    ----
    Erratic behaviour of nspatial <= 0
    Erratic behaviour of i < 0 or i > 2*nspatial
    """
    return i < nspatial


def spatial_index(i, nspatial):
    """ Returns the index of the spatial orbital that corresponds to the
    spin orbital `i`

    Parameter
    ---------
    i : int
        Index of the spin orbital in the Slater determinant
    nspatial : int
        Number of spatial orbitals

    Returns
    -------
    ind_spatial : int
        Index of the spatial orbital that corresponds to the spin orbital `i`

    Note
    ----
    Erratic behaviour of nspatial <= 0
    Erratic behaviour of i < 0 or i > 2*nspatial
    """
    if is_alpha(i, nspatial):
        return i
    else:
        return i - nspatial


def total_occ(sd):
    """
    Returns the total number of occupied orbitals in a Slater determinant

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring

    Returns
    -------
    int
        Number of occupied orbitals in Slater determinant
    """
    if sd is None:
        return 0
    else:
        return gmpy2.popcount(sd)


def annihilate(sd, *indices):
    """
    Annihilates an electron in the orbital `i` in a Slater determinant.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    indices : int
        The indices of the orbitals to annihilate

    Returns
    -------
    sd : int or None
        Integer that describes the occupation of a Slater determinant as a bitstring
        If the orbital is not occupied is annihilated, None is returned.
    """
    for i in indices:
        # if orbital is occupied
        if occ(sd, i):
            sd = gmpy2.bit_flip(sd, i)
        else:
            return None
    return sd


def create(sd, *indices):
    """
    Creates an electron in the orbital `i` in a Slater determinant.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    indices : int
        The indices of the orbitals to create

    Returns
    -------
    sd : int or None
        Integer that describes the occupation of a Slater determinant as a bitstring
        If the orbital is occupied is annihilated, None is returned.
    """
    for i in indices:
        if not occ(sd, i) and sd is not None:
            sd = gmpy2.bit_flip(sd, i)
        else:
            return None
    return sd


def excite(sd, *indices):
    """
    Excite electrons from first half of `indices:` to the second half of `indices`

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    indices : int
        The indices of the orbitals to annihilate and create
        The first half contain indices of orbitals to annihilate
        The second half contain indices of orbitals to create

    Returns
    -------
    excited_sd : int or None
        Integer that describes the occupation of a Slater determinant as a bitstring
    If the orbital `i` is unoccupied or the orbital `a` is occupied,
        None is returned

    Raises
    ------
    ValueError
        If the length of indices is not even
    """
    if (len(indices) % 2) != 0:
        raise ValueError("Unqual number of creators and annihilators")
    sd = annihilate(sd, *indices[:len(indices) // 2])
    sd = create(sd, *indices[len(indices) // 2:])
    return sd


def ground(nocc, norbs):
    """
    Creates a ground state Slater determinant (no occupied spin-orbitals)

    Parameters
    ----------
    n : int
        Number of occupied spin-orbitals
    norbs : int
        Total number of spin-orbitals

    Returns
    -------
    ground_sd : gmpy2.mpz instance
        Integer that describes the occupation of a Slater determinant as a bitstring

    Raises
    ------
    ValueError
        If the number of occupied orbitals is greater than the total number of orbitals
        If the total number of spin orbitals is odd

    Note
    ----
    Assumes that the spin-orbitals are ordered by energy and that the ground state Slater
    determinant is composed of the orbitals with the lowest energy
    Orders the alpha orbitals first, then the beta orbitals
    If the number of electrons is odd, then the last electron is put into an alpha orbital
    """
    if nocc > norbs:
        raise ValueError('Number of occupied spin-orbitals must be less than the total number of'
                         ' spin-orbitals')
    if norbs % 2 != 0:
        raise ValueError('Total number of spin-orbitals must be even')
    alpha_bits = gmpy2.bit_mask(nocc // 2 + nocc % 2)
    beta_bits = gmpy2.bit_mask(nocc // 2) << (norbs // 2)
    return alpha_bits | beta_bits


def is_internal_sd(sd):
    """ Checks if given Slater determinant is the same type as the one used internally in this
    module

    Parameter
    ---------
    sd
         Some representation of a Slater determinant

    Returns
    -------
    True if it is the right type
    True if it is not the right type
    """
    return isinstance(sd, type(gmpy2.mpz()))


def internal_sd(identifier):
    """ Creates a Slater determinant that is consistent with the inner workings of this module

    Parameters
    ----------
    identifier : int, list of int
        Some form of identifying a Slater determinant
        Can be an integer, whose binary form corresponds to the occupations
        Can be a list of integer, the indices of spin orbitals that are occupied

    Returns
    -------
    sd : gmpy2.mpz
        Internal representation of Slater determinant within this module

    Raises
    ------
    TypeError
        If not an integer and not an iterable
        If is an iterable of non integers
    """
    if isinstance(identifier, int):
        return gmpy2.mpz(identifier)
    elif hasattr(identifier, '__iter__'):
        identifier, test = tee(identifier, 2)
        if all(isinstance(i, int) for i in test):
            return create(gmpy2.mpz(0), *identifier)
        else:
            raise TypeError('Iterable must contain only integers to describe a Slater determinant')
    elif is_internal_sd(identifier):
        return identifier
    else:
        raise TypeError('Unsupported Slater determinant form, {0}'.format(type(identifier)))


def occ_indices(sd):
    """
    Returns indices of all of the occupied orbitals

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring

    Returns
    -------
    occ_indices : tuple of int
        Tuple of indices that corresponds to the occupied orbitals
    """
    if sd is None:
        return ()
    output = [gmpy2.bit_scan1(sd, 0)]
    while output[-1] is not None:
        output.append(gmpy2.bit_scan1(sd, output[-1] + 1))
    return tuple(output[:-1])


def vir_indices(sd, norbs):
    """
    Returns indices of all of the virtual orbitals

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    norbs : int
        Total number of spin orbitals

    Returns
    -------
    occ_indices : tuple of int
        Tuple of indices that corresponds to the virtual orbitals
    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if sd is None or norbs <= 0:
        return ()
    output = [gmpy2.bit_scan0(sd, 0)]
    while output[-1] < norbs:
        output.append(gmpy2.bit_scan0(sd, output[-1] + 1))
    return tuple(output[:-1])


def shared(sd1, sd2):
    """
    Finds the orbitals shared between two Slater determinants

    Parameters
    ----------
    sd1 : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    sd2 : int
        Integer that describes the occupation of a Slater determinant as a bitstring

    Returns
    -------
    tuple of ints
        Tuple of ints are the indices of the occupied orbitals shared by the two
        Slater determinants
    """
    return sd1 & sd2


def diff(sd1, sd2):
    """
    Returns the difference between two Slater determinants

    Parameters
    ----------
    sd1 : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    sd2 : int
        Integer that describes the occupation of a Slater determinant as a bitstring

    Returns
    -------
    2-tuple of tuple of ints
        First tuple of ints are the indices of the occupied orbitals of sd1 that
        are not occupied in sd2
        Second tuple of ints are the indices of the occupied orbitals of sd2 that
        are not occupied in sd1
    """
    sd_diff = sd1 ^ sd2
    sd1_diff = sd_diff & sd1
    sd2_diff = sd_diff & sd2
    return (occ_indices(sd1_diff), occ_indices(sd2_diff))


def combine_spin(alpha_bits, beta_bits, nspatial):
    """ Constructs a Slater determinant from the occupation of alpha and beta spin orbitals

    Parameters
    ----------
    alpha_bits : int
        Integer that describes the occupation of alpha spin orbitals as a bitstring
    beta_bits : int
        Integer that describes the occupation of beta spin orbitals as a bitstring
    nspatial : int
        Total number of spatial orbitals

    Returns
    -------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Indices less than nspatial correspond to the alpha spin orbitals
        Indices greater than or equal to nspatial correspond to the beta spin orbitals

    Note
    ----
    Erratic behaviour if the total number of spatial orbitals is less than the the
    actual number (i.e. if there are any occupied orbitals with indices greater than
    nspatial)

    Raises
    ------
    ValueError
        If number of spatial orbitals is less than or equal to 0
    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if nspatial <= 0:
        raise ValueError('Number of spatial orbitals must be greater than 0')
    return alpha_bits | (beta_bits << nspatial)


def split_spin(block_sd, nspatial):
    """ Splits a Slater determinant into the alpha and beta parts

    Parameters
    ----------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Indices less than nspatial correspond to the alpha spin orbitals
        Indices greater than or equal to nspatial correspond to the beta spin orbitals
    nspatial : int
        Total number of spatial orbitals

    Returns
    -------
    alpha_bits : int
        Integer that describes the occupation of alpha spin orbitals as a bitstring
    beta_bits : int
        Integer that describes the occupation of beta spin orbitals as a bitstring

    Raises
    ------
    ValueError
        If the number of spatial orbitals is less than or equal to 0
    Note
    ----
    Erratic behaviour if the total number of spatial orbitals is less than the the
    actual number (i.e. if there are any occupied orbitals with indices greater than
    nspatial)
    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if nspatial <= 0:
        raise ValueError('Number of spatial orbitals must be greater than 0')
    alpha_bits = gmpy2.t_mod_2exp(block_sd, nspatial)
    beta_bits = block_sd >> nspatial
    return (alpha_bits, beta_bits)


def interleave_index(i, nspatial):
    """ Converts index of an orbital in block sd notation to that of interleaved
    sd notation

    Parameter
    ---------
    i : int
        Index of orbital in block sd notation
    nspatial : int
        Number of spatial orbitals

    Returns
    -------
    Index of the same orbital in interleaved notation

    Raises
    ------
    ValueError
        If the index is less than zero
        If the index is greater than or equal to the number of spin orbitals
    """
    if i < 0:
        raise ValueError('Index must be greater than or equal to zero')
    elif i >= 2*nspatial:
        raise ValueError('Index must be less than the number of spin orbitals')
    if i < nspatial:
        return 2*i
    else:
        return 2*(i - nspatial) + 1


def deinterleave_index(i, nspatial):
    """ Converts index of an orbital in interleaved sd notation to that of block sd notation

    Parameter
    ---------
    i : int
        Index of orbital in interleaved sd notation
    nspatial : int
        Number of spatial orbitals

    Returns
    -------
    Index of the same orbital in block notation

    Raises
    ------
    ValueError
        If the index is less than zero
        If the index is greater than or equal to the number of spin orbitals
    """
    if i < 0:
        raise ValueError('Index must be greater than or equal to zero')
    elif i >= 2*nspatial:
        raise ValueError('Index must be less than the number of spin orbitals')
    if i%2 == 0:
        return i//2
    else:
        return i//2 + nspatial


def interleave(block_sd, nspatial):
    """ Turns sd from block form to the shuffled form

    Block form:
        alpha1, alpha2, ..., beta1, beta2, ...
    Shuffled form:
        alpha1, beta1, alpha2, beta2, ...

    Parameters
    ----------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Indices less than nspatial correspond to the alpha spin orbitals
        Indices greater than or equal to nspatial correspond to the beta spin orbitals
    nspatial : int
        Total number of spatial orbitals

    Returns
    -------
    shuffled_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Odd indices correspond to the alpha spin orbitals
        Even indices correspond to the beta spin orbitals

    Raises
    ------
    ValueError
        If the number of spatial orbitals is less than or equal to 0

    Note
    ----
    Erratic behaviour if the total number of spatial orbitals is less than the the
    actual number (i.e. if there are any occupied orbitals with indices greater than
    nspatial)
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


def deinterleave(shuffled_sd, nspatial):
    """ Turns sd from shuffled form to the block form

    Shuffled form:
        alpha1, beta1, alpha2, beta2, ...
    Block form:
        alpha1, alpha2, ..., beta1, beta2, ...

    Parameters
    ----------
    shuffled_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Odd indices correspond to the alpha spin orbitals
        Even indices correspond to the beta spin orbitals
    nspatial : int
        Total number of spatial orbitals

    Returns
    -------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Indices less than nspatial correspond to the alpha spin orbitals
        Indices greater than or equal to nspatial correspond to the beta spin orbitals

    Raises
    ------
    ValueError
        If the number of spatial orbitals is less than or equal to 0

    Note
    ----
    Erratic behaviour if the total number of spatial orbitals is less than the the
    actual number (i.e. if there are any occupied orbitals with indices greater than
    nspatial)
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


def get_spin(sd, nspatial):
    """ Returns the spin of the given slater determinant

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    nspatial : int
        Total number of spatial orbitals

    Returns
    -------
    spin : int
        Spin of the given slaterdeterminant

    Raises
    ------
    ValueError
        If number of spatial orbitals is less than or equal to zero

    """
    alpha_bits, beta_bits = split_spin(sd, nspatial)
    return (0.5)*(total_occ(alpha_bits) - total_occ(beta_bits))


def get_seniority(sd, nspatial):
    """ Returns the seniority of the given Slater determinant

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    nspatial : int
        Total number of spatial orbitals

    Returns
    -------
    seniority : int
        Seniority of the given Slater determinant

    Raises
    ------
    ValueError
        If number of spatial orbitals is less than or equal to zero
    """
    alpha_bits, beta_bits = split_spin(sd, nspatial)
    return total_occ(alpha_bits ^ beta_bits)


#FIXME: bad location
def find_num_trans(jumbled_set, ordered_set=None, is_creator=True):
    """ Finds the number of transpositions necessary to order an arbitrary order
    of annihilators into a strictly increasing order.

    Parameters
    ----------
    jumbled_set : tuple, iterable
        Nonordered set of indices of the annihilators
    ordered_set : tuple, iterable
        Ordered set of indices of the annihilators (smallest to largest)
        If not provided, the ordered set is generated by sorting the jumbled set
        If provided, the provided ordered set is used
        If used with the creation operators, give indices from smallest to largest
        order
    is_creator : bool
        If True, the objects being ordered are treated as creators (from largest
        to smallest)
        If False, the objects being ordered are treated as annihilators (from smallest
        to largest)

    Returns
    -------
    num_trans : int
        Number of hops needed to sort the jumbled_set

    Note
    ----
    Basically, we count the number of times each annihilator needs to hop over another
    to sort it. If the indices of the annihilators to its left are greater than its
    indext, then it hops over that annihilator. Once the annihilator with the lowest index
    hops over all annihilators with a lower index to its left, the annihilator with the
    second lowest index hops, and so on, until the annihilators are ordered from
    smallest to largest indices.
    """
    jumbled_set = np.array(jumbled_set)
    if is_creator:
        jumbled_set = jumbled_set[::-1]

    # Given that the first index corresponds to the smallest number, and the
    # last index corresponds to the largest number,
    if ordered_set is None:
        # location of each number in the jumbled set
        loc_jumbled_num = np.argsort(jumbled_set)
        # ordered set
        ordered_set = jumbled_set[loc_jumbled_num]
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


#FIXME: bad location
def find_num_trans_dumb(jumbled_set, ordered_set=None, is_creator=True):
    """ Finds the number of transpositions necessary to order an arbitrary order
    of annihilators (or annihilators) into a strictly increasing order.

    Same as find_num_trans except it can loop over all numbers in the jumbled set
    twice.

    Parameters
    ----------
    jumbled_set : tuple, iterable
        Nonordered set of indices of the annihilators
    ordered_set : tuple, iterable
        Ordered set of indices of the annihilators (smallest to largest)
        If not provided, the ordered set is generated by sorting the jumbled set
        If provided, the provided ordered set is used
        If used with the creation operators, give indices from smallest to largest
        order
    is_creator : bool
        If True, the objects being ordered are treated as creators (from largest
        to smallest)
        If False, the objects being ordered are treated as annihilators (from smallest
        to largest)

    Returns
    -------
    num_trans : int
        Number of hops needed to sort the jumbled_set

    Note
    ----
    Basically, we count the number of times each annihilator needs to hop over another
    to sort it. If the indices of the annihilators to its left are greater than its
    indext, then it hops over that annihilator. Once the annihilator with the lowest index
    hops over all annihilators with a lower index to its left, the annihilator with the
    second lowest index hops, and so on, until the annihilators are ordered from
    smallest to largest indices.
    """
    if is_creator:
        jumbled_set = jumbled_set[::-1]

    num_trans = 0
    # get ordered set
    if ordered_set is None:
        ordered_set = sorted(jumbled_set)

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
