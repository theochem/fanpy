""" Module for containing Slater determinant related things.

It may be useful to create a Slaterdeterminant class that contains these binary
representations of the Slater determinant and contains information on the
orbitals from which they are constructed

"""

def excite_single(bin_sd, index_a, index_v):
    """ Singly excites Slater determinant phi

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        Slater determinant

    index_a : int
        Index for the occupied orbital

    index_v : int
        Index for the virtual orbital

    Returns
    -------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        excited Slater determinant
    """
    # if the indices for the occupied orbitals are not occupied
    # if the indices for the virutal orbitals are occupied
    if (not is_occupied(bin_sd, index_a) or
        is_occupied(bin_sd, index_v)):
        return 0
    # Remove occupied
    bin_sd = bin_sd & ~(1 << index_a)
    # Add virtuals
    bin_sd = bin_sd | (1 << index_v)
    return bin_sd


def excite_double(bin_sd, index_a, index_b, index_v, index_w):
    """ Doubly excites Slater determinant 

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        Slater determinant

    index_a : int
        Index for the occupied orbital

    index_b : int
        Index for the occupied orbital

    index_v : int
        Index for the virtual orbital

    index_v : int
        Index for the virtual orbital

    Returns
    -------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        excited Slater determinant
    """
    # if the indices for the occupied orbitals are not occupied
    # if the indices for the virutal orbitals are occupied
    if (not is_occupied(bin_sd, index_a) or
        not is_occupied(bin_sd, index_b) or
        is_occupied(bin_sd, index_v) or
        is_occupied(bin_sd, index_w)):
        return 0
    # Remove occupied
    bin_sd = bin_sd & ~(1 << index_a) & ~(1 << index_b)
    # Add virtuals
    bin_sd = bin_sd | (1 << index_v) | (1 << index_w)
    return bin_sd


def is_occupied(bin_sd, orb_index):
    """ Checks if orbital is used in the slater determinant

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        Slater determinant

    orb_index : int
        Index of the orbital that is checked
        Starts from 0

    Returns
    -------
    bool
        True if orbital is in SD
        False if orbital is not in SD
    """
    return bool(bin_sd & (1 << orb_index))
