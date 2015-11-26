""" Module for containing Slater determinant related things.

It may be useful to create a Slaterdeterminant class that contains these binary
representations of the Slater determinant and contains information on the
orbitals from which they are constructed

"""

def remove_orbs(bin_sd, *indices):
    """ Removes orbitals from Slater determinent specifies by indices

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        Slater determinant

    indices : list of int
        List of indices that describes the orbital that will be removed
        Start from 0

    Returns
    -------
    bin_sd : int
        Integer that in binary form describes the Slater determinant that with
        the specified orbitals removed
        Zero if the selected orbitals are not occupied
    """
    for occ_index in indices:
        if is_occupied(bin_sd, occ_index):
            bin_sd &= ~(1 << occ_index)
        else:
            return 0
    return bin_sd

def add_orbs(bin_sd, *indices):
    """ Adds orbitals to Slater determinant specifies by indices

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        Slater determinant

    indices : list of int
        List of indices that describes the orbital that will be removed
        Start from 0

    Returns
    -------
    bin_sd : int
        Integer that in binary form describes the Slater determinant that with
        the specified orbitals removed
        Zero if the selected orbitals are occupied
    """
    for vir_index in indices:
        if is_occupied(bin_sd, vir_index):
            return 0
        else:
            bin_sd |= 1 << vir_index
    return bin_sd

def excite(bin_sd, *indices):
    """ Doubly excites Slater determinant

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        Slater determinant

    index_a : int
        Index for the occupied orbital
        Start from 0

    index_b : int
        Index for the occupied orbital
        Start from 0

    index_v : int
        Index for the virtual orbital
        Start from 0

    index_v : int
        Index for the virtual orbital
        Start from 0

    Returns
    -------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        excited Slater determinant
    """
    assert (len(indices) % 2) == 0, \
        "An equal number of annihilations and creations must occur."
    halfway = len(indices)//2
    # Add virtuals (Needs to be first because if it was last, we can still add virtuals)
    bin_sd = add_orbs(bin_sd, *indices[halfway:])
    # Remove occupieds
    bin_sd = remove_orbs(bin_sd, *indices[:halfway])
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
