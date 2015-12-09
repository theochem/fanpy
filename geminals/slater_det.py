"""
Functions for treatment of Slater determinants as binary integers.

"""

from __future__ import absolute_import, division, print_function


def remove_orbs(phi, *indices):
    """
    Remove an electron from spin-orbitals in a Slater determinant.

    Parameters
    ----------
    phi : int
        An integer that, in binary representation, describes the Slater determinant.
    indices : iterable of ints
        Iterable of the indices of the orbitals from which an electron is to be removed.

    Returns
    -------
    phi : int

    Notes
    -----
    Attempting to remove an electron from an unoccupied orbital annihilates the Slater
    determinant.  This is represented by returning None.

    """

    for occ_index in indices:
        if is_occupied(phi, occ_index):
            phi &= ~(1 << occ_index)
        else:
            return None
    return phi


def add_orbs(phi, *indices):
    """
    Add an electron to spin-orbitals in a Slater determinant.

    See remove_orbs().

    Notes
    -----
    Attempting to create an electron in an occupied orbital annihilates the Slater
    determinant.  This is represented by returning None.

    """

    for virt_index in indices:
        if phi is None or is_occupied(phi, virt_index):
            return None
        else:
            phi |= 1 << virt_index
    return phi


def excite_orbs(phi, *indices):
    """
    Excite electrons in a Slater determinant.

    See remove_orbs() and add_orbs().

    Raises
    ------
    AssertionError
        If indices do not have even number of elements (i.e., cannot be divided in half to
        obtain an equal number of annihilation and creation indices).

    """

    assert (len(indices) % 2) == 0, \
        "An equal number of annihilations and creations must occur."
    frontier = len(indices) // 2
    phi = remove_orbs(phi, *indices[:frontier])
    phi = add_orbs(phi, *indices[frontier:])
    return phi


def remove_pairs(phi, *indices):
    """
    Remove an alpha/beta electron pair from spatial orbitals in a Slater determinant.

    See remove_orbs().

    """

    for spatial_index in indices:
        alpha_index = spatial_index * 2
        beta_index = spatial_index * 2 + 1
        phi = remove_orbs(phi, alpha_index, beta_index)
    return phi


def add_pairs(phi, *indices):
    """
    Remove an alpha/beta electron pair from spatial orbitals in a Slater determinant.

    See add_orbs().

    """

    for spatial_index in indices:
        alpha_index = spatial_index * 2
        beta_index = spatial_index * 2 + 1
        phi = add_orbs(phi, alpha_index, beta_index)
    return phi


def excite_pairs(phi, *indices):
    """
    Excite alpha/beta electron pairs in a Slater determinant.

    See remove_pairs() and add_pairs().

    """

    assert len(indices) % 2 == 0, \
        "An equal number of annihilations and creations must occur."
    frontier = len(indices) // 2
    phi = remove_pairs(phi, *indices[:frontier])
    phi = add_pairs(phi, *indices[frontier:])
    return phi


def is_occupied(phi, index):
    """
    Check if a spin-orbital is occupied in a Slater determinant.

    Parameters
    ----------
    phi : int
        An integer that, in binary representation, describes the Slater determinant.
    index : int
        The index of the spin-orbital to check.

    Returns
    -------
    occupancy : bool

    """

    if phi is None:
        return None
    else:
        return bool(phi & (1 << index))


def is_pair_occupied(phi, index):
    """
    Check if an alpha/beta orbital pair are both occupied in a slater determinant.

    Parameters
    ----------
    phi : int
        An integer that, in binary representation, describes the Slater determinant.
    index : int
        The index of the spatial orbital to check (spin-orbitals 2*index and
        (2*index + 1) are checked).

    Returns
    -------
    occupancy : bool

    """

    return is_occupied(phi, 2 * index) and is_occupied(phi, 2 * index + 1)

# vim: set textwidth=90 :
