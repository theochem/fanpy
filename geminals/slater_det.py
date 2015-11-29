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
            return None
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
        if bin_sd is None or is_occupied(bin_sd, vir_index):
            return None
        else:
            bin_sd |= 1 << vir_index
    return bin_sd

def excite_orbs(bin_sd, *indices):
    """ Excites orbitals in Slater determinant

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        Slater determinant

    indices : list of int
        First half contains the indices for the occupied orbitals
        Second half contains the indices for the unoccupied orbitals
        Occupied orbitals are replaced by the unoccupied orbitals
        Start from 0

    Returns
    -------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        excited Slater determinant

    Raises
    ------
    AssertionError
        If indices do not have even number of elements (i.e. cannot be divided
        in half)
    """
    assert (len(indices) % 2) == 0, \
        "An equal number of annihilations and creations must occur."
    halfway = len(indices)//2
    # Remove occupieds
    bin_sd = remove_orbs(bin_sd, *indices[:halfway])
    # Add virtuals
    bin_sd = add_orbs(bin_sd, *indices[halfway:])
    return bin_sd


def remove_pairs(bin_sd, *indices):
    """ Removes alpha and beta orbitals from Slater determinent specified by indices

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the spin orbitals used to make the
        Slater determinant
        Even indices from the right (starting from 0) describe the alpha orbitals
        Odd indices from the right (starting from 0) describe the beta orbitals

    indices : list of int
        List of indices that describes the spatial orbital that will be removed
        Start from 0

    Returns
    -------
    bin_sd : int
        Integer that in binary form describes the Slater determinant that with
        the specified orbitals removed
        Zero if the selected orbitals are not occupied

    Example
    -------
    remove_pairs(bin_sd, 0) would remove the 0th spatial orbital, i.e. 0th and
    1st spin orbital, or 0th alpha and 0th beta orbitals.

    Note
    ----
    This code ASSUMES a certain structure for the Slater determinant indexing.
    If the alpha and beta orbitals are ordered differently, then this code will
    break
    """
    for spatial_index in indices:
        alpha_index = spatial_index*2
        beta_index = spatial_index*2 + 1
        bin_sd = remove_orbs(bin_sd, alpha_index, beta_index)
    return bin_sd

def add_pairs(bin_sd, *indices):
    """ Adds alpha and beta orbitals from Slater determinent specified by indices

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the spin orbitals used to make the
        Slater determinant
        Even indices from the right (starting from 0) describe the alpha orbitals
        Odd indices from the right (starting from 0) describe the beta orbitals

    indices : list of int
        List of indices that describes the spatial orbital that will be added
        Start from 0

    Returns
    -------
    bin_sd : int
        Integer that in binary form describes the Slater determinant that with
        the specified orbitals addd
        Zero if the selected orbitals are not occupied

    Example
    -------
    add_pairs(bin_sd, 0) would add the 0th spatial orbital, i.e. 0th and
    1st spin orbital, or 0th alpha and 0th beta orbitals.

    Note
    ----
    This code ASSUMES a certain structure for the Slater determinant indexing.
    If the alpha and beta orbitals are ordered differently, then this code will
    break
    """
    for spatial_index in indices:
        alpha_index = spatial_index*2
        beta_index = spatial_index*2 + 1
        bin_sd = add_orbs(bin_sd, alpha_index, beta_index)
    return bin_sd

def excite_pairs(bin_sd, *indices):
    """ Excites the alpha beta pairs of a Slater determinant

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the spin orbitals used to make the
        Slater determinant
        Even indices from the right (starting from 0) describe the alpha orbitals
        Odd indices from the right (starting from 0) describe the beta orbitals

    indices : list of int
        First half contains the indices for the occupied spatial orbitals
        Second half contains the indices for the unoccupied spatial orbitals
        Occupied spatial orbitals are replaced by the unoccupied spatial orbitals
        Start from 0

    Returns
    -------
    bin_sd : int
        Integer that in binary form describes the spin orbitals used to make the
        excited Slater determinant

    Raises
    ------
    AssertionError
        If indices do not have even number of elements (i.e. cannot be divided
        in half)

    Example
    -------
    excite_pairs(bin_sd, 0, 1) would replace the 0th spatial orbital with the
    1st spatial orbital, i.e. 0th and 1st spin orbital, or 0th alpha and 0th
    beta orbitals are replaced with the 2nd and 3rd spin orbitals, or the 1st
    alpha and beta orbitals

    Note
    ----
    This code ASSUMES a certain structure for the Slater determinant indexing.
    If the alpha and beta orbitals are ordered differently, then this code will
    break

    """
    assert (len(indices) % 2) == 0, \
        "An equal number of annihilations and creations must occur."
    halfway = len(indices)//2
    # Remove occupieds
    bin_sd = remove_pairs(bin_sd, *indices[:halfway])
    # Add virtuals
    bin_sd = add_pairs(bin_sd, *indices[halfway:])
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
    if bin_sd is None:
        return None
    else:
        return bool(bin_sd & (1 << orb_index))

def is_pair_occupied(bin_sd, orb_index):
    """ Checks if both alpha and beta orbital pair is used in the slater determinant

    Parameters
    ----------
    bin_sd : int
        Integer that in binary form describes the orbitals used to make the
        Slater determinant

    orb_index : int
        Index of the spatial orbital that is checked
        Starts from 0

    Returns
    -------
    bool
        True if both alpha and beta orbitals are in SD
        False if not both alpha and beta orbitals are not in SD

    Example
    -------
    is_pair_occupied(bin_sd, 0) would check if the 0th spatial orbital is included,
    i.e. 0th and 1st spin orbitals, or 0th alpha and beta orbitals

    Note
    ----
    This code ASSUMES a certain structure for the Slater determinant indexing.
    If the alpha and beta orbitals are ordered differently, then this code will
    break
    """
    return is_occupied(bin_sd, 2*orb_index) and is_occupied(bin_sd, 2*orb_index+1)
