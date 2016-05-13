import gmpy2

"""
Functions
---------
occ
    Check if an orbital is occupied in a Slater determinant
total_occ
    Returns the total number of occupied orbitals in a Slater determinant
occ_indices
    Returns indices of all of the occupied orbitals
vir_indices
    Returns indices of all of the virtual orbitals
annihilate
    Removes an electron in a Slater determinant
create
    Adds an electron in a Slater determinant
excite
    Excites an electron from one orbital to another in a Slater determinant
ground
    Creates a Slater determinant at the ground state
"""

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
    Excite an electron in orbital `i` to orbital `a`.

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
    AssertionError
        If the length of indices is not even
    """
    assert (len(indices)%2)==0, "Unqual number of creators and annihilators"
    sd = annihilate(sd, *indices[:len(indices)//2])
    sd = create(sd, *indices[len(indices)//2:])
    return sd

def ground(n):
    """
    Creates a ground state Slater determinant (no occupied orbitals)

    Parameters
    ----------
    n : int
        Number of occupied orbitals

    Returns
    -------
    ground_sd : gmpy2.mpz instance
        Integer that describes the occupation of a Slater determinant as a bitstring

    Note
    ----
    Assumes that the orbitals are ordered by energy and that the ground state Slater determinant
    is composed of the orbitals with the lowest energy
    """
    return gmpy2.bit_mask(n)

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
        return []
    output = [gmpy2.bit_scan1(sd, 0)]
    while output[-1] is not None:
        output.append(gmpy2.bit_scan1(sd, output[-1]+1))
    return tuple(output[:-1])

def vir_indices(sd, k):
    """
    Returns indices of all of the virtual orbitals

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    k : int
        Total number of orbitals

    Returns
    -------
    occ_indices : tuple of int
        Tuple of indices that corresponds to the virtual orbitals

    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if sd is None or k <= 0:
        return []
    output = [gmpy2.bit_scan0(sd, 0)]
    while output[-1] < k:
        output.append(gmpy2.bit_scan0(sd, output[-1]+1))
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
