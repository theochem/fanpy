import gmpy2

"""
Functions
---------
occ
    Check if an orbital is occupied in a Slater determinant
total_occ
    Returns the total number of occupied orbitals in a Slater determinant
annihilate
    Removes an electron in a Slater determinant
create
    Adds an electron in a Slater determinant
excite
    Excites an electron from one orbital to another in a Slater determinant
ground
    Creates a Slater determinant at the ground state
occ_indices
    Returns indices of all of the occupied orbitals
vir_indices
    Returns indices of all of the virtual orbitals
shared
    Returns indices of all orbitals shared between two Slater determinants
diff
    Returns the difference between two Slater determinants
combine_spin
    Constructs a Slater determinant in block form from alpha and beta occupations
split_spin
    Constructs the alpha and beta occupations from the Slater determinant
interleave
    Converts Slater determinants from block form to shuffled form
deinterleave
    Converts Slater determinants from shuffled form to block form
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

def ground(n, norbs):
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

    Note
    ----
    Assumes that the spin-orbitals are ordered by energy and that the ground state Slater determinant
    is composed of the orbitals with the lowest energy
    Orders the alpha orbitals first, then the beta orbitals
    If the number of electrons is odd, then the last electron is put into an alpha orbital
    """
    assert n<=norbs, 'Number of occupied spin-orbitals must be less than the total number of spin-orbitals'
    assert norbs%2 == 0, 'Total number of spin-orbitals must be even'
    alpha_bits = gmpy2.bit_mask(n//2+n%2)
    beta_bits = gmpy2.bit_mask(n//2) << (norbs//2)
    return alpha_bits | beta_bits

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
        output.append(gmpy2.bit_scan1(sd, output[-1]+1))
    return tuple(output[:-1])

def vir_indices(sd, norbs):
    """
    Returns indices of all of the virtual orbitals

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
    norbs : int
        Total number of orbitals

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

def combine_spin(alpha_bits, beta_bits, norbs):
    """ Constructs a Slater determinant from the occupation of alpha and beta spin orbitals

    Parameters
    ----------
    alpha_bits : int
        Integer that describes the occupation of alpha spin orbitals as a bitstring
    beta_bits : int
        Integer that describes the occupation of beta spin orbitals as a bitstring
    norbs : int
        Total number of spatial orbitals

    Returns
    -------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Indices less than norbs correspond to the alpha spin orbitals
        Indices greater than or equal to norbs correspond to the beta spin orbitals

    Note
    ----
    Erratic behaviour if the total number of spatial orbitals is less than the the
    actual number (i.e. if there are any occupied orbitals with indices greater than
    norbs)
    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    assert norbs > 0, 'Number of spatial orbitals must be greater than 0'
    return alpha_bits | (beta_bits << norbs)

def split_spin(block_sd, norbs):
    """ Splits a Slater determinant into the alpha and beta parts

    Parameters
    ----------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Indices less than norbs correspond to the alpha spin orbitals
        Indices greater than or equal to norbs correspond to the beta spin orbitals
    norbs : int
        Total number of spatial orbitals

    Returns
    -------
    alpha_bits : int
        Integer that describes the occupation of alpha spin orbitals as a bitstring
    beta_bits : int
        Integer that describes the occupation of beta spin orbitals as a bitstring

    Note
    ----
    Erratic behaviour if the total number of spatial orbitals is less than the the
    actual number (i.e. if there are any occupied orbitals with indices greater than
    norbs)
    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    assert norbs > 0, 'Number of spatial orbitals must be greater than 0'
    alpha_bits = gmpy2.t_mod_2exp(block_sd, norbs)
    beta_bits = block_sd >> norbs
    return (alpha_bits, beta_bits)

def interleave(block_sd, norbs):
    """ Turns sd from block form to the shuffled form

    Block form:
        alpha1, alpha2, ..., beta1, beta2, ...
    Shuffled form:
        alpha1, beta1, alpha2, beta2, ...

    Parameters
    ----------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Indices less than norbs correspond to the alpha spin orbitals
        Indices greater than or equal to norbs correspond to the beta spin orbitals
    norbs : int
        Total number of spatial orbitals

    Returns
    -------
    shuffled_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Odd indices correspond to the alpha spin orbitals
        Even indices correspond to the beta spin orbitals

    Note
    ----
    Erratic behaviour if the total number of spatial orbitals is less than the the
    actual number (i.e. if there are any occupied orbitals with indices greater than
    norbs)
    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    assert norbs > 0, 'Number of spatial orbitals must be greater than 0'
    shuffled_sd = gmpy2.mpz(0)
    for i in range(norbs):
        if gmpy2.bit_test(block_sd, i):
            shuffled_sd |= 1 << 2*i
        if gmpy2.bit_test(block_sd, i+norbs):
            shuffled_sd |= 1 << 2*i+1
    return shuffled_sd

def deinterleave(shuffled_sd, norbs):
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
    norbs : int
        Total number of spatial orbitals

    Returns
    -------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring
        Indices less than norbs correspond to the alpha spin orbitals
        Indices greater than or equal to norbs correspond to the beta spin orbitals

    Note
    ----
    Erratic behaviour if the total number of spatial orbitals is less than the the
    actual number (i.e. if there are any occupied orbitals with indices greater than
    norbs)
    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    assert norbs > 0, 'Number of spatial orbitals must be greater than 0'
    block_sd = gmpy2.mpz(0)
    for i in range(norbs):
        if gmpy2.bit_test(shuffled_sd, 2*i):
            block_sd |= 1 << i
        if gmpy2.bit_test(shuffled_sd, 2*i+1):
            block_sd |= 1 << i+norbs
    return block_sd
