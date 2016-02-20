from __future__ import absolute_import, division, print_function


def annihilate(sd, i):
    """
    Annihilate an electron from a spin-orbital in a Slater determinant.

    Parameters
    ----------
    sd : int
        The Slater Determinant upon which to operate.
    i : int
        The index of the spin-orbital upon which to operate.

    Returns
    -------
    sd : int or None
        The modified Slater Determinant.  If the Slater determinant is
        annihilated, None is returned.

    """

    if occupation(sd, i):
        sd = sd & (~(1 << i))
    else:
        sd = None
    return sd


def create(sd, a):
    """
    Create an electron in a spin-orbital in a Slater determinant.
    See `prowl.utils.slater.annihilate`.

    """

    if sd is None or occupation(sd, a):
        sd = None
    else:
        sd = sd | (1 << a)
    return sd


def excite(sd, i, a):
    """
    Excite an electron in a Slater determinant.
    See `prowl.utils.slater.annihilate`.

    """

    sd = annihilate(sd, i)
    sd = create(sd, a)
    return sd


def annihilate_pair(sd, i):
    """
    Annihilate an alpha/beta electron pair from a spatial orbital in a Slater
    determinant.
    See `prowl.utils.slater.annihilate`.

    """

    sd = annihilate(sd, 2 * i)
    sd = annihilate(sd, 2 * i + 1)
    return sd


def create_pair(sd, a):
    """
    Create an alpha/beta electron pair in a spatial orbital in a Slater
    determinant.
    See `prowl.utils.slater.annihilate`.

    """

    sd = create(sd, 2 * a)
    sd = create(sd, 2 * a + 1)
    return sd


def excite_pair(sd, i, a):
    """
    Excite an alpha/beta electron pair in a spatial orbital in a Slater
    determinant.
    See `prowl.utils.slater.annihilate`.

    """

    sd = annihilate(sd, 2 * i)
    sd = annihilate(sd, 2 * i + 1)
    sd = create(sd, 2 * a)
    sd = create(sd, 2 * a + 1)
    return sd


def occupation(sd, i):
    """
    Check if a spin-orbital is occupied in a Slater determinant.
    See `prowl.utils.slater.annihilate`.

    """

    if sd is None:
        return False
    else:
        return bool(sd & (1 << i))


def occupation_pair(sd, i):
    """
    Check if an alpha/beta orbital pair are both occupied in a slater determinant.
    See `prowl.utils.slater.annihilate`.

    """

    return occupation(sd, 2 * i) and occupation(sd, 2 * i + 1)


def number(sd):
    """
    Count the number of electrons in a Slater determinant.
    See `prowl.utils.slater.annihilate`.

    """

    if sd is None:
        return 0
    else:
        ones = 0
        while sd > 0:
            if (sd & 1) is 1:
                ones += 1
            sd >>= 1
        return ones


def create_multiple(sd, *indices):
    """
    Create multiple electrons in a Slater determinant.
    See `prowl.utils.slater.annihilate`.

    """
    for index in indices:
        sd = create(sd, index)
    return sd

def create_multiple_pairs(sd, *indices):
    """
    Create multiple electron pairs in a Slater determinant.
    See `prowl.utils.slater.annihilate`.

    """
    for index in indices:
        sd = create_pair(sd, index)
    return sd
