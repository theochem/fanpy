"""Antisymmetric Product of Rank-2 (APr2G) Geminals wavefunction."""
from __future__ import absolute_import, division, print_function
from wfns.wavefunction.geminals.rank2_geminal import RankTwoGeminal
from wfns.wavefunction.geminals.apig import APIG
from wfns.wrapper.docstring import docstring_class

__all__ = []


# FIXME: docstring copied from APIG
@docstring_class(indent_level=1)
class APr2G(RankTwoGeminal, APIG):
    """Antisymmetrized Product of Rank-2 Geminals (APr2G).

    APIG wavefunction with rank-2 geminal coefficient matrix.

    """
    pass
