from __future__ import absolute_import, division, print_function

from ..lib import apr2g
from .apig import APIG


class APr2G(APIG):
    """
    The Antisymmetrized Product of rank-2 Geminals (APr2G) wavefunction.
    See `prowl.lib.apr2g`.

    """

    # Bind methods
    _update_C = apr2g._update_C
    generate_guess = apr2g.generate_guess
    generate_view = apr2g.generate_view
    objective = apr2g.objective
    overlap = apr2g.overlap
