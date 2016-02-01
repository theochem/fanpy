from __future__ import absolute_import, division, print_function

from ..lib import ap1rog
from .apig import APIG


class AP1roG(APIG):
    """
    The Antisymmetrized Product of 1-reference-orbital Geminals (AP1roG) wavefunction.
    See `prowl.lib.ap1rog`.

    """

    # Bind methods
    generate_guess = ap1rog.generate_guess
    generate_pspace = ap1rog.generate_pspace
    generate_view = ap1rog.generate_view
    overlap = ap1rog.overlap
    objective = ap1rog.objective
