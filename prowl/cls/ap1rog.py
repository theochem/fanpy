from __future__ import absolute_import, division, print_function

import numpy as np
from ..lib import ap1rog
from .apig import APIG


class AP1roG(APIG):
    """
    The Antisymmetrized Product of 1-reference-orbital Geminals (AP1roG) wavefunction.
    See `prowl.lib.ap1rog`.

    """

    # Properties
    dtype = np.float64
    bounds = (-1, 1)

    # Bind methods
    generate_guess = ap1rog.generate_guess
    generate_pspace = ap1rog.generate_pspace
    generate_view = ap1rog.generate_view
    overlap = ap1rog.overlap
    overlap_deriv = ap1rog.overlap_deriv
