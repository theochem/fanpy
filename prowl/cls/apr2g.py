from __future__ import absolute_import, division, print_function

import numpy as np
from ..lib import apr2g
from ..lib import ap1rog
from .apig import APIG


class APr2G(APIG):
    """
    The Antisymmetrized Product of rank-2 Geminals (APr2G) wavefunction.
    See `prowl.lib.apr2g`.

    """

    # Properties
    dtype = np.complex128
    bounds = (-np.inf, np.inf)

    # Bind methods
    generate_guess = apr2g.generate_guess
    generate_pspace = ap1rog.generate_pspace
    generate_view = apr2g.generate_view
    hamiltonian_deriv = apr2g.hamiltonian_deriv
    jacobian = apr2g.jacobian
    objective = apr2g.objective
    overlap = apr2g.overlap
    overlap_deriv = apr2g.overlap_deriv
    solve = apr2g.solve
