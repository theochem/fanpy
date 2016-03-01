from __future__ import absolute_import, division, print_function

import numpy as np
from ..lib import apset1rog
from ..lib import apseqg
from ..lib import ap1rog
from .base import Base


class APset1roG(Base):
    """
    The Antisymmetrized Product of Sequential Geminals (APseqG) wavefunction.
    See `prowl.lib.apseqg`.

    """

    # Properties
    dtype = np.float64
    bounds = (-1, 1)

    # Bind methods
    generate_guess = apset1rog.generate_guess
    #generate_pspace = ap1rog.generate_pspace
    generate_pspace = apset1rog.generate_pspace
    generate_view = apset1rog.generate_view
    hamiltonian = apseqg.hamiltonian
    hamiltonian_deriv = apseqg.hamiltonian_deriv
    jacobian = apset1rog.jacobian
    objective = apset1rog.objective
    overlap = apset1rog.overlap
    overlap_deriv = apset1rog.overlap_deriv
