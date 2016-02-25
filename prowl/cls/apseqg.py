from __future__ import absolute_import, division, print_function

import numpy as np
from ..lib import apseqg
from ..lib import apig
from .base import Base


class APseqG(Base):
    """
    The Antisymmetrized Product of Sequential Geminals (APseqG) wavefunction.
    See `prowl.lib.apseqg`.

    """

    # Properties
    dtype = np.complex128
    bounds = (-1, 1)

    # Bind methods
    generate_guess = apseqg.generate_guess
    generate_pspace = apseqg.generate_pspace
    generate_view = apseqg.generate_view
    hamiltonian = apseqg.hamiltonian
    hamiltonian_deriv = apseqg.hamiltonian_deriv
    jacobian = apseqg.jacobian
    objective = apseqg.objective
    overlap = apseqg.overlap
    overlap_deriv = apseqg.overlap_deriv
