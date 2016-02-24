from __future__ import absolute_import, division, print_function

import numpy as np
from ..lib import apig
from .base import Base


class APIG(Base):
    """
    The Antisymmetrized Product of Interacting Geminals (APIG) wavefunction.
    See `prowl.lib.apig`.

    """

    # Properties
    dtype = np.complex128
    bounds = (-1, 1)

    # Bind methods
    generate_guess = apig.generate_guess
    generate_pspace = apig.generate_pspace
    generate_view = apig.generate_view
    hamiltonian = apig.hamiltonian
    hamiltonian_deriv = apig.hamiltonian_deriv
    jacobian = apig.jacobian
    objective = apig.objective
    overlap = apig.overlap
    overlap_deriv = apig.overlap_deriv
    truncate = apig.truncate
    density_matrix = apig.density_matrix
