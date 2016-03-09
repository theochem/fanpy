from __future__ import absolute_import, division, print_function

import numpy as np
from ..lib import doci, apig
from .ci import CI
from .base import Base


class DOCI(CI):
    """
    Configuration Interaction Wavefucntion
    See `prowl.lib.ci`

    """

    # Properties
    dtype = np.float64
    bounds = (-1, 1)

    # Bind methods
    generate_pspace = doci.generate_pspace
    hamiltonian = apig.hamiltonian
    hamiltonian_deriv = apig.hamiltonian_deriv
