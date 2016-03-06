from __future__ import absolute_import, division, print_function

import numpy as np
from ..lib import ci
from .base import Base


class CI(Base):
    """
    Configuration Interaction Wavefucntion
    See `prowl.lib.ci`

    """

    # Properties
    dtype = np.complex128
    bounds = (-1, 1)

    # Bind methods
    __init__ = ci.__init__
    generate_guess = ci.generate_guess
    generate_sds = ci.generate_sds
    #hamiltonian = ci.hamiltonian
    #hamiltonian_deriv = ci.hamiltonian_deriv
    #jacobian = ci.jacobian
    #objective = ci.objective
    #overlap = ci.overlap
    #overlap_deriv = ci.overlap_deriv
