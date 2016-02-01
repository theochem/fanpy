from __future__ import absolute_import, division, print_function

from ..lib import apig
from .base import Base


class APIG(Base):
    """
    The Antisymmetrized Product of Interacting Geminals (APIG) wavefunction.
    See `prowl.lib.apig`.

    """

    # Bind methods
    generate_guess = apig.generate_guess
    generate_pspace = apig.generate_pspace
    generate_view = apig.generate_view
    hamiltonian = apig.hamiltonian
    objective = apig.objective
    overlap = apig.overlap
