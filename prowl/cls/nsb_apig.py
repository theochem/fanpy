from __future__ import absolute_import, division, print_function

import numpy as np
from ..lib import nsb_apig
from .apig import APIG


class nsbAPIG(APIG):
    """
    The particle number symmetry-broken APIG class.
    See `prowl.lib.nsb_apig`.

    """

    # Bind methods
    __init__ = nsb_apig.__init__
    jacobian = nsb_apig.jacobian
    objective = nsb_apig.objective
