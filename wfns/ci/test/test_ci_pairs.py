""" Tests wfns.ci.ci_pairs
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from wfns.ci.ci_pairs import CIPairs
from wfns.sd_list import sd_list


def test_generate_civec():
    """ Tests wfns.ci.ci_pairs.CIPairs.generate_civec
    """
    test = CIPairs(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    assert test.generate_civec() == sd_list(test.nelec, test.nspatial, num_limit=None, spin=0,
                                            seniority=0, exc_orders=[2])

#FIXME: check to_ap1rog (after checking ap1rog)
