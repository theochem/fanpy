"""
Unit tests for geminals.horton_wrapper.

"""

from __future__ import absolute_import, division, print_function
from geminals.test.common import *
from geminals.horton_wrapper import ap1rog_from_horton


@test
def basic_test():
    ap1rog_from_horton(fn="test/h2.xyz", basis="3-21g", npairs=1)


run_tests()

# vim: set textwidth=90 :
