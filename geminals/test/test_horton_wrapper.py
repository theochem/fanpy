#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from test.common import *
from horton_wrapper import *

@test
def basic_test():
    from_horton(fn='test/h2.xyz', basis='3-21g', nocc=1)

run_tests()
 
# vim: set textwidth=90 :
