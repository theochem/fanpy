#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from test.common import run_tests
from horton_wrapper import *

def basic_test():
    from_horton(fn='test/h2.xyz', basis='3-21g', nocc=1)

tests = [ basic_test,
        ]

run_tests(tests)
 
# vim: set textwidth=90 :
