#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from Geminal import *


matrix = np.array([ [1,2,3],
                    [4,5,6],
                    [7,8,9] ])
assert Geminal.permanent(matrix) == 450


npairs = 3
norbs = 9
ham = (np.zeros((norbs, norbs)), np.zeros((norbs, norbs, norbs, norbs)), 1.0)
gem = Geminal(npairs, norbs)
result = gem.solve( dets=gem.pspace,
                    ham=ham,
                    x0=np.zeros(npairs*norbs + 1),
                    solver=quasinewton )
assert result['success']


# vim: set textwidth=90 :
