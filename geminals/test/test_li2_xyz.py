#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from Geminal import *
from HortonWrapper import *


file = 'test/li2.xyz'
basis = 'sto-3g'
nocc = 3
input = from_horton(file=file, basis=basis, nocc=nocc)
maxiter = 100

gem = Geminal(nocc, input['basis'].nbasis)
guess = np.zeros(gem.npairs*gem.norbs + 1)
guess[0] += input['energy']
guess[1:] += input['coeffs'].ravel()

dets = [ 0b111111 ] + [ gem.excite_single(0b111111, p, q) for q in range(gem.norbs) 
        for p in range(gem.norbs) if q != p ] + gem.pspace

dets = list(set(dets))

options = { #'method': 'anderson',
            'options': { 'maxiter':maxiter,
                         'disp': True,
                       },
          }

result = gem( dets=dets,
              ham=input['ham'],
              x0=guess,
              solver=lstsq,
              options=options )


# vim: set textwidth=90 :
