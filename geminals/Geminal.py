#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import numpy as np
from itertools import combinations, permutations
from Newton import newton
from scipy.optimize import root as quasinewton
from scipy.optimize import minimize as lstsq


class Geminal(object):

    def __init__(self, npairs, norbs, pspace=None):
        self.npairs = npairs
        self.norbs = norbs
        self.coeffs = None
        self.pspace = pspace if (pspace is not None) else self.generate_pspace()


    def __call__(self, x0, one, two, core, dets=None, jac=None, solver=lstsq, options={}):
        if solver is lstsq:
            objective = self.lstsq
        else:
            if len(dets) != x0.size:
                print("Warning: length of 'dets' should be length of guess, 1 + P*K.")
                print("Using the first {} determinants in 'dets'.".format(x0.size))
            objective = self.nonlin
        result = solver(objective, x0, jac=jac, args=(one, two, core, dets), **options)
        self.coeffs = result['x'][1:].reshape(self.npairs, self.norbs)
        return result


    @property
    def pspace(self):
        return self._pspace


    @pspace.setter
    def pspace(self, value):
        for phi in value:
            for i in range(0, self.norbs, 2):
                assert self.occupied(phi, i) == self.occupied(phi, i + 1), \
                    "Determinants in APIG projection space must be doubly occupied."
        self._pspace = value


    def generate_pspace(self):
        pspace = []
        for pairs in combinations(range(self.norbs), self.npairs):
            # Python's sum is used here because NumPy's sum doesn't respect
            # arbitrary-precision Python longs
            pspace.append(sum([ 2**(2*i) + 2**(2*i + 1) for i in pairs ]))
        return pspace


    @staticmethod
    def permanent(matrix):
        assert matrix.shape[0] is matrix.shape[1], \
            "Cannot compute the permanent of a non-square matrix."
        permanent = 0
        for permutation in permutations(range(matrix.shape[0])):
            index1, term = 0, 1
            for index0 in permutation:
                term *= matrix[index0][index1]
                index1 += 1
            permanent += term
        return permanent


    def overlap(self, phi, matrix):
        if phi == 0:
            return 0
        elif phi not in self.pspace:
            return 0
        else:
            # The ones in phi's binary representation determine the columns (orbitals) of
            # the coefficient matrix for which we want to evaluate the permanent; only
            # test the a-spin orbital in each pair because the geminal coefficients of
            # APIG correspond to electron pairs
            columns = [ i for i in range(self.norbs) if self.occupied(phi, 2*i) ]
            overlap = self.permanent(matrix[:,columns])
            return overlap


    def phi_H_psi(self, phi, C, one, two, core):
        result = core
        t0 = 0.0
        for p in range(self.norbs):
            number = 0.0
            if self.occupied(phi, 2*p):
                number += 1
            if self.occupied(phi, 2*p + 1):
                number += 1
            if number:
                number*= one[p,p]
            t0 += number

        t1 = 0.0
        for p in range(self.norbs):
            if self.occupied(phi, 2*p) and self.occupied(phi, 2*p + 1):
                for q in range(self.norbs):
                    excitation = self.excite(phi, 2*p, 2*p + 1, 2*q, 2*q + 1)
                    t1 += two[p,p,q,q]*self.overlap(excitation, C)
                    excitation = self.excite(phi, 2*p, 2*q + 1, 2*p, 2*q + 1)
                    t1 += two[p,q,p,q]*self.overlap(excitation, C)
                    #excitation = self.excite(phi, 2*q, 2*p + 1, 2*q, 2*p + 1)
                    #t1 += 0.5*two[p,q,p,q]*self.overlap(excitation, C)

        result += (t0*self.overlap(phi, C)) + 0.5*t1

        return result


    def nonlin(self, x0, one, two, core, dets=None):
        E = x0[0]
        C = x0[1:].reshape(self.npairs, self.norbs)
        if dets is None:
            dets = self.pspace
        vec = []
        for phi in dets[:x0.size]:
            vec.append(E*self.overlap(phi, C) - self.phi_H_psi(phi, C, one, two, core))
        return vec


    def lstsq(self, x0, one, two, core, dets=None):
        E = x0[0]
        C = x0[1:].reshape(self.npairs, self.norbs)
        if dets is None:
            dets = self.pspace
        Hvec = np.zeros(len(dets))
        Svec = np.zeros(len(dets))
        for i in range(len(dets)):
            Hvec[i] = self.phi_H_psi(dets[i], C, one, two, core)
            Svec[i] = self.overlap(dets[i], C)
        numerator = Hvec - E*Svec
        numerator = numerator.dot(numerator)
        denominator = Svec.dot(Svec)
        return numerator/denominator


    @classmethod
    def excite(cls, phi, *args):
        assert (len(args) % 2) == 0, \
            "An equal number of annihilations and creations must occur."
        halfway = len(args)//2

        for i in args[:halfway]:
            if cls.occupied(phi, i):
                phi &= ~(1 << i)
            else:
                return 0

        for i in args[halfway:]:
            if cls.occupied(phi, i):
                return 0
            else:
                phi |= 1 << i

        return phi


    @staticmethod
    def occupied(phi, orbital):
        return bool(phi & (1 << orbital))


# vim: set textwidth=90 :
