#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import numpy as np
from itertools import combinations, permutations
from NonlinearSolver import *


class Geminal(object):
    """A template for the Geminal class that implements a simple APIG wavefunction.

    Attributes
    ----------
    npairs : int
        The number of electron pairs in the geminal.
    norbs : int
        The number of orbitals in the geminal.
    coeffs : 2-index np.ndarray, dtype=np.float64
        The geminal coefficient matrix.
    pspace : list
        The set of Slater determinants making up the projection space.

    Methods
    -------
    __call__ :
        Optimizes the geminal coefficient matrix by solving the Projected Schrodinger
        Equation.
    pschrodinger :
        the objective function for the Projected Schrodinger Equation.
    permanent :
        Computes the permanent of a matrix.
    overlap :
        Computes the overlap of a Slater determinant with the geminal wavefunction.
    excite_single :
        Singly excites a Slater determinant.
    excite_double :
        Doubly excites a Slater determinant.
    occupied :
        Tests whether an orbital in a Slater determinant is occupied.

    """

    def __init__(self, npairs, norbs, pspace=None):
        """Initializes the Geminal instance.

        Parameters
        ---------
        npairs : int
            The number of electron pairs in the geminal.
        norbs : int
            The number of orbitals in the geminal.
        pspace : iterable, optional
            The option is given to manually define the list of Slater determinants making
            up the projection space.

        Raises
        ------
        AssertionError

        """

        self.npairs = npairs
        self.norbs = norbs
        self.coeffs = None
        self.pspace = []
        self._overlap_cache = {}

        if pspace is None:
            for pairs in combinations(range(self.norbs), self.npairs):
                # Python's sum is used here because NumPy's sum doesn't respect
                # arbitrary-precision Python longs
                self.pspace.append(sum([ 2**(2*i) + 2**(2*i + 1) for i in pairs ]))
        else:
            for phi in pspace:
                for i in range(0, self.norbs, 2):
                    assert self.occupied(phi, i) == self.occupied(phi, i + 1), \
                        "Determinants in APIG projection space must be doubly occupied."
            self.pspace = pspace


    def __call__(self, dets=None, ham=None, x0=None, jac=None, solver=newton, options={}):
        """Optimizes the geminal coefficient matrix by solving the Projected Schrodinger
        Equation.

        Parameters
        ----------
        dets : iterable
            The set of Slater determinants to be projected against the geminal
            wavefunction in order to construct the nonlinear system.
        ham : tuple
            The (2-index, 4-index, constant) terms of the Hamiltonian matrix for the
            wavefunction.
        x0 : one-index np.ndarray, dtype=np.float64, optional
            Contains initial guesses for the optimizable parameters of the Equation.  x[0]
            is the energy, and x[1:] is equivalent to `coefficient_matrix.ravel()`.
        jac : bool, optional
            If True, the Jacobian will be used for solving the nonlinear system.  If False
            or unspecified, then it will not.  Note that in this case, an appropriate
            quasi-Newton method that does not use the Jacobian must be chosen.
        solver : function, optional
            The function that contains the nonlinear solver.
        options : dict, optional
            Additional keywords options to pass to the solver.

        Returns
        -------
        result : dict
            See scipy.optimize.OptimizeResult.

        Raises
        ------
        AssertionError
        NotImplementedError

        """

        for keyword in (dets, ham, x0):
            assert keyword is not None, \
                "dets, ham, and x0 keyword arguments must be specified."

        assert len(dets) >= x0.size, "The system is underdetermined."

        if jac:
            raise NotImplementedError

        if solver is lstsq:
            system = lambda x0, dets, ham : \
                np.sum([ self.pschrodinger(x0, phi, ham)**2 for phi in dets ])
        else:
            system = lambda x0, dets, ham : \
                [ self.pschrodinger(x0, phi, ham) for phi in dets[:x0.size] ]

        result = solver(system, x0, jac=jac, args=(dets, ham), **options)
        if not result['success']:
            print("Warning: the optimization did not succeed.")
        self.coeffs = result['x'][1:].reshape(self.npairs, self.norbs)
        return result


    @staticmethod
    def permanent(matrix):
        """Computes the permanent of a matrix.

        Parameters
        ----------
        matrix : two-index indexable
            The matrix whose permanent is to be computed.

        Returns
        -------
        permanent :
            The permanent of `matrix`.

        Raises
        ------
        AssertionError

        """

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
        """Computes the overlap of a Slater determinant with the geminal wavefunction.

        Parameters
        ----------
        phi : int
            The Slater determinant.
        matrix :
            The (rectangular) coefficient matrix from which we can slice out the overlap
            matrix of `phi` with the geminal wavefunction and return <Phi|Psi>.

        Returns
        -------
        overlap :
            The overlap of `phi` with the geminal wavefunction.

        """

        if phi not in self.pspace:
            overlap = 0
        elif phi in self._overlap_cache:
            overlap = self._overlap_cache[phi]
        else:
            # The ones in phi's binary representation determine the columns (orbitals) of
            # the coefficient matrix for which we want to evaluate the permanent; only
            # test the a-spin orbital in each pair because the geminal coefficients of
            # APIG correspond to electron pairs
            slice = [ i for i in range(self.norbs) if self.occupied(phi, 2*i) ]
            overlap = self.permanent(matrix[:,slice])
            self._overlap_cache[phi] = overlap
        return overlap


    @staticmethod
    def excite_single(phi, p, q):
        """Singly excites a Slater determinant.

        Parameters
        ----------
        phi : int
            The Slater determinant to be excited.
        p : int
            The index of the orbital from which the excitation occurs.
        q : int
            The index of the orbital to which the excitation occurs.

        Returns
        -------
        excitation : int
            The singly-excited Slater determinant.

        """

        pmask = ~(1 << p) 
        qmask = (1 << q)
        return (phi & pmask) | qmask


    @staticmethod
    def excite_double(phi, p, q, s, r):
        """Doubly excites a Slater determinant.

        Parameters
        ----------
        phi : int
            The Slater determinant to be excited.
        p : int
            The index of the orbital from which the excitation occurs.
        q : int
            The index of the orbital from which the excitation occurs.
        s : int
            The index of the orbital to which the excitation occurs.
        r : int
            The index of the orbital to which the excitation occurs.

        Returns
        -------
        excitation : int
            The doubly-excited Slater determinant.

        Raises
        ------
        AssertionError :
            Indices `p` and `q` cannot be equal, and indices `s` and `r` cannot be equal.

        """

        assert (p != q) and (s != r), "Indices p == q and/or s == r are forbidden."

        pmask = ~(1 << p)
        qmask = ~(1 << q)
        smask = (1 << s) 
        rmask = (1 << r)
        return ((((phi & pmask) & qmask) | smask) | rmask)


    @staticmethod
    def occupied(phi, orbital):
        """Tests whether an orbital in a Slater determinant is occupied.

        Parameters
        ----------
        phi : int
            The integer to be tested.
        orbital : int
            The column of `phi`'s binary representation to be tested.

        Returns
        -------
        value : bool
            True if the `orbital`th orbital is occupied, otherwise False.

        """

        return bool(phi & (1 << orbital))


    def pschrodinger(self, x0, phi, ham):
        """Computes the value of the objective function for the Projected Schrodinger
        Equation at given values of energy and geminal coefficients.

        Parameters
        ----------
        x0 : 1-index np.ndarray, dtype=np.float64
            Contains values for the optimizable parameters of the Equation.  x[0] is the
            energy, and x[1:] is equivalent to `coefficient_matrix.ravel()`.
        phi : int
            The Slater determinant.
        ham : tuple
            The (2-index, 4-index, constant) terms of the Hamiltonian matrix for the
            wavefunction.

        Returns
        -------
        result : float
            The value of the objective function for the provided parameters.

        """

        # objective(Phi, E, C) = E<Phi|Psi(C)> - <Phi|H|Psi(C)>
        result  = self._E_phi_psi(x0, phi)
        result -= self._phi_H2_psi(x0, phi, ham[0])
        result -= self._phi_H4_psi(x0, phi, ham[1])
        result -= self._phi_Hc_psi(x0, phi, ham[2])
        self._overlap_cache.clear()
        return result


    def _E_phi_psi(self, x0, phi):
        """Computes E<Phi|Psi> given phi and a guess for E and coefficients.

        Parameters
        ----------
        See Geminal.pschrodinger.

        Returns
        -------
        result : float
            The value of E<Phi|Psi> for the provided parameters.

        """

        return x0[0]*self.overlap(phi, x0[1:].reshape(self.npairs, self.norbs))


    def _phi_H2_psi(self, x0, phi, ham):
        """Computes <Phi|H(2-index)|Psi> given phi and a guess for E and coefficients.

        Parameters
        ----------
        See Geminal.pschrodinger.

        Returns
        -------
        result : float
            The value of <Phi|H(2-index)|Psi> for the provided parameters.

        """

        result = 0.0
        for q in range(self.norbs):
            if not self.occupied(phi, q):
                continue
            for p in range(self.norbs):
                if self.occupied(phi, p) and (p != q):
                    continue
                # <Phi|H|Psi> = sum_{p,q}{T_{q}^{p}<Phi|Psi>}
                tmp = self.excite_single(phi, q, p)
                tmp = self.overlap(tmp, x0[1:].reshape(self.npairs, self.norbs))
                if tmp:
                    result += ham[p][q]*tmp
        return result


    def _phi_H4_psi(self, x0, phi, ham):
        """Computes <Phi|H(4-index)|Psi> given phi and a guess for E and coefficients.

        Parameters
        ----------
        See Geminal.pschrodinger.

        Returns
        -------
        result : float
            The value of <Phi|H(4-index)|Psi> for the provided parameters.

        """

        result = 0.0
        for r in range(self.norbs):
            if not self.occupied(phi, r):
                continue
            for s in range(self.norbs):
                if (s == r) or (not self.occupied(phi, s)):
                    continue
                for q in range(self.norbs):
                    if (not self.occupied(phi, q)) and (q != r) and (q != s):
                        continue
                    for p in range(self.norbs):
                        if (p == q) or ((not self.occupied(phi, p)) and (p != r) and (p != s)):
                            continue
                        # <Phi|H|Psi> = sum_{p,q,r,s}{T_{r,s}^{q,p}<Phi|Psi>}
                        tmp = self.excite_double(phi, r, s, q, p)
                        tmp = self.overlap(tmp, x0[1:].reshape(self.npairs, self.norbs))
                        if tmp:
                            result += ham[p][q][r][s]*tmp
        return result


    def _phi_Hc_psi(self, x0, phi, ham):
        """Computes <Phi|H(constant)|Psi> given phi and a guess for E and coefficients.

        Parameters
        ----------
        See Geminal.pschrodinger.

        Returns
        -------
        result : float
            The value of <Phi|H(constant)|Psi> for the provided parameters.

        """

        return ham*self.overlap(phi, x0[1:].reshape(self.npairs, self.norbs))


# vim: set textwidth=90 :
