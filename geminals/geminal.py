import numpy as np
from itertools import combinations, permutations
from Newton import newton
from scipy.optimize import root as quasinewton
from scipy.optimize import minimize as lstsq
from slater_det import excite, is_occupied

class Geminal(object):
    """
    Attributes
    ----------
    npairs : int
        Number of electron pairs
    norbs : int
        Number of spatial orbitals
    coeffs :
    pspace :
        Projection space
        Tuple of integers that, in binary, describes which spin orbitals are
        used to build the Slater determinant
        The 1's in the even positions describe alpha orbitals
        The 1's in the odd positions describe beta orbitals


    Notes
    -----
    Slater determinants are expressed by an integer that, in binary form,
    shows which spin orbitals are used to construct it. The position of each '1'
    from the right is the index of the orbital that is included in the Slater
    determinant. The even positions (i.e. 0, 2, 4, ...) are the alpha orbitals,
    and the odd positions (i.e. 1, 3, 5, ...) are the beta orbitals.
    e.g. 5=0b00110011 is a Slater determinant constructed with the
    0th and 2nd spatial orbitals, or the 0th, 1st, 5th, 6th spin orbitals (where
    the spin orbitals are ordered by the alpha beta pairs)

    The number of electrons is assumed to be even
    """

    def __init__(self, npairs, norbs, pspace=None):
        """

        Parameters
        ----------
        npairs : int
            Number of electron pairs
        norbs : int
            Number of spatial orbitals
        pspace : iterable of int
            Projection space
            Iterable of integers that, in binary, describes which spin orbitals are
            used to build the Slater determinant
            The 1's in the even positions describe alpha orbitals
            The 1's in the odd positions describe beta orbitals
        """
        assert isinstance(npairs, int)
        assert isinstance(norbs, int)
        assert hasattr(pspace, '__iter__')
        # initialize "private" variables
        self._npairs = npairs
        self._norbs = norbs
        self._pspace = pspace
        # check if the assigned values follow the desired conditions (using the setter)
        self.npairs = npairs
        self.norbs = norbs
        self.coeffs = None
        if pspace is not None:
            self.pspace = pspace
        else:
            self.generate_pspace()

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
    def npairs(self):
        """ Number of electron pairs

        Returns
        -------
        npairs : int
            Number of electron pairs
        """
        return self._npairs

    @npairs.setter
    def npairs(self, value):
        """ Sets the number of Electron Pairs

        Parameters
        ----------
        value : int
            Number of electron pairs

        Raises
        ------
        AssertionError
            If fractional number of electron pairs is given

        """
        assert isinstance(value, int), 'There can only be integral number of electron pairs'
        assert value <= self._norbs,\
        'Number of number of electron pairs must be less than the number of spatial orbitals'
        self._npairs = value

    @property
    def nelec(self):
        """ Number of electrons

        Returns
        -------
        nelec : int
            Number of electrons
        """
        return 2*self.npairs

    @property
    def norbs(self):
        """ Number of spatial orbitals

        Returns
        -------
        norbs : int
            Number of spatial orbitals
        """
        return self._norbs

    @norbs.setter
    def norbs(self, value):
        """ Sets the number of spatial orbitals

        Parameters
        ----------
        value : int
            Number of spatial orbitals

        Raises
        ------
        AssertionError
            If fractional number of spatial orbitals is given
        """
        assert isinstance(value, int), 'There can only be integral number of spatial orbitals'
        assert value >= self.npairs,\
        'Number of spatial orbitals must be greater than the number of electron pairs'
        self._norbs = value

    @property
    def pspace(self):
        """ Projection space

        Returns
        -------
        List of integers that, in binary, describes which spin orbitals are
        used to build the Slater determinant
        The 1's in the even positions describe alpha orbitals
        The 1's in the odd positions describe beta orbitals

        """
        return self._pspace

    @pspace.setter
    def pspace(self, value):
        """ Sets the projection space

        Parameters
        ----------
        value : list of int
            List of integers that, in binary, describes which spin orbitals are
            used to build the Slater determinant
            The 1's in the even positions describe alpha orbitals
            The 1's in the odd positions describe beta orbitals

        Raises
        ------
        AssertionError
            If any of the Slater determinants contains more orbitals than the number of electrons
            If any of the Slater determinants is unrestricted (not all alpha beta pairs are
            both occupied)
            If any of the Slater determinants has spin orbitals whose indices exceed the given
            number of spatial orbitals

        """
        for sd in value:
            bin_string = bin(sd)[2:]
            # Check that number of orbitals used to create spin orbitals is equal to number
            # of electrons
            assert bin_string.count('1') == self.nelec,\
            ('Given Slater determinant does not contain the same number of orbitals '
             'as the given number of electrons')
            # Check that adjacent orbitals in the Slater determinant are alpha beta pairs
            # i.e. all spatial orbitals are doubly occupied
            alpha_occ = bin_string[0::2]
            beta_occ = bin_string[1::2]
            assert alpha_occ == beta_occ, "Given Slater determinant is unrestricted"
            # Check that there are no orbitals are used that are outside of the
            # specified number (norbs)
            index_last_spin = len(bin_string)-1-bin_string.index('1')
            index_last_spatial = (index_last_spin)//2
            assert index_last_spatial < self.norbs-1,\
            ('Given Slater determinant contains orbitals whose indices exceed the given number of'
             'spatial orbitals')
        self._pspace = tuple(value)

    def generate_pspace(self):
        """ Generates the projection space

        Returns
        -------
        pspace : list
            List of numbers that in binary describes the occupations
        """
        pspace = []
        for pairs in combinations(range(self.norbs), self.npairs):
            # Python's sum is used here because NumPy's sum doesn't respect
            # arbitrary-precision Python longs

            # i here describes 
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
            columns = [ i for i in range(self.norbs) if is_occupied(phi, 2*i) ]
            overlap = self.permanent(matrix[:,columns])
            return overlap


    def phi_H_psi(self, phi, C, one, two, core):

        t0 = 0.0
        for p in range(self.norbs):
            number = 0.0
            if is_occupied(phi, 2*p):
                number += 1
            if is_occupied(phi, 2*p + 1):
                number += 1
            if number:
                number*= one[p,p]
            t0 += number
        t0 *= self.overlap(phi, C)

        t1 = 0.0
        t2 = 0.0
        for p in range(self.norbs):
            if is_occupied(phi, 2*p) and is_occupied(phi, 2*p + 1):
                for q in range(self.norbs):
                    if is_occupied(phi, 2*q) and is_occupied(phi, 2*q + 1):
                        t1 += 2*two[p,q,p,q]*self.overlap(phi, C)
                        t2 -= two[p,p,q,q]*self.overlap(phi, C)

        #print("t0: {}\tt1: {}\tt2: {}".format(t0, t1, t2))
        result = t0 + t1 + t2 + core

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


# vim: set textwidth=90 :
