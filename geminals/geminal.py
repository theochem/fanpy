from __future__ import absolute_import, division, print_function
import numpy as np
from math import factorial
from itertools import combinations, permutations
from newton import newton
from scipy.optimize import root as quasinewton
from scipy.optimize import minimize as lstsq
from slater_det import excite_pairs, is_pair_occupied, is_occupied

class APIG(object):
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
        pspace : {None, iterable of int}
            Projection space
            Iterable of integers that, in binary, describes which spin orbitals are
            used to build the Slater determinant
            The 1's in the even positions describe alpha orbitals
            The 1's in the odd positions describe beta orbitals
            By default, a projection space is generated using generate_pspace
        """
        assert isinstance(npairs, int)
        assert isinstance(norbs, int)
        
        # initialize "private" variables
        self._npairs = npairs
        self._norbs = norbs
        self._pspace = pspace
        self._coeffs = None

        # check if the assigned values follow the desired conditions (using the setter)
        self.npairs = npairs
        self.norbs = norbs
        if pspace is not None:
            assert hasattr(pspace, '__iter__')
            self.pspace = pspace
        else:
            self.pspace = self.generate_pspace()


    def __call__(self, x0, one, two, **kwargs):

        # Default options
        defaults = {
                     'jac': None,
                     'proj': None,
                     'solver': quasinewton,
                     'options': None,
                   }
        defaults.update(kwargs)
        jac = defaults['jac']
        proj = defaults['proj']
        solver = defaults['solver']
        options = defaults['options']
        
        # Check options
        if not options:
            options = {}
        if not proj:
            proj = self.pspace
        if jac:
            raise NotImplementedError

        # Construct reduced Hamiltonian terms
        ham = self.reduce_hamiltonian(one, two)

        # Solve for geminal coefficients
        if solver is lstsq:
            # Doing the least-squares method
            objective = self.lstsq
            options['tol'] = 1.0e-9
        else:
            # Doing the nonlinear-system method, so check for over/under-determination
            objective = self.nonlin
            if len(proj) > len(x0):
                print("Warning: length of 'proj' should be length of guess, 'P*K'.")
                print("Using the first {} determinants in 'proj'.".format(len(x0)))
            elif len(proj) < len(x0):
                raise ValueError("'proj' is too short, the system is underdetermined.")
      
        # Run the solver (includes intemediate normalization)
        print(solver)
        print(objective)
        result = solver(objective, x0, jac=jac, args=(ham, proj), **options)

        # Update the optimized coefficients
        self.coeffs = result['x']
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
            If number of electron pairs is a float
            If number of electron pairs is less than or equal to zero
            If fractional number of electron pairs is given

        """
        assert isinstance(value, int), 'There can only be integral number of electron pairs'
        assert value > 0, 'Number of electron pairs cannot be less than or equal to zero'
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
    def pspace(self, list_sds):
        """ Sets the projection space

        Parameters
        ----------
        list_sds : list of int
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
        for sd in list_sds:
            bin_string = bin(sd)[2:]
            # Add zeros on the left so that bin_string has even numbers
            bin_string = '0'*(len(bin_string)%2) + bin_string
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
            assert index_last_spatial < self.norbs,\
            ('Given Slater determinant contains orbitals whose indices exceed the '
             'given number of spatial orbitals')
        self._pspace = tuple(list_sds)


    @property
    def ground(self):
        return min(self.pspace)


    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        assert value.size == self.npairs*self.norbs,\
                ('Given geminals coefficient matrix does not have the right number of '
                 'coefficients')
        self._coeffs = value.reshape(self.npairs, self.norbs)


    def generate_pspace(self):
        """ Generates a well determined projection space

        Returns
        -------
        tuple_sd : tuple of int
            Tuple of integers that in binary describes the orbitals used to make the Slater
            determinant

        Raises
        ------
        AssertionError
            If the same Slater determinant is generated more than once
            If not enough Slater determinants can be generated

        Note
        ----
        We need to have npairs*norbs+1 dimensional projection space
        First is the ground state HF slater determinant (using the first npair
        spatial orbitals) [1]
        Then, all single pair excitations from any occupieds to any virtuals
        (ordered HOMO to lowest energy for occupieds and LUMO to highest energy
        for virtuals)
        Then, all double excitations of the appropriate number of HOMOs to
        appropriate number of LUMOs. The appropriate number is the smallest number
        that would generate the remaining slater determinants. This number will
        maximally be the number of occupied spatial orbitals.
        Then all triple excitations of the appropriate number of HOMOs to
        appropriate number of LUMOs

        It seems that certain combinations of npairs and norbs is not possible
        because we are assuming that we only use pair excitations. For example,
        a 2 electron system cannot ever have appropriate number of Slater determinants,
        a 4 electron system must have 6 spatial orbitals,
        a 6 electron system must have 6 spatial orbitals.
        a 8 electron system must have 7 spatial orbitals.

        If we treat E as a parameter
        """
        list_sd = []
        # convert string of a binary into an integer
        hf_ground = int('1'*2*self.npairs, 2)
        list_sd.append(hf_ground)
        ind_occ = [i for i in range(self.norbs) if is_pair_occupied(hf_ground, i)]
        ind_occ.reverse() # reverse ordering to put HOMOs first
        ind_vir = [i for i in range(self.norbs) if not is_pair_occupied(hf_ground, i)]
        num_needed = self.npairs*self.norbs+1
        # Add pair excitations
        num_excited = 1
        num_combs = lambda n, r: factorial(n)/factorial(r)/factorial(n-r)
        while len(list_sd) < num_needed and num_excited <= len(ind_occ):
            # Find out what is the smallest number of frontier orbitals (HOMOS and LUMOs)
            # that can be used
            for i in range(2, len(ind_occ)+1):
                if num_combs(i, 2)**2 >= num_needed-len(list_sd):
                    num_frontier = i
                    break
            else:
                num_frontier = max(len(ind_occ), len(ind_vir))+1
            # Add excitations from all possible combinations of num_frontier of HOMOs
            for occs in combinations(ind_occ[:num_frontier], num_excited):
                # to all possible combinations of num_frontier of LUMOs
                for virs in combinations(ind_vir[:num_frontier], num_excited):
                    occs_virs = list(occs) + list(virs)
                    list_sd.append(excite_pairs(hf_ground, *occs_virs))
            num_excited += 1
        assert len(list_sd) == len(set(list_sd)),\
            ('Woops, something went wrong. Same Slater determinant was generated '
             'more than once')
        assert len(list_sd) >= num_needed,\
            ('Could not generate enough Slater determinants')
        # Truncate and return
        return tuple(list_sd[:num_needed])


    @staticmethod
    def permanent(matrix):
        """ Calculates the permanent of a matrix

        Parameters
        ----------
        matrix : np.ndarray(N,N)
            Two dimensional square numpy array

        Returns
        -------
        permanent : float

        Raises
        ------
        AssertionError
            If matrix is not square
        """
        assert matrix.shape[0] is matrix.shape[1], \
            "Cannot compute the permanent of a non-square matrix."
        permanent = 0
        row_indices = range(matrix.shape[0])
        for col_indices in permutations(row_indices):
            permanent += np.product(matrix[row_indices, col_indices])
        return permanent


    def overlap(self, slater_det, gem_coeff):
        """ Calculate the overlap between a slater determinant and geminal wavefunction

        Parameters
        ----------
        slater_det : int
            Integers that, in binary, describes the orbitals used to make the Slater
            determinant
        gem_coeff : np.ndarray(P,K)
            Coefficient matrix for the geminal wavefunction

        Returns
        -------
        overlap : float
            Overlap between the slater determinant and the geminal wavefunction

        """
        # If bad Slater determinant
        if slater_det is None:
            return 0
        # If Slater determinant has different number of electrons
        elif bin(slater_det).count('1') != self.nelec:
            return 0
        # If Slater determinant has any of the (of the alpha beta pair) alpha
        # and beta orbitals do not have the same occupation
        elif any(is_occupied(slater_det, i*2) != is_occupied(slater_det, i*2+1)
                 for i in range(self.norbs)):
            return 0
        # Else
        else:
            ind_occ = [i for i in range(self.norbs) if is_pair_occupied(slater_det, i)]
            return self.permanent(gem_coeff[:, ind_occ])


    def phi_H_psi(self, phi, C, ham):

        t0 = 0.0
        t1 = 0.0
        t2 = 0.0

        for i in range(self.norbs):
            if is_pair_occupied(phi, i):
                t0 += ham[0][i]

                for j in range(i + 1, self.norbs):
                    if is_pair_occupied(phi, j):
                        t1 += ham[1][i,j]

                for a in range(self.norbs):
                    if not is_pair_occupied(phi, a):
                        excitation = excite_pairs(phi, i, a)
                        t2 += ham[2][i,a]*self.overlap(excitation, C)

        return (t0 + t1)*self.overlap(phi, C) + t2 


    def construct_guess(self, x0):
        C = x0.reshape(self.npairs, self.norbs)
        return C


    def nonlin(self, x0, ham, proj):
        C = self.construct_guess(x0)
        vec = [self.overlap(self.ground, C) - 1.0]
        energy = self.phi_H_psi(self.ground, C, ham)
        for phi in proj:
            tmp = energy*self.overlap(phi, C) 
            tmp -= self.phi_H_psi(phi, C, ham)
            vec.append(tmp)
            if len(vec) == x0.size:
                break
        return vec


    def lstsq(self, x0, ham, proj):
        C = self.construct_guess(x0)
        Hvec = np.zeros(len(proj))
        Svec = np.zeros(len(proj))
        for i in range(len(proj)):
            Hvec[i] = self.phi_H_psi(proj[i], C, ham)
            Svec[i] = self.overlap(proj[i], C)
        numerator = Hvec - self.phi_H_psi(self.ground, C, ham)*Svec
        numerator = numerator.dot(numerator)
        denominator = Svec.dot(Svec)
        return numerator/denominator


    def normalize(self, x0):
        C = self.construct_guess(x0)
        return np.abs(self.overlap(self.ground, C) - 1.0)


    def reduce_hamiltonian(self, one, two):

        dp = np.zeros(one.shape[0])
        for p in range(dp.shape[0]):
            dp[p] += 2*one[p,p] + two[p,p,p,p]

        dpq = np.zeros(one.shape)
        for p in range(dpq.shape[0]):
            for q in range(dpq.shape[1]):
                dpq[p,q] += 4*two[p,q,p,q] - 2*two[p,q,q,p]

        gpq = np.zeros(one.shape)
        for p in range(gpq.shape[0]):
            for q in range(gpq.shape[0]):
                gpq[p,q] = two[p,p,q,q]

        return dp, dpq, gpq


class AP1roG(APIG):

    def construct_guess(self, x0):
        C = np.eye(self.npairs, M=self.norbs)
        C[:,self.npairs:] += x0.reshape(self.npairs, self.norbs - self.npairs)
        return C
   
   
    @property
    def coeffs(self):
        return self._coeffs


    @coeffs.setter
    def coeffs(self, value):
        assert value.size == self.npairs*(self.norbs - self.npairs),\
                ('Given geminals coefficient matrix does not have the right number of '
                 'coefficients')
        coeffs = np.eye(self.npairs, M=self.norbs)
        coeffs[:,self.npairs:] += value.reshape(self.npairs, self.norbs - self.npairs)
        self._coeffs = coeffs


    def generate_pspace(self):
        """ Generates the projection space

        Returns
        -------
        pspace : list
            List of numbers that in binary describes the occupations
        """
        base = sum([ 2**(2*i) + 2**(2*i + 1) for i in range(self.npairs) ])
        pspace = [base]
        for unoccup in range(self.npairs, self.norbs):
            for i in range(self.npairs):
            # Python's sum is used here because NumPy's sum doesn't respect
            # arbitrary-precision Python longs

            # pspace = all single pair excitations
                pspace.append(excite_pairs(base, i, unoccup))
        # Uniquify
        return list(set(pspace))

    def overlap(self, phi, matrix):
        if phi == 0:
            return 0
        elif phi not in self.pspace:
            return 0
        else:
            from_index = []
            to_index = []
            excite_count = 0
            for i in range(self.npairs):
                if not is_pair_occupied(phi, i):
                    excite_count += 1
                    from_index.append(i)
            for i in range(self.npairs, self.norbs):
                if not is_pair_occupied(phi, i):
                    to_index.append(i)

            if excite_count == 0:
                return 1
            elif excite_count == 1:
                return matrix[from_index[0], to_index[0]]
            elif excite_count == 2:
                overlap = matrix[from_index[0], to_index[0]] * \
                          matrix[from_index[1], to_index[1]]
                overlap -= matrix[from_index[0], to_index[1]] * \
                           matrix[from_index[1], to_index[0]]
                return overlap
            else:
                raise Exception(str(excite_count))


    def nonlin(self, x0, ham, proj):
        C = self.construct_guess(x0)
        vec = []
        for phi in proj:
            tmp = self.phi_H_psi(self.ground, C, ham)*self.overlap(phi, C) 
            tmp -= self.phi_H_psi(phi, C, ham)
            vec.append(tmp**2)
            if len(vec) == x0.size:
                break
        return vec


# vim: set textwidth=90 :
