"""
APIG geminal wavefunction class.

"""

from __future__ import absolute_import, division, print_function
from itertools import combinations, permutations
import numpy as np
from scipy.misc import comb
from scipy.optimize import root
from geminals.slater_det import excite_orbs, excite_pairs, is_occupied, is_pair_occupied


class APIG(object):
    """
    A generalized antisymmetrized product of interacting geminals (APIG) implementation.

    Attributes
    ----------
    npairs : int
        The number of electron pairs in the geminal wavefunction.
    norbs : int
        The number of spatial orbitals in the geminal wavefunction.
    nelec : int
        The number of electrons in the geminal wavefunction (2 * npairs).
    ham : tuple of a 2-index np.ndarray, a 4-index np.ndarray
        The two- and four- electron integral Hamiltonian matrices.
    core_energy : float
        The core energy of the system.
    coeffs : 2-index np.ndarray
        The optimized geminal coefficient matrix.
    _coeffs_optimized : bool
        Whether the geminal coefficient matrix has already been optimized.
    _is_complex : bool
        Whether the geminal coefficient matrix is complex
        Set to True if coeffs is complex
        Forces coefficients to be complex if True
    pspace : iterable of ints.
        An iterable of integers that, in binary representation, describe the Slater
        determinants in the geminal wavefunction's projection space.
    energy : float
        The ground-state energy of the geminal wavefunction.
    energies : dict
        The keys correspond to a type of energy, i.e., "coulomb", or "exchange", and the
        values are floats corresponding to the geminal wavefunction's energy of that type.

    Other attributes (developer stuff)
    ----------------------------------
    _exclude_ground : bool
        Whether to disallow the nonlinear system to be solved during coefficient
        optimization from containing an equation corresponding to the ground state.
        This may be needed if the equation can be reduced to "energy == energy",
        making the Jacobian matrix singular.
    _normalize : bool
        Whether to include an equation for intermediate normalization into the nonlinear
        system to be solved during coefficient optimization.  This may not be necessary
        for some geminal types (e.g. AP1roG), where normalization is always satisfied at
            the optimal coefficients.
    _row_indices : iterable
        An iterable returning the range of rows in the geminal coefficient matrix that
        contain the optimizable parameters.  It *should* always be equivalent to
        range(number_of_rows).
    _col_indices : iterable
        An iterable returning the range of columns in the geminal coefficient matrix that
        contain the optimizable parameters.  This often varies between geminal types.

    Methods
    -------
    __init__
        Initialize the APIG instance.
    solve_coeffs
        Optimize the geminal coefficients.
    _generate_x0
        Generate a guess at the optimal geminal coefficients.
    _construct_coeffs
        Construct a coefficient matrix from a guess `x0`.
    generate_pspace
        Generate an appropriate projection space for the optimization of the geminal
        coefficients.
    permanent
        Compute the permanent of a matrix.
    permanent_derivative
        Compute the partial derivative of the permanent of a matrix wrt one of its
        coefficients.
    overlap
        Compute the overlap of a Slater determinant with the geminal wavefunction.
    compute_energy
        Compute the energy of the geminal wavefunction.
    _double_compute_energy
        Efficient backend for compute_energy() for doubly-occupied Slater determinants.
    _brute_compute_energy
        Most general backend for compute_energy().
    nonlin
        Construct a nonlinear system of equations for the optimization of the geminal
        coefficients.
    nonlin_jac
        Compute the Jacobian of nonlin().

    Notes
    -----
    Slater determinants are expressed by an integer that, in binary form, shows which spin
    orbitals are used to construct it. The position of each "1" from the right is the
    index of the orbital that is included in the Slater determinant. The even positions
    (i.e. 0, 2, 4, ...) are the alpha orbitals, and the odd positions (i.e. 1, 3, 5, ...)
    are the beta orbitals.  E.g., 5=0b00110011 is a Slater determinant constructed with
    the 0th and 2nd spatial orbitals, or the 0th, 1st, 5th, 6th spin orbitals (where the
    spin orbitals are ordered by the alpha beta pairs).  The number of electrons is
    assumed to be even.

    Behaviour for complex inputs
    ----------------------------
    If self._is_complex is True, then all self._coeffs assigned will be turned complex
    If self._coeffs is assigned to a complex array, then self_is_complex will be
    assigned True


    """

    #
    # Class-wide (behaviour-changing) attributes and properties
    #

    _exclude_ground = False
    _normalize = True

    @property
    def _row_indices(self):
        return range(0, self.npairs)

    @property
    def _col_indices(self):
        return range(0, self.norbs)

    #
    # Special methods
    #

    def __init__(self, npairs, norbs, ham=None, coeffs=None, pspace=None, is_complex=False):
        """ Initialize the APIG instance

        Parameters
        ----------
        npairs : int
            Number of electron pairs in the geminal wavefunction.
        norbs : int
            Number of spatial orbitals in the geminal wavefunction.
        ham : tuple of a 2-index np.ndarray, a 4-index np.ndarray
            The two- and four- electron integral Hamiltonian matrices in molecular
            orbital basis
        coeffs : np.ndarray(P,K)
            Initial guess for the geminal coefficient matrix.
        pspace : iterable of ints.
            An iterable of integers that, in binary representation, describe the Slater
            determinants in the geminal wavefunction's projection space.
        is_complex : bool
            Flag for complex coefficient matrix

        """

        # Initialize private variables
        self._npairs = None
        self._norbs = None
        self._ham = None
        self._pspace = None
        self._coeffs = None
        self._coeffs_optimized = False
        self._core_energy = 0
        self._overlap_derivative = False
        self._overlap_indices = None
        self._is_complex = is_complex

        # Assign attributes their values using their setters
        self.norbs = norbs
        self.npairs = npairs
        self.ham = ham
        self.coeffs = coeffs
        self.pspace = pspace if pspace else self.generate_pspace()

    #
    # Methods
    #

    def solve_coeffs(self, **kwargs):
        """
        Optimize the geminal coefficients.

        Parameters
        ----------
        kwargs : dict, optional
            The options whose defaults should be overridden.  See "Defaults" section.

        Defaults
        --------
        x0 : 1-index np.ndarray, optional
            An initial guess at the optimal geminal coefficient matrix.  Has shape equal
            to APIG.coeffs.ravel().  Defaults to the optimized coefficients of the
            corresponding AP1roG instance.
        proj : indexable of ints, optional
            The user can opt to specify the projection space used to construct the
            nonlinear system.  Defaults to the class' own projection space generator (see
            `cls.generate_pspace`).
        jac : bool, optional
            Whether to use the Jacobian to solve the nonlinear system.  This is, at a low
            level, the choice of whether to use MINPACK's hybrdj (True) or hybrd (False)
            subroutines for Powell's method.  Defaults to True.
        solver_opts : dict, optional
            Additional options to pass to the internal solver, which is SciPy's interface
            to MINPACK.  See scipy.optimize.root.  Defaults to some sane choices.
        verbose : bool, optional
            Whether to print information about the optimization after its termination.
            Defaults to False.

        Returns
        -------
        result : OptimizeResult
            This object behaves as a dict.  See scipy.optimize.OptimizeResult.

        Raises
        ------
        AssertionError
            If the nonlinear system is underdetermined.

        """

        # Specify and override default options
        defaults = {
            "x0": None,
            "proj": None,
            "jac": True,
            "solver_opts": {},
            "verbose": False,
        }
        defaults.update(kwargs)
        solver_defaults = {
            "xatol": 1.0e-12,
            "method": "hybr",
        }
        solver_defaults.update(defaults["solver_opts"])
        x0 = defaults["x0"] if defaults["x0"] is not None else self._generate_x0()
        proj = defaults["proj"] if defaults["proj"] is not None else self.pspace
        jac = self.nonlin_jac if defaults["jac"] else False

        # Handle projection space behaviour
        if self._exclude_ground:
            proj = list(proj)
            proj.remove(self.ground)
        assert ((len(proj) >= x0.size/2 and self._is_complex) or
                (len(proj) >= x0.size and not self._is_complex)), \
            ("The nonlinear system is underdetermined because the specified"
             " projection space is too small.")

        # Run the solver
        print("Optimizing geminal coefficients...")
        result = root(self.nonlin, x0, jac=jac, args=(proj,), options=solver_defaults)

        # Update instance with optimized coefficients if successful, or else print a
        # warning
        if result["success"]:
            self.coeffs = result["x"]
            self._coeffs_optimized = True
            print("Coefficient optimization was successful.")
        else:
            print("Warning: solution did not converge; coefficients were not updated.")

        # Display some information
        if defaults["verbose"]:
            print("Number of objective function evaluations: {}".format(result["nfev"]))
            if "njev" in result:
                print("Number of Jacobian evaluations: {}".format(result["njev"]))
            svd_u, svd_jac, svd_v = np.linalg.svd(result["fjac"])
            svd_value = max(svd_jac) / min(svd_jac)
            print("max(SVD)/min(SVD) [closer to 1 is better]: {}".format(svd_value))

        return result

    def _generate_x0(self):
        """
        Construct an initial guess at the optimal APIG geminal coefficients.

        Returns
        -------
        x0 : 1-index np.ndarray
            The guess at the coefficients.

        """
        params = self.npairs * (self.norbs - self.npairs)
        scale = 0.2 / params
        def scaled_random():
            random_nums = scale*(np.random.rand(self.npairs, self.norbs-self.npairs)-0.5)
            return random_nums
        if not self._is_complex:
            x0 = np.hstack((np.identity(self.npairs), scaled_random()))
            return x0.ravel()
        else:
            x0_real = np.hstack((np.identity(self.npairs), scaled_random())).ravel()
            x0_imag = np.hstack((np.identity(self.npairs), scaled_random())).ravel()
            return np.hstack((x0_real, x0_imag))

    def _construct_coeffs(self, x0):
        """


        Parameters
        ----------
        x0 : 1-index np.ndarray
            A guess at the optimal geminal coefficients.

        Returns
        -------
        coeffs : 2-index np.ndarray
            `x0` as a coefficient matrix.

        """
        coeffs = None
        if not self._is_complex:
            coeffs = x0.reshape(self.npairs, self.norbs)
        else:
            # Instead of dividing coeffs into a real and imaginary part and adding
            # them, we add the real part to the imaginary part
            # Imaginary part is assigned first because this forces the numpy array
            # to be complex
            coeffs = (x0[x0.size//2:]).reshape(self.npairs, self.norbs)*1j
            # Add the real part
            coeffs += (x0[:x0.size//2]).reshape(self.npairs, self.norbs)
        return coeffs

    def generate_pspace(self):
        """
        Generate an appropriate projection space.

        Notes
        -----
        We need to have an `npairs * norbs`-dimensional projection space.  First is the
        ground state HF slater determinant (using the first npair spatial orbitals) [1].
        Then, all single pair excitations from any occupieds to any virtuals (ordered HOMO
        to lowest energy for occupieds and LUMO to highest energy for virtuals).  Then,
        all double excitations of the appropriate number of HOMOs to appropriate number of
        LUMOs.  The appropriate number is the smallest number that would generate the
        remaining slater determinants.  This number will maximally be the number of
        occupied spatial orbitals.  Then all triple excitations of the appropriate number
        of HOMOs to appropriate number of LUMOs.

        It seems that certain combinations of npairs and norbs is not possible because we
        are assuming that we only use pair excitations. For example, a 2 electron system
        cannot ever have appropriate number of Slater determinants, a 4 electron system
        must have 6 spatial orbitals, a 6 electron system must have 6 spatial orbitals.  a
        8 electron system must have 7 spatial orbitals.

        Returns
        -------
        pspace : tuple of ints
            Iterable of integers that, in binary, describes which spin orbitals are used
            to build the Slater determinant.

        Raises
        ------
        AssertionError
            If the method is unable to generate an appropriate projection space.

        """

        # Find the minimum rank of a non-underdetermined system
        min_rank = self.npairs * self.norbs + int(self._exclude_ground)

        # Find the ground state and occupied/virtual indices
        ground = self.ground
        pspace = [ground]
        ind_occ = [i for i in range(self.norbs) if is_pair_occupied(ground, i)]
        ind_occ.reverse()  # reverse ordering to put HOMOs first
        ind_virt = [i for i in range(self.norbs) if not is_pair_occupied(ground, i)]

        # Add pair excitations
        ind_excited = 1
        while len(pspace) < min_rank and ind_excited <= len(ind_occ):
            # Determine the smallest usable set of frontier (HOMO/LUMO) orbitals
            for i in range(2, len(ind_occ) + 1):
                if comb(i, 2, exact=True) ** 2 >= min_rank - len(pspace):
                    nfrontier = i
                    break
            else:
                nfrontier = max(len(ind_occ), len(ind_virt)) + 1
            # Add excitations from all possible combinations of nfrontier HOMOs...
            for occs in combinations(ind_occ[:nfrontier], ind_excited):
                # ...to all possible combinations of nfrontier LUMOs
                for virts in combinations(ind_virt[:nfrontier], ind_excited):
                    orbs = list(occs) + list(virts)
                    pspace.append(excite_pairs(ground, *orbs))

            ind_excited += 1

        # Add single excitations (necessary for systems with less than two electron pairs
        # and probably for some minimal basis set cases) by exciting betas to alphas
        if self.npairs <= 2:
            for i in ind_occ:
                for j in ind_virt:
                    pspace.append(excite_orbs(ground, i * 2 + 1, j * 2))

        # Sanity checks
        assert len(pspace) == len(set(pspace)), \
            "generate_pspace() is making duplicate Slater determinants.  This shouldn't happen!"
        assert len(pspace) >= min_rank, \
            "generate_pspace() was unable to generate enough Slater determinants."

        # Return projection space truncated to `min_rank`
        return tuple(pspace[:min_rank])

    @staticmethod
    def permanent(matrix):
        """
        Combinatorically compute the permanent of a matrix.

        Parameters
        ----------
        matrix : 2-index np.ndarray
            A 2-dimensional, square NumPy array.

        Returns
        -------
        permanent : float

        Raises
        ------
        AssertionError
            If the matrix is not square.

        """

        assert matrix.shape[0] is matrix.shape[1], \
            "The permanent of a non-square matrix cannot be computed."
        permanent = 0
        row_indices = range(matrix.shape[0])
        for col_indices in permutations(row_indices):
            permanent += np.product(matrix[row_indices, col_indices])
        return permanent

    @staticmethod
    def permanent_derivative(matrix, i, j):
        """
        Calculates the partial derivative of a permanent with respect to one of its
        coefficients.

        Parameters
        ----------
        matrix : 2-index np.ndarray
            A 2-dimensional, square NumPy array.
        i : int
            `i` in the indices (i, j) of the coefficient with respect to which the partial
            derivative is computed.
        j : int
            See `i`.  This is `j`.

        Returns
        -------
        derivative : float

        Raises
        ------
        AssertionError
            If the matrix is not square.

        """

        assert matrix.shape[0] is matrix.shape[1], \
            "Cannot compute the permanent of a non-square matrix."

        # Permanent is invariant wrt row/column exchange; put coefficient (i,j) at (0,0)
        rows = list(range(matrix.shape[0]))
        cols = list(range(matrix.shape[1]))
        if i is not 0:
            rows[0], rows[i] = rows[i], rows[0]
        if j is not 0:
            cols[0], cols[j] = cols[j], cols[0]

        # Get values of the permutations that include the coefficient (i,j) by
        # multiplying along left- and right- hand diagonals, wrapping around where
        # necessary.  Don't actually include coeff (i,j) since it is differentiated out.
        left = matrix.shape[1] - 1
        right = 1
        prod_l = prod_r = 1
        for row in range(1, matrix.shape[0]):
            prod_l *= matrix[rows, :][:, cols][row, left]
            prod_r *= matrix[rows, :][:, cols][row, right]
            left -= 1
            right += 1
        return prod_l + prod_r

    def overlap(self, phi, coeffs=None):
        """
        Calculate the overlap between a Slater determinant and the geminal wavefunction.

        Parameters
        ----------
        phi : int
            An integer that, in binary representation, describes the Slater determinant.
        coeffs : 2-index np.ndarray, optional.
            The coefficient matrix for the geminal wavefunction.  Defaults to the
            optimized coefficients.

        Returns
        -------
        overlap : float

        Raises
        ------
        AssertionError
            If `coeffs` is not specified and the optimization has not already been
            completed.

        """

        if coeffs is None:
            assert self._coeffs_optimized, \
                "The geminal coefficient matrix has not yet been optimized."
            coeffs = self.coeffs

        # If the Slater determinant is bad
        if phi is None:
            return 0

        # If the Slater det. and geminal wavefcuntion have a different number of electrons
        elif bin(phi).count("1") != self.nelec:
            return 0

        # If the Slater determinant is non-singlet
        elif any(is_occupied(phi, i * 2) != is_occupied(phi, i * 2 + 1) for i in range(self.norbs)):
            return 0

        ind_occ = [i for i in range(self.norbs) if is_pair_occupied(phi, i)]

        if self._overlap_derivative:
            # If deriving wrt a coefficient appearing in the permanent
            if self._overlap_indices[1] in ind_occ:
                indices = (self._overlap_indices[0], ind_occ.index(self._overlap_indices[1]))
                return self.permanent_derivative(coeffs[:, ind_occ], *indices) / 2.0
            # If deriving wrt a coefficient not appearing in the permanent
            return 0

        # If not deriving
        return self.permanent(coeffs[:, ind_occ])

    def compute_energy(self, phi=None, coeffs=None):
        """
        Calculate the energy of the geminal wavefunction projected onto a Slater
        determinant.

        Parameters
        ----------
        phi : int, optional
            An integer that, in binary representation, describes the Slater determinant.
            Defaults to the ground state.
        coeffs : two-index np.ndarray, optional
            The geminal coefficient matrix.  Defaults to the optimized coefficients.

        Returns
        -------
        energy : float

        Raises
        ------
        AssertionError
            If `coeffs` is not specified and the optimization has not already been
            completed.

        Notes
        -----
        Assumes that molecular orbitals are restricted spatial orbitals.

        """

        if phi is None:
            phi = self.ground
        if coeffs is None:
            assert self._coeffs_optimized, \
                "The geminal coefficient matrix has not yet been optimized."
            coeffs = self._coeffs

        # Do the fast pairwise calculation if possible
        bin_phi = bin(phi)[2:]
        # Pad `bin_phi` with zeroes on the left so that it is of even length
        bin_phi = ("0" * (len(bin_phi) % 2)) + bin_phi
        alpha_occ = bin_phi[0::2]
        beta_occ = bin_phi[1::2]
        if alpha_occ == beta_occ:
            return self._double_compute_energy(phi, coeffs)
        else:
            return self._brute_compute_energy(phi, coeffs)

    def _double_compute_energy(self, phi, coeffs):
        """
        Backend to APIG.compute_energy().  See APIG.compute_energy.

        Notes
        -----
        Assumes the Slater determinant is doubly occupied.  One- and two- electron
        integrals are expressed wrt spatial orbitals.  Assumes that the one- and two-
        electron integrals are Hermitian.

        """

        one, two = self.ham
        olp = self.overlap(phi, coeffs)
        one_electron = 0.0
        coulomb0 = 0.0
        coulomb1 = 0.0
        exchange = 0.0

        for i in range(self.norbs):
            if is_pair_occupied(phi, i):
                one_electron += 2 * one[i, i]
                coulomb0 += two[i, i, i, i]

                for j in range(i + 1, self.norbs):
                    if is_pair_occupied(phi, j):
                        coulomb0 += 4 * two[i, j, i, j]
                        exchange -= 2 * two[i, j, j, i]

                for a in range(self.norbs):
                    if not is_pair_occupied(phi, a):
                        excitation = excite_pairs(phi, i, a)
                        coulomb1 += two[i, i, a, a] * self.overlap(excitation, coeffs)
        return one_electron * olp, coulomb0 * olp + coulomb1, exchange * olp

    def _brute_compute_energy(self, phi, coeffs):
        """
        Backend to APIG.compute_energy().  See APIG.compute_energy.

        Notes
        -----
        Makes no assumption about the Slater determinant structure.  One- and two-
        electron integrals are expressed wrt spatial orbitals.  Assumes that the one- and
        two- electron integrals are Hermitian.

        """

        one, two = self.ham

        # Get spin indices
        ind_occ = [i for i in range(2 * self.norbs) if is_occupied(phi, i)]
        ind_virt = [i for i in range(2 * self.norbs) if not is_occupied(phi, i)]
        one_electron = 0.0
        coulomb = 0.0
        exchange = 0.0
        ind_first_occ = 0

        for i in ind_occ:
            ind_first_virt = 0
            # Add `i` to `ind_virt` because excitation to same orbital is possible
            tmp_ind_virt_1 = sorted(ind_virt + [i])

            for k in tmp_ind_virt_1:
                single_excitation = excite_orbs(phi, i, k)
                if i % 2 == k % 2:
                    one_electron += one[i // 2, k // 2] * self.overlap(single_excitation, coeffs)
                # Avoid repetition by ensuring  `j > i` is satisfied
                for j in ind_occ[ind_first_occ + 1:]:
                    # Add indices `i` and `j` to `ind_virt` because excitation to same
                    # orbital is possible, and avoid repetition by ensuring `l > k` is
                    # satisfied
                    tmp_ind_virt_2 = sorted([j] + tmp_ind_virt_1[ind_first_virt + 1:])
                    for l in tmp_ind_virt_2:
                        double_excitation = excite_orbs(single_excitation, j, l)
                        overlap = self.overlap(double_excitation, coeffs)
                        if overlap == 0:
                            continue
                        # In <ij|kl>, `i` and `k` must have the same spin, as must
                        # `j` and `l`
                        if i % 2 == k % 2 and j % 2 == l % 2:
                            coulomb += two[i // 2, j // 2, k // 2, l // 2] * overlap
                        # In <ij|lk>, `i` and `l` must have the same spin, as must
                        # `j` and `k`
                        if i % 2 == l % 2 and j % 2 == k % 2:
                            exchange -= two[i // 2, j // 2, l // 2, k // 2] * overlap

                ind_first_virt += 1
            ind_first_occ += 1

        return one_electron, coulomb, exchange

    def nonlin(self, x0, pspace):
        """
        Construct the nonlinear system of equation which will be used to optimize the
        geminal coefficients.

        Parameters
        ----------
        x0 : 1-index np.ndarray
            A guess at the optimal geminal coefficients.

        pspace : iterable of ints
            Iterable of integers that, in binary, describes which spin orbitals are used
            to build the Slater determinant.

        Returns
        -------
        objective : 1-index np.ndarray
            The objective function of the nonlinear system.

        """
        # Handle differences in indexing between geminal methods
        eqn_offset = int(self._normalize)*(int(self._is_complex)+1)
        coeffs = self._construct_coeffs(x0)
        energy = sum(self.compute_energy(self.ground, coeffs))
        objective = np.zeros(x0.size)
        if not self._is_complex:
            # intermediate normalization
            if self._normalize:
                objective[0] = self.overlap(self.ground, coeffs) - 1.0
            for d in range(x0.size - eqn_offset):
                objective[d + eqn_offset] = (energy*self.overlap(pspace[d], coeffs)
                                             -sum(self.compute_energy(pspace[d], coeffs)))
        else:
            # Not too sure what to do here
            # Either make the real part = 1
            #if self._normalize:
            #    objective[0] = self.overlap(self.ground, coeffs) - 1.0
            # Or make the norm = 1
            #if self._normalize:
            #    objective[0] = np.absolute(self.overlap(self.ground, coeffs) - 1.0)
            # Or make real part = 1, imag part = 0
            if self._normalize:
                objective[0] = np.real(self.overlap(self.ground, coeffs)-1)
                objective[1] = np.imag(self.overlap(self.ground, coeffs))
            for d in range((x0.size - eqn_offset)//2):
                # Complex
                comp_eqn = (energy*self.overlap(pspace[d], coeffs)
                            -sum(self.compute_energy(pspace[d], coeffs)))
                # Real part
                objective[(d+0) + eqn_offset] = np.real(comp_eqn)
                # Imaginary part
                objective[(d+x0.size//2) + eqn_offset] = np.imag(comp_eqn)
        return objective

    def nonlin_jac(self, x0, pspace):
        """
        Construct the Jacobian of APIG.nonlin().  See APIG.nonlin().

        Returns
        -------
        objective : 2-index np.ndarray
            The Jacobian of the objective function of the nonlinear system.

        """

        # Handle differences in indexing between geminal methods
        eqn_offset = int(self._normalize) + int(self._is_complex)

        # Initialize Jacobian and some temporary values
        # (J)_dij = d(F_d)/d(c_ij)
        coeffs = self._construct_coeffs(x0)
        jac = np.zeros((x0.size, x0.size))

        # The objective functions {F_d} are of this form:
        # F_d = (d<phi0|H|psi>/dc_ij)*<phi'|psi>/dc_ij
        #           + <phi0|H|psi>*(d<phi'|psi>/dc_ij) - d<phi'|H|psi>/dc_ij

        # Compute the undifferentiated parts
        energy = sum(self.compute_energy(self.ground, coeffs))
        if not self._is_complex:
            energy_tmp = np.zeros(x0.size)
            for d in range(x0.size):
                energy_tmp[d] = self.overlap(pspace[d], coeffs)
        else:
            energy_tmp = np.zeros(x0.size//2)
            for d in range(x0.size//2):
                energy_tmp[d] = self.overlap(pspace[d], coeffs)

        # Overwrite APIG.overlap()-related attributes in order to take the correct partial
        # derivatives
        self._overlap_derivative = True
        count = 0
        for i in self._row_indices:
            for j in self._col_indices:
                self._overlap_indices = (i, j)
                if not self._is_complex:
                    if self._normalize:
                        jac[0, count] = self.overlap(self.ground, coeffs)
                    # Compute the differentiated parts and construct the whole Jacobian
                    for d in range(x0.size - eqn_offset):
                        jac[d + eqn_offset, count] = \
                            sum(self.compute_energy(self.ground, coeffs)) \
                            * energy_tmp[d] \
                            + energy * self.overlap(pspace[d], coeffs) \
                            - sum(self.compute_energy(pspace[d], coeffs))
                else:
                    if self._normalize:
                        derivative = self.overlap(self.ground, coeffs)
                        jac[0, count] = np.real(derivative)
                        jac[1, count] = np.imag(derivative)
                    # Compute the differentiated parts and construct the whole Jacobian
                    for d in range((x0.size-eqn_offset)//2):
                        derivative = (sum(self.compute_energy(self.ground, coeffs))
                                      *energy_tmp[d]
                                      +energy*self.overlap(pspace[d], coeffs)
                                      -sum(self.compute_energy(pspace[d], coeffs)))
                        jac[d+eqn_offset, count] = np.real(derivative)
                        jac[d+eqn_offset+x0.size//2, count+x0.size//2] = np.imag(derivative)
                count += 1

        # Replace the original APIG.overlap()-attributes and return the Jacobian
        self._overlap_indices = None
        self._overlap_derivative = False
        return jac

    #
    # Properties
    #

    @property
    def npairs(self):
        return self._npairs

    @npairs.setter
    def npairs(self, value):
        assert isinstance(value, int), \
            "The number of electron pairs must be an integer."
        assert value > 0, \
            "The number of electron pairs must be greater than zero."
        assert value <= self.norbs, \
            "The number of electron pairs must be less or equal to the number of spatial orbitals."
        self._npairs = value

    @property
    def nelec(self):
        return 2 * self.npairs

    @property
    def ground(self):
        return int(self.nelec * "1", 2)

    @property
    def norbs(self):
        return self._norbs

    @norbs.setter
    def norbs(self, value):
        assert isinstance(value, int), \
            "There can only be integral number of spatial orbitals."
        assert value >= self.npairs, \
            "Number of spatial orbitals must be greater than the number of electron pairs."

        self._norbs = value

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        if value is None:
            return
        elif len(value.shape) == 1 and not self._is_complex:
            assert value.size == self.npairs * self.norbs, \
                "The guess `x0` is not the correct size for the geminal coefficient matrix."
            self._coeffs = value.reshape(self.npairs, self.norbs)
        elif len(value.shape) == 1 and self._is_complex:
            assert value.size == 2*self.npairs*self.norbs, \
                "The guess `x0` is not the correct size for the geminal coefficient matrix."
            coeffs_real = value[:value.size/2].reshape(self.npairs, self.norbs)
            coeffs_imag = value[value.size/2:].reshape(self.npairs, self.norbs)
            self._coeffs = coeffs_real+coeffs_imag*1j
        else:
            assert value.shape == (self.npairs, self.norbs), \
                "The specified geminal coefficient matrix must have shape (npairs, norbs)."
            self._coeffs = value
        if self._coeffs.dtype == 'complex':
            self._is_complex = True
        elif self._is_complex:
            self._coeffs += 0j
        self._coeffs_optimized = True

    @property
    def ham(self):
        return self._ham

    @ham.setter
    def ham(self, value):
        assert (4 > len(value) > 1), \
            "The specified `ham` is too long (2 or 3 elements expected, {} given.".format(len(value))
        assert isinstance(value[0], np.ndarray) and isinstance(value[1], np.ndarray), \
            "One- and two- electron integrals (ham[0:2]) must be NumPy arrays."
        if len(value) == 3:
            assert isinstance(value[2], float), \
                "The core energy must be a float."
            self._core_energy = value[2]
        assert value[0].shape == tuple([self.norbs] * 2), \
            "The one-electron integral is not expressed wrt to the right number of spatial MOs."
        assert value[1].shape == tuple([self.norbs] * 4), \
            "The one-electron integral is not expressed wrt to the right number of spatial MOs."
        assert np.allclose(value[0], value[0].T), \
            "The one-electron integral matrix is not Hermitian."
        assert np.allclose(value[1], np.einsum("jilk", value[1])), \
            "The two-electron integral matrix does not satisfy <ij|kl> == <ji|lk>, or is not in physicists' notation."
        assert np.allclose(value[1], np.conjugate(np.einsum("klij", value[1]))), \
            "The two-electron integral matrix does not satisfy <ij|kl> == <kl|ij>^(dagger), or is not in physicists' notation."
        self._ham = value[:2]

    @property
    def core_energy(self):
        return self._core_energy

    @property
    def pspace(self):
        return self._pspace

    @pspace.setter
    def pspace(self, value):
        for phi in value:
            bin_phi = bin(phi)[2:]
            # Pad `bin_phi` with zeroes on the left so that it is of even length
            bin_phi = ("0" * (len(bin_phi) % 2)) + bin_phi
            assert bin_phi.count("1") == self.nelec, \
                "Slater det. {} does not contain the same number of occupied spin-orbitals as number of electrons.".format(bin_phi)
            # if self.npairs > 2:
            assert bin_phi[0::2] == bin_phi[1::2], \
                "Slater det. {} is unrestricted.".format(bin_phi)
            index_last_spin = len(bin_phi) - 1 - bin_phi.index("1")
            index_last_spatial = index_last_spin // 2
            assert index_last_spatial < self.norbs, \
                "Slater det. {} contains orbitals whose indices exceed the specified number of spatial orbitals.".format(bin_phi)
        self._pspace = tuple(value)

    @property
    def energy(self):
        return sum(self.compute_energy()) + self.core_energy

    @property
    def energies(self):
        one_electron, coulomb, exchange = self.compute_energy()
        energies = {
            "one_electron": one_electron,
            "coulomb": coulomb,
            "exchange": exchange,
        }
        if self.core_energy:
            energies["core"] = self.core_energy
        return energies

# vim: set textwidth=90 :
