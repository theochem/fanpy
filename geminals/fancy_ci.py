"""
Fancy CI solved using Projected schrodinger method

"""

from __future__ import absolute_import, division, print_function
from itertools import combinations
import numpy as np
from scipy.optimize import root
from geminals.slater_det import excite_orbs, is_occupied


class FancyCI(object):
    """
    Parent class for the projected schrodinger methods

    Attributes
    ----------
    nelec : int
        Number of electrons in the wavefunction.
    norbs : int
        Number of spin orbitals in the wavefunction.
    ham : tuple of a 2-index np.ndarray, a 4-index np.ndarray
        The two- and four- electron integral Hamiltonian matrices in spatial or
        spin molecular orbital basis.
    params : np.ndarray(M,)
        Parameters that control the behaviour of the function
    pspace : iterable of ints.
        An iterable of integers that, in binary representation, describe the Slater
        determinants in the wavefunction's projection space.

    CHECK
    -----
    core_energy : float
        Core energy of the system.
    energies : dict
        The keys correspond to a type of energy, i.e., "coulomb", or "exchange", and the
        values are floats corresponding to the wavefunction's energy of that type.

    Other attributes (developer stuff)
    ----------------------------------
    _exclude_ground : bool
        Whether to disallow the nonlinear system to be solved during coefficient
        optimization from containing an equation corresponding to the ground state.
        This may be needed if the equation can be reduced to "energy == energy",
        making the Jacobian matrix singular.
    _normalize : bool
        Whether to include an equation for intermediate normalization into the nonlinear
        system to be solved during coefficient optimization.

    Methods
    -------
    __init__
        Initialize the ProjSchrMethod instance.
    solve_params
        Optimize the coefficients.
    _generate_init
        Generate a guess at the optimal coefficients.
    generate_pspace
        Generate an appropriate projection space for the optimization of the
        coefficients.
    overlap
        Compute the overlap of a Slater determinant with the wavefunction.
    compute_energy
        Compute the energy of the wavefunction.
    _double_compute_energy
        Efficient backend for compute_energy() for doubly-occupied Slater determinants.
    _brute_compute_energy
        Most general backend for compute_energy().
    nonlin
        Construct a nonlinear system of equations for the optimization of the
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
    spin orbitals are ordered by the alpha beta pairs).
    """

    #
    # Class-wide (behaviour-changing) attributes and properties
    #

    _exclude_ground = False
    _normalize = True

    #
    # Special methods
    #

    def __init__(self, nelec, norbs, ham, init_params=None, pspace=None,
                 is_complex=False, is_spatial=True):
        """
        Initialize the FancyCI instance.

        Parameters
        ----------
        nelec : int
            Number of electrons in system
        norbs : int
            Number of spin orbitals in system
        ham : tuple of a 2-index np.ndarray, a 4-index np.ndarray
            Two- and four- electron integral Hamiltonian matrices in
            molecular orbital basis.
        init_params : np.ndarray(M,)
            Initial guess for the parameters of the function
        pspace : iterable of ints.
            An iterable of integers that, in binary representation, describe the Slater
            determinants in the wavefunction's projection space.
        is_complex : bool
            Flag for complex coefficient matrix
        is_spatial : bool
            Flag for spatial molecular orbital basis

        """
        # Flags
        self._is_complex = is_complex
        self._is_spatial = is_spatial
        # Initialize private variables
        self._nelec = nelec
        self._norbs = norbs
        self._ham = None
        self._params = None
        self._pspace = None
        # Assign attributes their values using their setters
        self.nelec = nelec
        self.norbs = norbs
        self.ham = ham
        self.params = init_params if init_params is not None else self._generate_init()
        self.pspace = pspace if pspace is not None else self.generate_pspace()

    #
    # Properties
    #
    @property
    def offset_spatial(self):
        """ Corrects number of orbitals when they are spatial

        """
        if self._is_spatial:
            return 2
        else:
            return 1

    @property
    def offset_complex(self):
        """ Corrects number of orbitals when they are complex

        """
        if self._is_complex:
            return 2
        else:
            return 1

    @property
    def num_params(self):
        """ Number of parameters needed

        Note
        ----
        Arbitrary here. Number of orbitals was used because some number was needed to run the tests
        """
        # FIXME: Need to get rid of arbitrary num_params here. Should be replaced with a NotImplementedError
        return self.norbs*self.offset_complex

    @property
    def nelec(self):
        """ Number of electrons
        """
        return self._nelec

    @nelec.setter
    def nelec(self, value):
        """ Setter for number of electrons

        Raises
        ------
        AssertionError
            If value is not a positive integer
            If value is greater than the number of orbitals
        """
        assert isinstance(value, int), \
            "The number of electrons must be an integer."
        assert value > 0, \
            "The number of electrons must be greater than zero."
        assert value <= self.norbs*self.offset_spatial, \
            "The number of electrons must be less or equal to the number of spin orbitals."
        self._nelec = value

    @property
    def ground_sd(self):
        """ Ground state slater determinant

        Slater determinant with the first (nelec) number of orbitals occupied
        """
        return int(self.nelec * "1", 2)

    @property
    def norbs(self):
        """ Number of spin orbitals

        """
        return self._norbs

    @norbs.setter
    def norbs(self, value):
        """ Sets the number of spin orbitals

        Raises
        ------
        AssertionError
            If value is not a positive integer
            If value is less than the number of electrons
        """
        assert isinstance(value, int) and value > 0, \
            "Number of spatial orbitals must be a positive integer."
        assert value*self.offset_spatial >= self.nelec, \
            "Number of spatial orbitals must be greater than the number of electrons."
        self._norbs = value

    @property
    def params(self):
        """ Parameters for the "function" of the wavefunction
        """
        return self._params

    @params.setter
    def params(self, value):
        """ Sets the parameters for the "function" of the wavefunction

        Raises
        ------
        AssertionError
            If value is not a one dimensional numpy array
            If value does not have the same size as the parameters

        Note
        ----
        If the parameters are complex, then the given value must separate the real
        and the imaginary parts, and connect the real part with the imaginary part
        into a one dimensional array
        """
        assert isinstance(value, np.ndarray) and len(value.shape) == 1,\
            'Parameters must be given as a one dimensional numpy array'
        assert value.size == self.num_params,\
            'Number of given parameters is different from the number of parameters needed'
        self._params = value

    @property
    def ham(self):
        """ Tuple of one and two electron integrals

        """
        #TODO: Separate into one and two electron parts
        return self._ham[:2]

    @property
    def core_energy(self):
        """ Core hamiltonian energy
        """
        if len(self._ham) == 3:
            return self._ham[2]
        else:
            return None

    @ham.setter
    def ham(self, value):
        """ Tuple of one and two electron integrals

        """
        #TODO: Separate into one and two electron parts
        assert (3 >= len(value) >= 2), \
            "The specified `ham` is too long (2 or 3 elements expected, {} given.".format(len(value))
        assert isinstance(value[0], np.ndarray) and isinstance(value[1], np.ndarray), \
            "One- and two- electron integrals (ham[0:2]) must be NumPy arrays."
        if len(value) == 3:
            assert isinstance(value[2], float), "The core energy must be a float."
        assert value[0].shape == tuple([self.norbs] * 2), \
            "The one-electron integral is not expressed wrt to the right number of MOs."
        assert value[1].shape == tuple([self.norbs] * 4), \
            "The one-electron integral is not expressed wrt to the right number of MOs."
        assert np.allclose(value[0], value[0].T), \
            "The one-electron integral matrix is not Hermitian."
        assert np.allclose(value[1], np.einsum("jilk", value[1])), \
            "The two-electron integral matrix does not satisfy <ij|kl> == <ji|lk>, or is not in physicists' notation."
        assert np.allclose(value[1], np.conjugate(np.einsum("klij", value[1]))), \
            "The two-electron integral matrix does not satisfy <ij|kl> == <kl|ij>^(dagger), or is not in physicists' notation."
        self._ham = value[:3]

    @property
    def pspace(self):
        """ Projection space
        """
        return self._pspace

    @pspace.setter
    def pspace(self, value):
        """ Setter for the projection space

        Raises
        ------
        AssertionError
            If given Slater determinant does not contain the same number of electrons
            as the number of electrons
            If given Slater determinant is expressed wrt an orbital that doesn't exist
        """
        assert hasattr(value, '__iter__')
        for phi in value:
            bin_phi = bin(phi)[2:]
            # Pad `bin_phi` with zeroes on the left so that it is of even length
            bin_phi = ("0" * (len(bin_phi) % 2)) + bin_phi
            assert bin_phi.count("1") == self.nelec, \
                ("Slater det. {0} does not contain the same number of occupied"
                 " spin-orbitals as number of electrons.".format(bin_phi))
            index_last_spin = len(bin_phi) - 1 - bin_phi.index("1")
            assert index_last_spin < self.norbs*self.offset_spatial, \
                ("Slater det. {0} contains orbitals whose indices exceed the"
                 " specified number of orbitals.".format(bin_phi))
        self._pspace = tuple(value)

    @property
    def energy(self):
        """ Total energy of the system
        """
        return sum(self.compute_energy(self.ground_sd, self.params)) + self.core_energy

    @property
    def energies(self):
        """ Energies of the system, separated into a dictionary
        """
        one_electron, coulomb, exchange = self.compute_energy(self.ground_sd, self.params)
        energies = {
            "one_electron": one_electron,
            "coulomb": coulomb,
            "exchange": exchange,
        }
        if self.core_energy:
            energies["core"] = self.core_energy
        return energies


    #
    # Methods
    #

    def generate_pspace(self):
        """
        Generate the CISD projection space.

        Returns
        -------
        pspace : tuple of ints
            Iterable of integers that, in binary, describes which spin orbitals are used
            to build the Slater determinant.

        Raises
        ------
        AssertionError
            If duplicate Slater determinants were generated
        """

        # Find the ground state and occupied/virtual indices
        ground = self.ground_sd
        pspace = [ground]
        ind_occ = [i for i in range(self.norbs*self.offset_spatial) if is_occupied(ground, i)]
        ind_occ.reverse()  # reverse ordering to put HOMOs first
        ind_vir = [i for i in range(self.norbs*self.offset_spatial) if not is_occupied(ground, i)]
        # Add Single excitations
        for i in ind_occ:
            for j in ind_vir:
                pspace.append(excite_orbs(ground, i, j))
        # Add Double excitations
        for i,j in combinations(ind_occ, 2):
            for k,l in combinations(ind_vir, 2):
                pspace.append(excite_orbs(ground, i, j, k, l))
        # Sanity checks
        assert len(pspace) == len(set(pspace)), \
            "generate_pspace() is making duplicate Slater determinants.  This shouldn't happen!"
        return tuple(pspace)

    def _generate_init(self):
        """
        Construct an initial guess

        Returns
        -------
        init : 1-index np.ndarray
            Guess at the parameters

        """
        scale = 0.2 / self.num_params
        def scaled_random():
            random_nums = scale*(np.random.rand(self.num_params)-0.5)
            random_nums[0] = 1
            return random_nums
        if not self._is_complex:
            return scaled_random()
        else:
            x0_real = scaled_random()
            x0_imag = scaled_random()
            return np.hstack((x0_real, x0_imag))

    def the_function(self, params, elec_config):
        """ The function that assigns a coefficient to a given configuration given some
        parameters

        Parameters
        ----------
        params : np.ndarray(M,)
            Parameters that
        elec_config : int
            Slater determinant that corresponds to certain electron configuration

        Returns
        -------
        coefficient : float
            Coefficient of thet specified Slater determinant

        Raises
        ------
        NotImplementedError
        """
        if self._is_complex:
            # Instead of dividing params into a real and imaginary part and adding
            # them, we add the real part to the imaginary part
            # Imaginary part is assigned first because this forces the numpy array
            # to be complex
            temp_params = 1j*params[params.size//2:]
            # Add the real part
            temp_params += params[:params.size//2]
            params = temp_params
        raise NotImplementedError

    def differentiate_the_function(self, params, elec_config, index):
        """ Differentiates the function wrt to a specific parameter

        Parameters
        ----------
        params : np.ndarray(M,)
            Set of numbers
        elec_config : int
            Slater determinant that corresponds to certain electron configuration
        index : int
            Index of the parameter with which we are differentiating

        Returns
        -------
        coefficient : float
            Coefficient of thet specified Slater determinant

        Raises
        ------
        NotImplementedError
        """
        assert 0 <= index < params.size
        raise NotImplementedError

    def overlap(self, phi, params, differentiate_wrt=None):
        """
        Calculate the overlap between a Slater determinant and the geminal wavefunction.

        Parameters
        ----------
        phi : int
            An integer that, in binary representation, describes the Slater determinant.
        params : np.ndarray(K,), optional.
            Coefficient matrix for the wavefunction.
        differentiate_wrt : {None, int}
            Index wrt the overlap will be differentiated
            If unspecified, overlap is not differentiated

        Returns
        -------
        overlap : float

        """
        # If the Slater determinant is bad
        if phi is None:
            return 0
        # If the Slater det. and wavefuntion have a different number of electrons
        elif bin(phi).count("1") != self.nelec:
            return 0
        elif differentiate_wrt is None:
            return self.the_function(params, phi)
        else:
            assert isinstance(differentiate_wrt, int)
            index = differentiate_wrt
            return self.differentiate_the_function(params, phi, index)

    def compute_energy(self, phi, params, differentiate_wrt=None):
        """
        Calculate the energy of the wavefunction projected onto a Slater
        determinant.

        Parameters
        ----------
        phi : int
            An integer that, in binary representation, describes the Slater determinant.
            Defaults to the ground state.
        params : np.ndarray(M,)
            Set of parameters that describe the function
        differentiate_wrt : {None, int}
            Index wrt the overlap will be differentiated
            If unspecified, overlap is not differentiated

        Returns
        -------
        energy : tuple of float
            Tuple of the one electron, coulomb, and exchange energies

        Raises
        ------
        AssertionError
            If `params` is not specified and self.params is None

        Notes
        -----
        Makes no assumption about the Slater determinant structure.
        Assumes that one- and two-electron integrals are Hermitian.

        """
        one, two = self.ham
        one_electron = 0.0
        coulomb = 0.0
        exchange = 0.0
        ind_first_occ = 0
        diff_index = differentiate_wrt
        # Set divisor
        # One if hamiltonian is epxressed wrt spin orbitals
        # Two if hamiltonian is expressed wrt spatial orbitals
        div = self.offset_spatial
        # Get spin indices
        ind_occ = [i for i in range(self.norbs*div) if is_occupied(phi, i)]
        ind_vir = [i for i in range(self.norbs*div) if not is_occupied(phi, i)]
        for i in ind_occ:
            ind_first_vir = 0
            # Add `i` to `ind_vir` because excitation to same orbital is possible
            tmp_ind_vir_1 = sorted(ind_vir + [i])
            for k in tmp_ind_vir_1:
                single_excitation = excite_orbs(phi, i, k)
                if i%2 == k%2:
                    one_electron += (one[i//div, k//div]*
                                     self.overlap(single_excitation,
                                                  params,
                                                  differentiate_wrt=diff_index))
                # Avoid repetition by ensuring  `j > i` is satisfied
                for j in ind_occ[ind_first_occ + 1:]:
                    # Add indices `i` and `j` to `ind_vir` because excitation to same
                    # orbital is possible, and avoid repetition by ensuring `l > k` is
                    # satisfied
                    tmp_ind_vir_2 = sorted([j] + tmp_ind_vir_1[ind_first_vir + 1:])
                    for l in tmp_ind_vir_2:
                        double_excitation = excite_orbs(single_excitation, j, l)
                        overlap = self.overlap(double_excitation,
                                               params,
                                               differentiate_wrt=diff_index)
                        if overlap == 0:
                            continue
                        # In <ij|kl>, `i` and `k` must have the same spin, as must
                        # `j` and `l`
                        if i%2 == k%2 and j%2 == l%2:
                            coulomb += two[i//div, j//div, k//div, l//div]*overlap
                        # In <ij|lk>, `i` and `l` must have the same spin, as must
                        # `j` and `k`
                        if i%2 == l%2 and j%2 == k%2:
                            exchange -= two[i//div, j//div, l//div, k//div]*overlap
                ind_first_vir += 1
            ind_first_occ += 1
        return one_electron, coulomb, exchange

    def nonlin(self, guess, pspace):
        """
        Construct the nonlinear system of equation which will be used to optimize the
        coefficients.

        Parameters
        ----------
        guess : np.ndarray(M,)
            Guess at the optimal parameters

        pspace : iterable of ints
            Iterable of integers that, in binary, describes which spin orbitals are used
            to build the Slater determinant.

        Returns
        -------
        objective : 1-index np.ndarray
            The objective function of the nonlinear system.

        """
        energy = sum(self.compute_energy(self.ground_sd, guess))
        objective = np.zeros(guess.size)
        # Toggles intermediate normalization
        eqn_offset = 0
        if self._normalize:
            eqn_offset += 1
        if self._is_complex:
            eqn_offset += 1

        if not self._is_complex:
            # intermediate normalization
            if self._normalize:
                objective[0] = self.overlap(self.ground_sd, guess) - 1.0
            for d in range(guess.size - eqn_offset):
                objective[d + eqn_offset] = (energy*self.overlap(pspace[d], guess)
                                             -sum(self.compute_energy(pspace[d], guess)))
        else:
            # Not too sure what to do here
            # Either make the real part = 1
            #if self._normalize:
            #    objective[0] = self.overlap(self.ground, guess) - 1.0
            # Or make the norm = 1
            #if self._normalize:
            #    objective[0] = np.absolute(self.overlap(self.ground, guess) - 1.0)
            # Or make real part = 1, imag part = 0
            if self._normalize:
                olp = self.overlap(self.ground_sd, guess)
                objective[0] = np.real(olp-1)
                objective[1] = np.imag(olp)
            for d in range((guess.size - eqn_offset)//2):
                # Complex
                comp_eqn = (energy*self.overlap(pspace[d], guess)
                            -sum(self.compute_energy(pspace[d], guess)))
                # Real part
                objective[(d+0) + eqn_offset] = np.real(comp_eqn)
                # Imaginary part
                objective[(d+guess.size//2) + eqn_offset] = np.imag(comp_eqn)
        return objective

    def nonlin_jac(self, params, pspace):
        """
        Construct the Jacobian.

        Returns
        -------
        objective : 2-index np.ndarray
            The Jacobian of the objective function of the nonlinear system.

        """

        # Handle differences in indexing between geminal methods
        eqn_offset = 0
        if self._normalize:
            eqn_offset += 1
        if self._is_complex:
            eqn_offset += 1

        # (J)_dij = d(F_d)/d(c_ij)
        jac = np.zeros((params.size, params.size))
        ground_energy = sum(self.compute_energy(self.ground_sd, params))

        for j in range(params.size):
            if not self._is_complex:
                if self._normalize:
                    jac[0, j] = self.overlap(self.ground_sd, params, j)
                for i in range(params.size-eqn_offset):
                    jac[i+eqn_offset, j] = (sum(self.compute_energy(pspace[i], params, j)) +
                                            ground_energy*self.overlap(pspace[i], params, j))
            else:
                if self._normalize:
                    derivative = self.overlap(self.ground_sd, params, j)
                    jac[0, j] = np.real(derivative)
                    jac[1, j] = np.imag(derivative)
                for i in range((params.size-eqn_offset)//2):
                    derivative = (sum(self.compute_energy(pspace[i], params, j)) +
                                  ground_energy*self.overlap(pspace[i], params, j))
                    jac[i+eqn_offset, j] = np.real(derivative)
                    jac[i+eqn_offset+params.size//2, j+params.size//2] = np.imag(derivative)
        return jac

    def solve_params(self, **kwargs):
        """
        Optimize the parameters

        Parameters
        ----------
        x0 : 1-index np.ndarray, optional
            An initial guess at the optimal parameters.
            Defaults to `cls._generate_init()`
        pspace : iterable of ints, optional
            Projection space used to construct the nonlinear equations
            Defaults to `cls.generate_pspace`
        jac : bool, optional
            Whether to use the Jacobian to solve the nonlinear system.  This is, at a low
            level, the choice of whether to use MINPACK's hybrdj (True) or hybrd (False)
            subroutines for Powell's method.
            Defaults to True.
        solver_opts : dict, optional
            Additional options to pass to the internal solver, which is SciPy's interface
            to MINPACK.  See scipy.optimize.root.
            Defaults to xatol=1.0e-12, method="hybr"
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
            "init_guess": None,
            "pspace": None,
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
        x0 = defaults["x0"] if defaults["x0"] is not None else self._generate_init()
        pspace = defaults["pspace"] if defaults["pspace"] is not None else self.pspace
        jac = self.nonlin_jac if defaults["jac"] else False

        # Handle projeection space behaviour
        if self._exclude_ground:
            pspace = list(pspace)
            pspace.remove(self.ground_sd)
        assert ((len(pspace) >= x0.size/2 and self._is_complex) or
                (len(pspace) >= x0.size and not self._is_complex)), \
            ("The nonlinear system is underdetermined because the specified"
             " projection space is too small.")

        # Run the solver
        print("Optimizing {0} parameters...".format(self.__class__.__name__))
        result = root(self.nonlin, x0, jac=jac, args=(pspace,), options=solver_defaults)

        # Update instance with optimized coefficients if successful, or else print a
        # warning
        if result["success"]:
            self.params = result["x"]
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

# vim: set textwidth=90 :
