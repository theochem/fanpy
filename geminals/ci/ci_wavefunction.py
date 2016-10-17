from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np

from ..wavefunction import Wavefunction
from .. import slater
from .ci_matrix import is_alpha, spatial_index


class CIWavefunction(Wavefunction):
    """ Wavefunction expressed as a linear combination of Slater determinants

    Contains the necessary information to variationally solve the CI wavefunction

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : np.ndarray(K,K)
        One electron integrals for the spatial orbitals
    Ha : np.ndarray(K,K)
        One electron integrals for the alpha spin orbitals
    Hb : np.ndarray(K,K)
        One electron integrals for the beta spin orbitals
    G : np.ndarray(K,K,K,K)
        Two electron integrals for the spatial orbitals
    Ga : np.ndarray(K,K,K,K)
        Two electron integrals for the alpha spin orbitals
    Gb : np.ndarray(K,K,K,K)
        Two electron integrals for the beta spin orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nspatial : int
        Number of spatial orbitals
    nspin : int
        Number of spin orbitals (alpha and beta)
    nelec : int
        Number of electrons
    npair : int
        Number of electron pairs
        Assumes that the number of electrons is even
    nparticle : int
        Number of quasiparticles (electrons)
    ngeminal : int
        Number of geminals
    nci : int
        Number of (user-specified) Slater determinants

    Private
    -------
    _energy : float
        Electronic energy

    Abstract Properties
    -------------------
    _nci : int
        Total number of (default) Slater determinants

    Abstract Methods
    ----------------
    compute_civec
        Generates a list of Slater determinants
    compute_ci_matrix
        Generates the Hamiltonian matrix of the Slater determinants
    """
    # FIXME: turn C into property and have a better attribute name
    __metaclass__ = ABCMeta

    #
    # Default attribute values
    #
    @abstractproperty
    def _nci(self):
        """ Default number of configurations
        """
        pass

    def dict_sd_coeff(self, exc_lvl=0):
        """ Dictionary of the coefficient

        Parameters
        ----------
        exc_lvl : int
            Excitation level of the wavefunction
            0 is the ground state wavefunction
            1 is the first excited wavefunction

        Returns
        -------
        Dictionary of SD to coefficient
        """
        if not isinstance(exc_lvl, int):
            raise TypeError('Excitation level must be an integer')
        if exc_lvl < 0:
            raise ValueError('Excitation level cannot be negative')
        return {sd: coeff for sd, coeff in zip(self.civec, self.sd_coeffs[:, exc_lvl].flat)}

    #
    # Special methods
    #

    def __init__(
            self,
            # Mandatory arguments
            nelec=None,
            H=None,
            G=None,
            # Arguments handled by base Wavefunction class
            dtype=None,
            nuc_nuc=None,
            # Arguments handled by FullCI class
            nci=None,
            civec=None,
            spin=None
    ):

        super(CIWavefunction, self).__init__(
            nelec=nelec,
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
        )
        self.assign_nci(nci=nci)
        self.assign_spin(spin=spin)
        self.assign_civec(civec=civec)
        self.sd_coeffs = np.zeros([self.nci, self.nci])
        self._energy = np.zeros(self.nci)

    #
    # Assignment methods
    #
    def assign_nci(self, nci=None):
        """ Sets number of projection determinants

        Parameters
        ----------
        nci : int
            Number of configuratons

        Note
        ----
        nci is also modified in assign_civec
        """
        #FIXME: cyclic dependence on civec
        if nci is None:
            nci = self._nci
        if not isinstance(nci, int):
            raise TypeError('Number of determinants must be an integer')
        self.nci = nci

    def assign_spin(self, spin=None):
        """ Sets the spin of the projection determinants

        Parameters
        ----------
        spin : int
            Total spin of the wavefunction
            Default is no spin (all spins possible)
            0 is singlet, 0.5 and -0.5 are doublets, 1 and -1 are triplets, etc
            Positive spin means that there are more alpha orbitals than beta orbitals
            Negative spin means that there are more beta orbitals than alpha orbitals
        """
        if spin is not None and not isinstance(spin, (int, float)):
            raise TypeError('Invalid spin of the wavefunction')
        self.spin = spin

    def assign_civec(self, civec=None):
        """ Sets the Slater determinants used in the wavefunction

        Parameters
        ----------
        civec : iterable of int
            List of Slater determinants (in the form of integers that describe
            the occupation as a bitstring)
        """
        #FIXME: cyclic dependence on nci
        if civec is None:
            civec = self.compute_civec()
        if not isinstance(civec, (list, tuple)):
            raise TypeError("civec must be a list or a tuple")
        # FIXME: need to see if civec satisfies spin
        self.civec = tuple(civec)
        # NOTE: nci is modified here!!!
        if len(civec) != self.nci:
            self.nci = len(civec)

    #
    # Computation methods
    #
    @abstractmethod
    def compute_civec(self):
        """ Generates Slater determinants

        Number of Slater determinants generated is determined strictly by the size of the
        projection space (self.nci). First row corresponds to the ground state SD, then
        the next few rows are the first excited, then the second excited, etc

        Returns
        -------
        civec : list of ints
            Integer that describes the occupation of a Slater determinant as a bitstring

        """
        pass

    @abstractmethod
    def compute_ci_matrix(self):
        """ Returns Hamiltonian matrix in the Slater determinant basis

        ..math::
            H_{ij} = \big< \Phi_i \big| H \big| \Phi_j \big>

        Returns
        -------
        matrix : np.ndarray(K, K)
        """
        pass

    def compute_energy(self, include_nuc=True, exc_lvl=0):
        """ Returns the energy of the system

        Parameters
        ----------
        include_nuc : bool
            Flag to include nuclear nuclear repulsion
        exc_lvl : int
            Excitation level of the wavefunction
            0 is the ground state wavefunction
            1 is the first excited wavefunction

        Returns
        -------
        energy : float
            Total energy if include_nuc is True
            Electronic energy if include_nuc is False
        """
        if not isinstance(exc_lvl, int):
            raise TypeError('Excitation level must be an integer')
        if exc_lvl < 0:
            raise ValueError('Excitation level cannot be negative')
        nuc_nuc = self.nuc_nuc if include_nuc else 0.0
        return self._energy[exc_lvl] + nuc_nuc

    def to_other(self, Other, exc_lvl=0):
        """ Converts CIWavefunction to ProjWavefunction as best as possible

        Parameters
        ----------
        Other : ProjWavefunction class
            Class of the wavefunction to turn into
        exc_lvl : int
            Excitation level of the wavefunction
            0 is the ground state wavefunction
            1 is the first excited wavefunction

        Returns
        -------
        new_instance : Other instance
            Instance of the specified wavefunction with parameters/coefficients
            that tries to resemble self
        """
        new_instance = Other(nelec=self.nelec,
                             H=self.H,
                             G=self.G,
                             dtype=self.dtype,
                             nuc_nuc=self.nuc_nuc,
                             orbtype=self.orbtype,
                             nproj=None,
                             x=None,)
        sd_coeff = dict_sd_coeff(self, exc_lvl=0)
        def objective(x_vec):
            """ Function to minimize

            We will find the root of this function, which means that we find the
            x_vec such that the objective gives back the smallest vector

            parameters
            ----------
            x_vec : np.ndarray(K,)
                One dimensional numpy array

            Returns
            -------
            val : np.ndarray(K,)
                One dimensional numpy array
            """
            val = np.empty(self.nci, dtype=self.dtype)
            for i, sd in enumerate(self.civec):
                val[i] = new_instance.compute_overlap(sd) - sd_coeff[sd]
            return val
        result = least_squares(self.objective, new_instance.x)
        new_instance.assign_civec(result.x)
        return new_instance


    def get_density_matrix(self, notation='physicist', val_threshold=1e-4):
        """ Returns the first and second order density matrices

        Second order density matrix uses the Physicist's notation:
        ..math::
            \Gamma_{ijkl} = < \Psi | a_i^\dagger a_k^\dagger a_l a_j | \Psi >
        Chemist's notation is also implemented
        ..math::
            \Gamma_{ijkl} = < \Psi | a_i^\dagger a_j^\dagger a_k a_l | \Psi >
        but is commented out

        Paramaters
        ----------
        val_threshold : float
            If the term has weight that is less than this threshold, it is discarded
        notation : 'physicist', 'chemist'
            Flag for physicist or chemist notation
            Default is Physicist's notation

        Returns
        -------
        one_density : np.ndarray(self.nspatial, self.nspatial)
            One electron density matrix
        two_density : np.ndarray(self.nspatial, self.nspatial, self.nspatial, self.nspatial)
            Two electron density matrix

        NOTE
        ----
        I'm not 100% on the signs of the num. I feel like there is a sign change
        missing in the singlet excitation two electron density part. e.g. if I have
        ..math::
            g_{ijkj} = G_{ijkj} a_i a_j a_i^\dagger a_j^\dagger a_k a_j a_j^\dagger a_k

        there should be an extra negative sign due to changing position of
        :math:`a_j` and :math:`a_i^\dagger`
        """
        ns = self.nspatial
        assert notation in ['physicist', 'chemist']

        temp_sorting = sorted(zip(self.sd_coeffs, self.civec), key=lambda x: abs(x[0]), reverse=True)
        sorted_x, sorted_sd = zip(*temp_sorting)

        one_density = np.zeros([ns]*2)
        two_density = np.zeros([ns]*4)
        for a, sd1 in enumerate(sorted_sd):
            for b, sd2 in enumerate(sorted_sd[a:]):
                b += a
                num = sorted_x[a] * sorted_x[b]
                # orbitals that are not shared by the two determinants
                left_diff, right_diff = slater.diff(sd1, sd2)
                shared_indices = slater.occ_indices(slater.shared(sd1, sd2))

                # moving all the shared orbitals toward one another (in the middle)
                num_transpositions_0 = sum(len([j for j in shared_indices if j<i]) for i in left_diff)
                num_transpositions_1 = sum(len([j for j in shared_indices if j<i]) for i in right_diff)
                num_transpositions = num_transpositions_0 + num_transpositions_1
                sign = (-1)**num_transpositions
                num *= sign

                if len(right_diff) != len(left_diff) or len(left_diff) > 2:
                    continue
                # if they're the same
                elif len(left_diff) == 0:
                    for ind, i in enumerate(shared_indices):
                        I = spatial_index(i, ns)
                        one_density[I, I] += num
                        for j in shared_indices[ind+1:]:
                            J = spatial_index(j, ns)
                            if notation == 'physicist':
                                two_density[I, J, I, J] += num
                                two_density[J, I, J, I] += num
                            elif notation == 'chemist':
                                two_density[I, I, J, J] += num
                                two_density[J, J, I, I] += num
                            if is_alpha(i, ns) == is_alpha(j, ns):
                                if notation == 'physicist':
                                    two_density[I, J, J, I] -= num
                                    two_density[J, I, I, J] -= num
                                elif notation == 'chemist':
                                    two_density[I, J, I, J] -= num
                                    two_density[J, I, J, I] -= num
                # if single excitation
                elif len(left_diff) == 1:
                    i = left_diff[0]
                    k = right_diff[0]
                    I = spatial_index(i, ns)
                    K = spatial_index(k, ns)
                    if is_alpha(i, ns) == is_alpha(k, ns):
                        one_density[I, K] += num
                        one_density[K, I] += num
                    for j in shared_indices:
                        J = spatial_index(j, ns)
                        if is_alpha(i, ns) == is_alpha(k, ns):
                            if notation == 'physicist':
                                two_density[I, J, K, J] += num
                                two_density[J, I, J, K] += num
                                two_density[K, J, I, J] += num
                                two_density[J, K, J, I] += num
                            elif notation == 'chemist':
                                two_density[I, K, J, J] += num
                                two_density[J, J, K, I] += num
                                two_density[K, I, J, J] += num
                                two_density[J, J, I, K] += num
                        if is_alpha(i, ns) == is_alpha(j, ns) and is_alpha(j, ns) == is_alpha(k, ns):
                            if notation == 'physicist':
                                two_density[I, J, J, K] -= num
                                two_density[J, I, K, J] -= num
                                two_density[K, J, J, I] -= num
                                two_density[J, K, I, J] -= num
                            elif notation == 'chemist':
                                two_density[I, J, K, J] -= num
                                two_density[J, K, J, I] -= num
                                two_density[K, J, I, J] -= num
                                two_density[J, I, J, K] -= num
                # if double excitation
                elif len(left_diff) == 2:
                    i, j = left_diff
                    k, l = right_diff
                    I = spatial_index(i, ns)
                    J = spatial_index(j, ns)
                    K = spatial_index(k, ns)
                    L = spatial_index(l, ns)
                    if is_alpha(i, ns) == is_alpha(k, ns) and is_alpha(j, ns) == is_alpha(l, ns):
                        if notation == 'physicist':
                            two_density[I, J, K, L] += num
                            two_density[J, I, L, K] += num
                            two_density[K, L, I, J] += num
                            two_density[L, K, J, I] += num
                        elif notation == 'chemist':
                            two_density[I, K, L, J] += num
                            two_density[J, L, K, I] += num
                            two_density[K, I, J, L] += num
                            two_density[L, J, I, K] += num
                    if is_alpha(i, ns) == is_alpha(l, ns) and is_alpha(j, ns) == is_alpha(k, ns):
                        if notation == 'physicist':
                            two_density[I, J, L, K] -= num
                            two_density[J, I, K, L] -= num
                            two_density[K, L, J, I] -= num
                            two_density[L, K, I, J] -= num
                        elif notation == 'chemist':
                            two_density[I, L, K, J] -= num
                            two_density[J, K, L, I] -= num
                            two_density[K, J, I, L] -= num
                            two_density[L, I, J, K] -= num
        return one_density, two_density
