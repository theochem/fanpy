from __future__ import absolute_import, division, print_function

import numpy as np
from gmpy2 import mpz

from .. import slater
from ..math_tools import permanent_ryser

from .proj_wavefunction import ProjectionWavefunction
from .apg import APG


class APseqG(APG):
    """ Antisymmetric Product of Sequantially Interacting Geminals

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : np.ndarray(K,K) or tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        If tuple of np.ndarray (length 2), one electron integrals for the (alpha, alpha)
        and the (beta, beta) unrestricted orbitals
    G : np.ndarray(K,K,K,K) or tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        If tuple of np.ndarray (length 3), two electron integrals for the
        (alpha, alpha, alpha, alpha), (alpha, beta, alpha, beta), and
        (beta, beta, beta, beta) unrestricted orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nelec : int
        Number of electrons
    orb_type : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    params : np.ndarray(K)
        Guess for the parameters
        Iteratively updated during convergence
        Initial guess before convergence
        Coefficients after convergence
    cache : dict of mpz to float
        Cache of the Slater determinant to the overlap of the wavefunction with this
        Slater determinant
    d_cache : dict of (mpz, int) to float
        Cache of the Slater determinant to the derivative(with respect to some index)
        of the overlap of the wavefunction with this Slater determinan

    Properties
    ----------
    _methods : dict of func
        Dictionary of methods that are used to solve the wavefunction
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals
    npair : int
        Number of electron pairs (rounded down)
    nparam : int
        Number of parameters used to define the wavefunction
    nproj : int
        Number of Slater determinants to project against
    ref_sd : int or list of int
        Reference Slater determinants with respect to which the norm and the energy
        are calculated
        Integer that describes the occupation of a Slater determinant as a bitstring
        Or list of integers
    template_params : np.ndarray(K)
        Default numpy array of parameters.
        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

    Method
    ------
    __init__(nelec=None, H=None, G=None, dtype=None, nuc_nuc=None, orb_type=None)
        Initializes wavefunction
    __call__(method="default", **kwargs)
        Solves the wavefunction
    assign_dtype(dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_integrals(H, G, orb_type=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
        (and the wavefunction)
    assign_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_nelec(nelec)
        Assigns the number of electrons
    _solve_least_squares(**kwargs)
        Solves the system of nonliear equations (and the wavefunction) using
        least squares method
    assign_params(params=None)
        Assigns the parameters used to describe the wavefunction.
        Adds random noise from the template if necessary
    assign_pspace(pspace=None)
        Assigns projection space
    overlap(sd, deriv=None)
        Retrieves overlap from the cache if available, otherwise compute overlap
    compute_norm(sd=None, deriv=None)
        Computes the norm of the wavefunction
    compute_energy(include_nuc=False, sd=None, deriv=None)
        Computes the energy of the wavefunction
    objective(x)
        The objective (system of nonlinear equations) associated with the projected
        Schrodinger equation
    jacobian(x)
        The Jacobian of the objective
    compute_pspace
        Generates a tuple of Slater determinants onto which the wavefunction is projected
    compute_overlap
        Computes the overlap of the wavefunction with one or more Slater determinants
    compute_hamiltonian
        Computes the hamiltonian of the wavefunction with respect to one or more Slater
        determinants
        By default, the energy is determined with respect to ref_sd
    normalize
        Normalizes the wavefunction (different definitions available)
        By default, the norm should the projection against the ref_sd squared
    """

    def __init__(self,
                 # Mandatory arguments
                 nelec=None,
                 H=None,
                 G=None,
                 # Arguments handled by base Wavefunction class
                 dtype=None,
                 nuc_nuc=None,
                 # Arguments handled by ProjWavefunction class
                 params=None,
                 pspace=None,
                 # sequence list
                 seq_list=None
                 ):
        super(ProjectionWavefunction, self).__init__(nelec=nelec,
                                                     H=H,
                                                     G=G,
                                                     dtype=dtype,
                                                     nuc_nuc=nuc_nuc,
                                                     )
        self.assign_params(params=params)
        self.assign_pspace(pspace=pspace)
        self.assign_seq_list(seq_list=seq_list)
        del self._energy
        self.cache = {}
        self.d_cache = {}
        self.dict_orbpair_gem = {}
        self.dict_gem_orbpair = {}

    # Remove methods from parent
    # FIXME: there probably is a better way to do this.
    def assign_adjacency(self, adjacency=None):
        """ Method from APG that will not be used in this class

        Parameters
        ----------
        adjacency

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def default_pmatch_generator(self, occ_indices):
        """ Method from APG that will not be used in this class

        Parameters
        ----------
        occ_indices

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    # New assignment method
    def assign_seq_list(self, seq_list=None):
        """ Assigns the list of sequence orders that will be included

        Parameters
        ----------
        seq_list : None, list of ints
            Order of sequence that will be included in the wavefunction
            Default is 0

        """
        if seq_list is None:
            seq_list = [0]
        if hasattr(seq_list, '__iter__'):
            raise TypeError('seq_list must be iterable')
        if any(i<0 for i in seq_list):
            raise ValueError('All of the sequence order must be positive')
        self.seq_list = seq_list

    #FIXME: this needs to be implemented
    @property
    def template_params(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_params : np.ndarray(K, )

        """
        raise NotImplementedError

    def compute_overlap(self, sd, deriv=None, seq_list=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization
        seq_list : None, list
            Order of the sequences to include
            Default is 0

        Returns
        -------
        overlap : float
        """
        # caching is done wrt mpz objects, so you should convert sd to mpz first
        sd = mpz(sd)

        # get indices of the occupied orbitals
        occ_indices = slater.occ_indices(sd)
        # because orbitals are ordered by alpha orbs then beta orbs,
        # we need to interleave/shuffle them to pair the alpha and the beta orbitals together
        occ_interleave_indices = np.array([2*i if i<self.nspatial
                                           else 2*(i-self.nspatial)+1
                                           for i in occ_indices])
        # find differences between any two indices
        indices_diff = np.abs(occ_interleave_indices - occ_interleave_indices[:, np.newaxis])

        # find orbital indices that correspond to each sequence
        orb_indices_pairs = []
        for seq in seq_list:
            # find orbital that correspond to sequence order
            seq_orbs = np.where(np.triu(indices_diff)==seq)
            # append to list after pairing up the orbitals into pairs
            orb_indices_pairs.append(zip(*seq_orbs))
        # remove repeated indices
        orb_indices_pairs = list(set(orb_indices_pairs))

        # convert orbital indices to geminal indices
        gem_indices = []
        for index in orb_indices_pairs:
            try:
                gem_indices.append(self.dict_orbpair_gem[index])
            except KeyError:
                self.dict_orbpair_gem[index] = self.ngem
                self.dict_gem_orbpair[self.ngem] = index
                gem_indices.append(self.ngem)

        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.npair, self.ngem)

        val = 0.0
        # if no derivatization
        if deriv is None:
            val = permanent_ryser(gem_coeffs[:, gem_indices])
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.params.size - 1:
            row_to_remove = deriv // self.ngem
            col_to_remove = deriv % self.ngem
            orbs_to_remove = self.dict_gem_orbpair[col_to_remove]
            # both orbitals of the geminal must be present in the Slater determinant
            if orbs_to_remove[0] in occ_indices and orbs_to_remove[1] in occ_indices:
                row_inds = [i for i in range(self.npair) if i != row_to_remove]
                col_inds = [self.dict_orbpair_gem[i] for i in gem_indices if
                            self.dict_orbpair_gem[i] != col_to_remove]
                # NOTE: length of row_inds and col_inds must be the same
                if len(row_inds) == 0 and len(col_inds) == 0:
                    val += 1
                else:
                    val += permanent_ryser(gem_coeffs[row_inds][:, col_inds])
                self.d_cache[(sd, deriv)] = val
        return val
