from __future__ import absolute_import, division, print_function

import numpy as np
import itertools as it
from gmpy2 import mpz

from .. import slater
from ..sd_list import ci_sd_list
from ..math_tools import permanent_ryser
from ..graphs import generate_complete_pmatch
from .proj_wavefunction import ProjectionWavefunction
from .proj_hamiltonian import hamiltonian

# FIXME: rename
class APG(ProjectionWavefunction):
    """ Antisymmetric Product Geminals

    ..math::
        G_p^\dagger = \sum_{ij} C_{p;ij} a_i^\dagger a_j^\dagger

    ..math::
        \big| \Psi_{\mathrm{APG}} \big>
        &= \prod_{p=1}^P G_p^\dagger \big| \theta \big>\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} | C(\mathbf{m}) |^+ \big| \mathbf{m} \big>
    where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a
    Slater determinant.

    Note that for now, we make no assumptions about the structure of the :math:`C_{p;ij}`.
    We do not impose conditions like :math:`C_{p;ij} = C_{p;ji}`

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
    adjacency : np.ndarray
        Adjacency matrix that shows the correlation between any orbital pairs.
        This matrix will be used to get the pairing scheme
    dict_orbpair_gem : dict of (int, int) to int
        Dictionary of indices of the orbital pairs to the index of the geminal
    dict_gem_orbpair : dict of int to (int, int)
        Dictionary of indices of index of the geminal to the the orbital pairs

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
    template_coeffs : np.ndarray(K)
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
    assign_adjacency
        Assigns the adjacency matrix which is used to obtain the get the coupling scheme
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
                 # Adjacnecy
                 adjacency=None,
                 # Arguments for saving parameters
                 save_params=False
    ):
        super(ProjectionWavefunction, self).__init__(
            nelec=nelec,
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
        )
        self.save_params = save_params
        self.assign_adjacency(adjacency=adjacency)
        self.assign_params(params=params)
        self.assign_pspace(pspace=pspace)
        if params is None:
            self.params[-1] = self.compute_energy(ref_sds=self.default_ref_sds)
        del self._energy
        self.cache = {}
        self.d_cache = {}

    def assign_adjacency(self, adjacency=None):
        """ Assigns the adjacency matrix

        Parameters
        ----------
        adjacency : np.ndarray
            Adjacency matrix
            Square boolean numpy array that describes the correlations between orbitals
        """
        if adjacency is None:
            adjacency = np.ones((self.nspin, self.nspin), dtype=bool)
            adjacency -= np.diag(np.diag(adjacency))
        if not isinstance(adjacency, np.ndarray):
            raise TypeError, 'Adjacency matrix must be a numpy array'
        if adjacency.dtype != 'bool':
            raise TypeError, 'Adjacency matrix must be a boolean array'
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError, 'Adjacency matrix must be square'
        if not np.allclose(adjacency, adjacency.T):
            raise ValueError, 'Adjacency matrix is not symmetric'
        if not np.all(np.diag(adjacency) == False):
            raise ValueError, 'Adjacency matrix must have a diagonal of zero (or False)'
        # here, we encountered problems when due to numpy passing arrays as references
        # ()modifying one affected the other). SO we make a copy of it
        adjacency = np.copy(adjacency)
        self.adjacency = np.copy(adjacency)
        # remove the lower triangular matrix (removes repetition )
        adjacency[np.tril_indices(adjacency.shape[0])] = False
        # find all the edges/orbital pairs that are correlated
        orbpairs = zip(*np.where(adjacency != False))
        # if C_{p;ij} = C_{p;ji}
        # orbpairs = zip(*np.where(np.triu(adjacency, k=1) != False))
        self.dict_orbpair_gem = {orbpair:i for i, orbpair in enumerate(orbpairs)}
        self.dict_gem_orbpair = {i:orbpair for i, orbpair in enumerate(orbpairs)}

    @property
    def template_coeffs(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_coeffs : np.ndarray(K, )

        Note
        ----
        Assumes that C_{p;ij} = C_{p;ji}

        """
        gem_coeffs = np.zeros((self.npair, self.ngem), dtype=self.dtype)
        for i in range(self.npair):
            gem_ind = self.dict_orbpair_gem[(i, i+self.nspatial)]
            gem_coeffs[i, gem_ind] = 1
        return gem_coeffs

    @property
    def ngem(self):
        """ Number of geminals

        Returns
        -------
        int

        """
        return len(self.dict_gem_orbpair)


    def default_pmatch_generator(self, occ_indices):
        """ Generator for the perfect matchings needed to construct the Slater determinant

        Parameters
        ----------
        occ_indices : list of int
            List of spin orbital indices that are occupied in a given Slater determinant

        Returns
        -------
        Generator for the perfect matchings

        Notes
        -----
        Assumes that graph (of correlation) is a complete
        """
        return generate_complete_pmatch(occ_indices)

    def compute_pspace(self, num_sd):
        """ Generates Slater determinants to project onto

        Parameters
        ----------
        num_sd : int
            Number of Slater determinants to generate

        Returns
        -------
        pspace : list of gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        """
        return ci_sd_list(self, num_sd)

    def compute_overlap(self, sd, pairing_schemes=None, deriv=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.
        ..math::
            \big< \Phi_k \big| \Psi_{\mathrm{APIG}} \big>
            &= \big< \Phi_k \big| \prod_{p=1}^P G_p^\dagger \big| \theta \big>\\
            &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} | C(\mathbf{m}) |^+ \big< \Phi_k \big| \mathbf{m} \big>
            &= | C(\Phi_k) |^+
        where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a
        Slater determinant (DOCI).

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        pairing_scheme : generator of list of list of 2 ints
            Contains all of available pairing schemes for the given Slater determinant
            For example, {0b1111:(((0,2),(1,3)), ((0,1),(2,3)))} would associate the
            Slater determinant `0b1111` with pairing schemes ((0,2), (1,3)), where
            `0`th and `2`nd orbitals are paired with `1`st and `3`rd orbitals, respectively,
            and ((0,1),(2,3)) where `0`th and `1`st orbitals are paired with `2`nd
            and `3`rd orbitals, respectively.
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        overlap : float
        """
        # caching is done wrt mpz objects, so you should convert sd to mpz first
        sd = mpz(sd)
        # get indices of the occupied orbitals
        occ_indices = slater.occ_indices(sd)
        # get the pairing schemes
        if pairing_schemes is None:
            pairing_schemes = self.default_pmatch_generator(occ_indices)
        # FIXME: this part is not cheap. it should move somewhere else.
        else:
            pairing_schemes, temp = it.tee(pairing_schemes)
            for i in temp:
                # if C_{p;ij} = C_{p;ji}
                for j in i:
                    # assert j[0] < j[1]
                    assert all(len(j) == 2), 'All pairing in the scheme must be in 2'
                assert set(occ_indices) == set(k for j in i for k in j), 'Pairing '
                'scheme must use the same indices as the occupied orbitals of the '
                'given Slater determinant'
        # FIXME: pairing_schemes is assumed to be good (there are geminals such that all schemes present are possible)
        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.npair, self.ngem)

        val = 0.0
        # if no derivatization
        if deriv is None:
            for scheme in pairing_schemes:
                sign = slater.find_num_trans([j for i in scheme for j in i],
                                             occ_indices,
                                             is_creator=True)
                indices = [self.dict_orbpair_gem[i] for i in scheme]
                matrix = gem_coeffs[:, indices]
                val += sign * permanent_ryser(matrix)
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.params.size - 1:
            row_to_remove = deriv // self.ngem
            col_to_remove = deriv % self.ngem
            orbs_to_remove = self.dict_gem_orbpair[col_to_remove]
            # both orbitals of the geminal must be present in the Slater determinant
            if orbs_to_remove[0] in occ_indices and orbs_to_remove[1] in occ_indices:
                for scheme in pairing_schemes:
                    sign = slater.find_num_trans([j for i in scheme for j in i],
                                                 occ_indices,
                                                 is_creator=True)
                    # geminal must be an allowed pairing schemes
                    if orbs_to_remove in scheme:
                        row_inds = [i for i in range(self.npair) if i!=row_to_remove]
                        col_inds = [self.dict_orbpair_gem[i] for i in scheme if self.dict_orbpair_gem[i]!=col_to_remove]
                        if len(row_inds) == 0 and len(col_inds) == 0:
                            val += sign * 1
                        else:
                            val += sign * permanent_ryser(gem_coeffs[row_inds][:, col_inds])
                self.d_cache[(sd, deriv)] = val
        return val

    def compute_hamiltonian(self, sd, deriv=None):
        """ Computes the hamiltonian of the wavefunction with respect to a Slater
        determinant

        ..math::
            \big< \Phi_i \big| H \big| \Psi_{mathrm{APIG}} \big>

        Since only Slater determinants from DOCI will be used, we can use the DOCI
        Hamiltonian

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        float
        """
        return sum(hamiltonian(self, sd, self.orb_type, deriv=deriv))

    # FIXME: remove
    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in
        ProjectionWavefunction.compute_norm

        Some of the cache are emptied because the parameters are rewritten
        """
        # # build geminal coefficient
        # gem_coeffs = self.params[:-1].reshape(self.npair, self.ngem)
        # # normalize the geminals
        # norm = np.sum(gem_coeffs**2, axis=1)
        # gem_coeffs *= np.abs(norm[:, np.newaxis])**(-0.5)
        # # flip the negative norms
        # gem_coeffs[norm < 0, :] *= -1
        # normalize the wavefunction
        norm = self.compute_norm()
        gem_coeffs *= norm**(-0.5 / self.npair)
        # set attributes
        self.params[:-1] = gem_coeffs.flatten()
        # empty cache
        for sd in self.default_ref_sds:
            del self.cache[sd]
            for i in (j for j in self.d_cache.keys() if j[0] == sd):
                del self.d_cache[i]
            # Following requires d_cache to be a dictionary of dictionary
            # self.d_cache[sd] = {}
