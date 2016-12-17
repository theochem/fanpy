from __future__ import absolute_import, division, print_function

import numpy as np

from ..graphs import generate_biclique_pmatch
from .proj_wavefunction import ProjectionWavefunction
from .apg import APG


# FIXME: rename
class APsetG(APG):
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
                 # orbital sets
                 dict_setind_orbs=None,
                 # Arguments for saving parameters
                 params_save_name=''
    ):
        # FIXME: this fucking mess
        super(ProjectionWavefunction, self).__init__(
            nelec=nelec,
            H=H,
            G=G,
            dtype=dtype,
            nuc_nuc=nuc_nuc,
        )
        self.assign_params_save(params_save_name=params_save_name)
        self.assign_orbsets(dict_setind_orbs=dict_setind_orbs)
        self.assign_params(params=params)
        self.assign_pspace(pspace=pspace)
        if params is None:
            self.params[-1] = self.compute_energy(ref_sds=self.default_ref_sds)
        del self._energy
        self.cache = {}
        self.d_cache = {}

    def assign_orbsets(self, dict_setind_orbs=None):
        # assign orbsets
        if dict_setind_orbs is None:
            dict_setind_orbs = {0:tuple(range(self.nspatial)),
                       1:tuple(range(self.nspatial, self.nspin))}
        if not isinstance(dict_setind_orbs, dict):
            raise TypeError('dict_setind_orbs should be a dictionary')
        if not all([isinstance(value, tuple) for value in dict_setind_orbs.values()]):
            raise ValueError('dict_setind_orbs should only have values that are tuples')
        if sorted([j for i in dict_setind_orbs.values() for j in i]) != list(range(self.nspin)):
            raise ValueError('dict_setind_orbs is missing some orbitals or contains too many orbitals')
        self.dict_setind_orbs = dict_setind_orbs
        self.dict_orb_setind = {k:i for i,j in dict_setind_orbs.items() for k in j}
        # make adjacency matrix
        adjacency = np.ones((self.nspin, self.nspin), dtype=bool)
        for orbset in dict_setind_orbs.values():
            orbset = np.array(orbset)
            adjacency[orbset[:, np.newaxis], orbset] = False
        self.assign_adjacency(adjacency=adjacency)

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
        Assumes that the graph (of correlaton) is complete bipartite
        """
        # assumes complete bipartite graph

        if len(self.dict_setind_orbs) != 2:
            raise AssertionError('Automatic perfect match not supported for partite graphs with more than two sets')
        set_one, set_two = [], []
        for i in occ_indices:
            if self.dict_orb_setind[i] == 0:
                set_one.append(i)
            else:
                set_two.append(i)
        if len(set_one) == len(set_two):
            return generate_biclique_pmatch(set_one, set_two)
        else:
            return []
