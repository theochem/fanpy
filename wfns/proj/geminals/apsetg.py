""" Antisymmeterized Product of set divided Geminals
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from ... import slater
from ...graphs import generate_biclique_pmatch
from ..geminals.apg import APG

__all__ = []


class APsetG(APG):
    """ Antisymmeterized Product of Geminals Wavefunction

    Here, the geminal wavefunction is built with the perfect matchings of a graph whose edges
    are the orbital pairs that are used to construct the wavefunction. For example, the APG
    wavefunction is analogous to all perfect matchings of a complete graph. The APsetG wavefunction
    is analogous to all perfect matchings of a complete bipartite graph. Though this structure is
    very flexible, the cost of the wavefunction is restricted by the cost of permanent evaluation.
    For now, the perfect matching representation of a geminal wavefunction is restricted to the
    evaluation of permanents, so the numerically efficient approximations will also be restricted to
    those of the permanent (i.e. borchardt theorem and reference Slater determinant).

    If you want to use another function to evaluate the overlap, such as a pfaffian, you will be
    better off constructing a new child of Geminal class. Otherwise, a child of APG class should be
    sufficient after overwriting template_orbpairs and assign_pmatch_generator.

    Class Variables
    ---------------
    _nconstraints : int
        Number of constraints
    _seniority : int, None
        Seniority of the wavefunction
        None means that all seniority is allowed
    _spin : float, None
        Spin of the wavefunction
        :math:`\frac{1}{2}(N_\alpha - N_\beta)` (Note that spin can be negative)
        None means that all spins are allowed

    Attributes
    ----------
    nelec : int
        Number of electrons
    one_int : 1- or 2-tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        1-tuple for spatial (restricted) and generalized orbitals
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
    two_int : 1- or 3-tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        In physicist's notation
        1-tuple for spatial (restricted) and generalized orbitals
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components)
    dtype : {np.float64, np.complex128}
        Numpy data type
    nuc_nuc : float
        Nuclear-nuclear repulsion energy
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    pspace : tuple of gmpy2.mpz
        Slater determinants onto which the wavefunction is projected
    ref_sds : tuple of gmpy2.mpz
        Slater determinants that will be used as a reference for the wavefunction (e.g. for
        initial guess, energy calculation, normalization, etc)
    params : np.ndarray
        Parameters of the wavefunction (including energy)
    cache : dict of sd to float
        Cache of the overlaps that are calculated for each Slater determinant encountered
    d_cache : dict of gmpy2.mpz to float
        Cache of the derivative of overlaps that are calculated for each Slater determinant and
        derivative index encountered
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j
        to the column index of the geminal coefficient matrix
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j
    template_coeffs : np.ndarray
        Initial guess coefficient matrix for the given reference Slater determinants
    partitions : tuple of np.ndarray of int
        Sets of indices that are not connected (correlated) to one another


    Properties
    ----------
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals
    nparams : int
        Number of parameters
    nproj : int
        Number of Slater determinants
    npair : int
        Number of electron pairs
    ngem : int
        Number of geminals
    template_orbpairs : tuple
        List of orbital pairs that will be used to construct the geminals

    Method
    ------
    __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_integrals(self, one_int, two_int, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
    assign_pspace(self, pspace=None)
        Assigns the tuple of Slater determinants onto which the wavefunction is projected
        Default uses `generate_pspace`
    generate_pspace(self)
        Generates the default tuple of Slater determinants with the appropriate spin and seniority
        in increasing excitation order.
        The number of Slater determinants is truncated by the number of parameters plus a magic
        number (42)
    assign_ref_sds(self, ref_sds=None)
        Assigns the reference Slater determinants from which the initial guess, energy, and norm are
        calculated
        Default is the first Slater determinant of projection space
    assign_params(self, params=None)
        Assigns the parameters of the wavefunction (including energy)
        Default contains coefficients from abstract property, `template_coeffs`, and the energy of
        the reference Slater determinants with the coefficients from `template_coeffs`
    assign_orbpairs(self, orbpairs=None)
        Assigns the orbital pairs that will be used to construct geminals
    get_overlap(self, sd, deriv=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    compute_norm(self, ref_sds=None, deriv=None)
        Calculates the norm from given Slater determinants
        Default `ref_sds` is the `ref_sds` given by the intialization
        Default is no derivatization
    compute_hamitonian(self, slater_d, deriv=None)
        Calculates the expectation value of the Hamiltonian projected onto the given Slater
        determinant, `slater_d`
        By default no derivatization
    compute_energy(self, include_nuc=False, ref_sds=None, deriv=None)
        Calculates the energy projected onto given Slater determinants
        Default `ref_sds` is the `ref_sds` given by the intialization
        By default, electronic energy, no derivatization
    objective(self, x, weigh_constraints=True)
        Objective of the equations that will need to be solved (to solve the Projected Schrodinger
        equation)
    jacobian(self, x, weigh_constraints=True)
        Jacobian of the objective
    compute_overlap(self, sd, deriv=None)
        Calculates the overlap between the wavefunction and a Slater determinant
    """
    @property
    def template_orbpairs(self):
        """ List of orbital pairs that will be used to construct the geminals
        """
        return tuple((i, j) for i in range(self.nspatial) for j in range(self.nspatial, self.nspin))


    def assign_orbpairs(self, orbpairs=None):
        """ Assigns the orbital pairs that will be used to construct geminals

        Parameters
        ----------
        orbpairs : tuple/list of 2-tuple of ints
            Each 2-tuple is an orbital pair that is allowed to contribute to geminals

        Raises
        ------
        TypeError
            If `orbpairs` is not a tuple/list of 2-tuples
        ValueError
            If an orbital pair has the same integer
        NotImplementedError
            If orbital pairs are given such that the resulting graph is not complete
            If orbital pairs are given such that the resulting graph is not bipartite
        """
        super(APsetG, self).assign_orbpairs(orbpairs=orbpairs)
        # set adjacency matrix
        adjacency = np.zeros((self.nspin, self.nspin), dtype=bool)
        for i, j in self.dict_orbpairs_ind.keys():
            adjacency[i, j] = True
        adjacency += adjacency.T
        # find partitions
        partitions = []
        for vertex_adjacency in adjacency:
            indices = np.where(-vertex_adjacency)[0]
            if indices in partitions:
                continue
            # make sure that the vertices that are not connected are all the same for each partition
            # of vertices (i.e. graph is complete partite graph)
            if not all(np.all(indices[:, np.newaxis] - partite_set != 0)
                       for partite_set in partitions):
                raise NotImplementedError('Only orbital pairs corresponding to complete partite'
                                          ' graphs are supported')
            partitions.append(indices)
        if len(partitions) > 2:
            raise NotImplementedError('Only bipartite sets are supported')
        self.partitions = tuple(partitions)


    def assign_pmatch_generator(self, pmatch_generator=None):
        """ Assigns the function that is used to generate the perfect matchings of a Slater
        determinant

        Parameters
        ----------
        pmatch_generator : function
            Function that returns the perfect matchings available for a given Slater determinant
            Input is list/tuple of orbitals indices that are occupied in the Slater determinant
            Generates a `npair`-tuple of 2-tuple where each 2-tuple is an orbital pair.
            For example, {0b1111:(((0,2),(1,3)), ((0,1),(2,3)))} would associate the
            Slater determinant `0b1111` with pairing schemes ((0,2), (1,3)), where
            `0`th and `2`nd orbitals are paired with `1`st and `3`rd orbitals, respectively,
            and ((0,1),(2,3)) where `0`th and `1`st orbitals are paired with `2`nd
            and `3`rd orbitals, respectively.

        Raises
        ------
        TypeError
            If pmatch_generator is not a function
        ValueError
            If pairing scheme contains a pair with the wrong number (not 2) of orbital indices
            If pairing scheme contains more than `npair` orbital pairs
            If pairing scheme contains orbital indices that are not in the given Slater determinant
        """
        if pmatch_generator is None:
            def func(occ_indices):
                """ Separates occupied indices into the contribution by the two sets before feeding
                it to the generatE_biclique_pmatch
                """
                set_one = []
                set_two = []
                for i in occ_indices:
                    if i in self.partitions[0]:
                        set_one.append(i)
                    else:
                        set_two.append(i)
                return generate_biclique_pmatch(set_one, set_two)
            pmatch_generator = func

        if not callable(pmatch_generator):
            raise TypeError('Given pmatch_generator is not a function')
        # quite expensive
        for sd in self.pspace:
            occ_indices = slater.occ_indices(sd)
            for scheme in pmatch_generator(occ_indices):
                if not all(len(pair) == 2 for pair in scheme):
                    raise ValueError('All pairing in the scheme must be in 2')
                if len(scheme) != self.npair:
                    raise ValueError('There is at least one redundant orbital pair')
                if set(occ_indices) != set(orb_ind for pair in scheme for orb_ind in pair):
                    raise ValueError('Pairing scheme contains orbitals that is not contained in'
                                     ' the provided Slater determinant')

        self.pmatch_generator = pmatch_generator
