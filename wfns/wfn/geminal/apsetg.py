"""Antisymmeterized Product of Set divided Geminals (APsetG) Wavefunction."""
from wfns.backend import graphs
from wfns.wfn.geminal.base import BaseGeminal


# FIXME: do not use all orbital pairs in APsetG wavefunction (i.e. define assign_orbpairs)
class APsetG(BaseGeminal):
    r"""Antisymmeterized Product of Set divided Geminals (APsetG) Wavefunction.

    .. math::

        G^\dagger_p = \sum_{i \in S_A} \sum_{j \in S_B} C_{pij} a^\dagger_i a^\dagger_j

    where :math:`S_A` and :math:`S_B` are two sets of orbitals that are mutually exclusive (no
    shared orbitals) and exhaustive (form the complete basis set as a whole).

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction.
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j to the
        column index of the geminal coefficient matrix.
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    param_shape : tuple of int
        Shape of the parameters.
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    npair : int
        Number of electron pairs.
    norbpair : int
        Number of orbital pairs used to construct the geminals.
    template_params : np.ndarray
        Default parameters of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_dtype(self, dtype)
        Assign the data type of the parameters.
    assign_memory(self, memory=None)
        Assign memory available for the wavefunction.
    assign_ngem(self, ngem=None)
        Assign the number of geminals.
    assign_orbpairs(self, orbpairs=None)
        Assign the orbital pairs that will be used to construct the geminals.
    assign_params(self, params=None, add_noise=False)
        Assign the parameters of the geminal wavefunction.
    compute_permanent(self, col_inds, row_inds=None, deriv=None)
        Compute the permanent of the matrix that corresponds to the given orbital pairs.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.
    generate_possible_orbpairs(self, occ_indices)
        Yield the possible orbital pairs that can construct the given Slater determinant.

    """
    def generate_possible_orbpairs(self, occ_indices):
        """Yield the possible orbital pairs that can construct the given Slater determinant.

        Generates all possible orbital pairing schemes where one orbital is selected from one set
        and the other orbital is selected from the other. This is equivalent to finding all the
        perfect matchings (pairing schemes) within a bipartite graph with the given two sets
        (:math:`S_A` and :mat:`S_B`) of vertices (occupied orbitals).

        Parameters
        ----------
        occ_indices : N-tuple of int
            Indices of the orbitals from which the Slater determinant is constructed.
            Must be strictly increasing.

        Yields
        ------
        orbpairs : P-tuple of 2-tuple of ints
            Indices of the creation operators (grouped by orbital pairs) that construct the Slater
            determinant.
        sign : int
            Signature of the transpositions required to shuffle the `orbpairs` back into the
            original order in `occ_indices`.

        """
        alpha_occ_indices = []
        beta_occ_indices = []
        for i in occ_indices:
            if i < self.nspatial:
                alpha_occ_indices.append(i)
            else:
                beta_occ_indices.append(i)
        yield from graphs.generate_biclique_pmatch(alpha_occ_indices, beta_occ_indices, occ_indices,
                                                   is_decreasing=False)
