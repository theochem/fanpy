"""Antisymmeterized Product of Set divided Geminals (APsetG) Wavefunction."""
from __future__ import absolute_import, division, print_function
from ...backend import graphs
from .base_geminal import BaseGeminal

__all__ = []


class APsetG(BaseGeminal):
    """Antisymmeterized Product of Set divided Geminals (APsetG) Wavefunction.

    Attributes
    ----------
    nelec : int
        Number of electrons
    nspin : int
        Number of spin orbitals (alpha and beta)
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction
    params : np.ndarray
        Parameters of the wavefunction
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j
        to the column index of the geminal coefficient matrix
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j

    Properties
    ----------
    npairs : int
        Number of electorn pairs
    nspatial : int
        Number of spatial orbitals
    ngem : int
        Number of geminals
    spin : float, None
        Spin of the wavefunction
        :math:`\frac{1}{2}(N_\alpha - N_\beta)` (Note that spin can be negative)
        None means that all spins are allowed
    seniority : int, None
        Seniority (number of unpaired electrons) of the wavefunction
        None means that all seniority is allowed
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters
    template_params : np.ndarray
        Template for the initial guess of geminal coefficient matrix
        Depends on the attributes given

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_nspin(self, nspin)
        Assigns the number of spin orbitals
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_params(self, params)
        Assigns the parameters of the wavefunction
    assign_ref_sds(self, ref_sds=None)
        Assigns the reference Slater determinants from which the initial guess, energy, and norm are
        calculated
        Default is the first Slater determinant of projection space
    assign_orbpairs(self, orbpairs=None)
        Assigns the orbital pairs that will be used to construct geminals
    compute_permanent(self, orbpairs, deriv_row_col=None)
        Compute the permanent that corresponds to the given orbital pairs
    get_overlap(self, sd, deriv_ind=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    generate_possible_orbpairs(self, occ_indices)
        Yields the possible orbital pairs that can construct the given Slater determinant.
    """
    def generate_possible_orbpairs(self, occ_indices):
        """Yields the possible orbital pairs that can construct the given Slater determinant.

        Parameters
        ----------
        occ_indices : N-tuple of int
            Indices of the orbitals from which the Slater determinant is constructed

        Yields
        ------
        orbpairs : P-tuple of 2-tuple of ints
            Indices of the creation operators (grouped by orbital pairs) that construct the Slater
            determinant.
        """
        alpha_occ_indices = []
        beta_occ_indices = []
        for i in occ_indices:
            if i < self.nspatial:
                alpha_occ_indices.append(i)
            else:
                beta_occ_indices.append(i)
        yield from graphs.generate_biclique_pmatch(alpha_occ_indices, beta_occ_indices,
                                                   is_decreasing=False)
