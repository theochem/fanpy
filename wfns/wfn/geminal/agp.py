"""Antisymmeterized Geminal Power (AGP) Wavefunction.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from wfns.wavefunction.geminals.apg import APG
from wfns.backend import math_tools

__all__ = []


class AGP(APG):
    """Antisymmeterized Geminal Power (AGP) Wavefunction.

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
    @property
    def template_params(self):
        """Return the template of the parameters in a AGP wavefunction.

        Uses the spatial orbitals (alpha beta pair) of HF ground state as reference.

        Returns
        -------
        np.ndarray

        Note
        ----
        Need `nelec`, `norbpair` (i.e. `dict_ind_orbpair`), and `dtype`
        """
        params = np.zeros((1, self.norbpair), dtype=self.dtype)
        for i in range(self.ngem):
            try:
                col_ind = self.dict_orbpair_ind[(i, i+self.nspatial)]
            except KeyError as error:
                raise ValueError('Given orbital pairs do not contain the default orbitals pairs '
                                 'used in the template. Please give a different set of orbital '
                                 'pairs or provide the paramaters') from error
            params[:, col_ind] += 1
        return params

    def compute_permanent(self, col_inds, row_inds=None, deriv_row_col=None):
        """Compute the permanent that corresponds to the given orbital pairs

        Parameters
        ----------
        col_inds : np.ndarray
            Indices of the columns of geminal coefficient matrices that will be used.
        row_inds : np.ndarray
            Indices of the rows of geminal coefficient matrices that will be used.
        deriv : 2-tuple of int, None
            Row and column indices of the element with respect to which the permanent is derivatized
            Default is no derivatization

        Returns
        -------
        permanent :float
        """
        col_inds = np.array(col_inds)

        if deriv_row_col is None:
            return np.math.factorial(self.ngem) * np.prod(self.params[:, col_inds])
        else:
            # cut out rows and columns that corresponds to the element with which the permanent is
            # derivatized
            col_inds_trunc = col_inds[col_inds != deriv_row_col[1]]
            if col_inds_trunc.size == col_inds.size:
                return 0.0
            elif col_inds_trunc.size == 0:
                return np.math.factorial(self.ngem)
            else:
                return np.math.factorial(self.ngem) * np.prod(self.params[:, col_inds_trunc])
