"""Antisymmetric Product of One-Reference-Orbital (AP1roG) Geminals wavefunction."""
from __future__ import absolute_import, division, print_function
import numpy as np
from ...backend import slater
from .. import base_wavefunction
from .apig import APIG

__all__ = []


class AP1roG(APIG):
    """Antisymmetric Product of One-Reference-Orbital (AP1roG) Geminals Wavefunction

    .. math::

        \ket{\Psi_{\mathrm{AP1roG}}}
        &= \prod_{q=1}^P T_q^\dagger \ket{\theta}\\
        &= \prod_{q=1}^P \left( a_q^\dagger a_{\bar{q}}^\dagger +
           \sum_{i=P+1}^B c_{q;i}^{(\mathrm{AP1roG})} a_i^\dagger a_{\bar{i}}^\dagger \right)
           \ket{\theta}\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}}
           | C(\mathbf{m})_{\mathrm{AP1roG}} |^+ \ket{\mathbf{m}}

    where :math:`P` is the number of electron pairs and :math:`\mathbf{m}` is a seniority-zero
    Slater determinant.

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
    cache : dict of sd to float
        Cache of the overlaps that are calculated for each Slater determinant encountered
    d_cache : dict of gmpy2.mpz to float
        Cache of the derivative of overlaps that are calculated for each Slater determinant and
        derivative index encountered
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j
        to the column index of the geminal coefficient matrix
        These orbital pairs are not occupied in the reference Slater determinant
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j
        These orbital pairs are not occupied in the reference Slater determinant
    dict_reforbpair_ind
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j
        to the row index of the geminal coefficient matrix
        These orbital pairs are occupied in the reference Slater determinant

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
    __init__(self, nelec, one_int, two_int, dtype=None)
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
    def __init__(self, nelec, nspin, dtype=None, ngem=None, orbpairs=None, ref_sd=None, params=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons
        nspin : int
            Number of spin orbitals
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type
            Default is `np.float64`
        ngem : int, None
            Number of geminals
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal
        ref_sd : int, gmpy2.mpz, None
            Reference Slater determinant
            Default is the HF ground state
        params : np.ndarray
            Geminal coefficient matrix

        Note
        ----
        Need to skip over APIG.__init__ because `assign_ref_sd` must come before `assign_params`.
        """
        super(base_wavefunction.BaseWavefunction, self).__init__(nelec, nspin, dtype=dtype)
        self.assign_ngem(ngem=ngem)
        self.assign_ref_sd(sd=ref_sd)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_params(params=params)

    @property
    def template_params(self):
        """Return the template of the parameters in a AP1roG wavefunction.

        Uses the spatial orbitals (alpha beta pair) of HF ground state as reference.

        Returns
        -------
        np.ndarray

        Note
        ----
        Need `nelec`, `norbpair` (i.e. `dict_ind_orbpair`), and `dtype`
        """
        params = np.zeros((self.ngem, self.norbpair), dtype=self.dtype)
        return params

    def assign_ref_sd(self, sd=None):
        """Set the reference Slater determinant.

        Parameters
        ----------
        sd : int, gmpy2.mpz, None
            Slater determinant the the AP1roG will use as a reference
            Default is the HF ground state

        Raises
        ------
        TypeError
            If given sd cannot be turned into a Slater determinant (i.e. not integer or list of
            integers)
        ValueError
            If given sd does not have the correct spin
            If given sd does not have the correct seniority

        Note
        ----
        Depends on `nelec`, `nspin`, `spin`, `seniority`
        Multiple Slater determinant references is not allowed. Passing an list of ints will be
        treated like the indices of the occupied orbitals rather than the corresponding Slater
        determinants.
        """
        if sd is None:
            sd = slater.ground(self.nelec, self.nspin)
        sd = slater.internal_sd(sd)
        if slater.total_occ(sd) != self.nelec:
            raise ValueError('Given Slater determinant does not have the correct number of '
                             'electrons')
        elif self.spin is not None and slater.get_spin(sd, self.nspatial):
            raise ValueError('Given Slater determinant does not have the correct spin.')
        elif self.seniority is not None and slater.get_seniority(sd, self.nspatial):
            raise ValueError('Given Slater determinant does not have the correct seniority.')
        self.ref_sd = sd

    def assign_ngem(self, ngem=None):
        """Set the number of geminals.

        Parameters
        ----------
        ngem : int, None
            Number of geminals

        Raises
        ------
        TypeError
            If number of geminals is not an integer
        ValueError
            If number of geminals is less than the number of electron pairs
        NotImplementedError
            If number of geminals is not equal to the number of electron pairs

        Note
        ----
        Needs to have `npair` defined (i.e. `nelec` must be defined)
        """
        super().assign_ngem(ngem)
        if self.ngem != self.npair:
            raise NotImplementedError('Currently, AP1roG does not support more than the exactly '
                                      ' right number of geminals (i.e. number of electron pairs).')

    def assign_orbpairs(self, orbpairs=None):
        """Set the orbital pairs that will be used to construct the AP1roG.

        Parameters
        ----------
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal
            Default is all possible orbital pairs

        Raises
        ------
        TypeError
            If `orbpairs` is not an iterable
            If an orbital pair is not given as a list or a tuple
            If an orbital pair does not contain exactly two elements
            If an orbital index is not an integer
        ValueError
            If an orbital pair has the same integer
            If an orbital pair occurs more than once

        Note
        ----
        Must have `ref_sd` and `nspin` defined for the default option
        Remove columns that corresponds to the reference Slater determinant
        """
        super().assign_orbpairs(orbpairs=orbpairs)
        # Need to split up the orbital pairs by those occuped by the reference Slater determinant
        # and those that are not
        occ_counter = 0
        vir_counter = 0
        # delete existing dict_orbpair_ind (there still is dict_ind_orbpair)
        del self.dict_orbpair_ind
        dict_orbpair_ind = {}
        dict_reforbpair_ind = {}
        for i in range(self.norbpair):
            orbpair = self.dict_ind_orbpair[i]
            # if orbital pair is occupied in the reference Slater determinant
            if slater.occ(self.ref_sd, orbpair[0]) and slater.occ(self.ref_sd, orbpair[1]):
                dict_reforbpair_ind[orbpair] = occ_counter
                occ_counter += 1
            # otherwise
            else:
                dict_orbpair_ind[orbpair] = vir_counter
                vir_counter += 1
        self.dict_orbpair_ind = dict_orbpair_ind
        self.dict_reforbpair_ind = dict_reforbpair_ind
        self.dict_ind_orbpair = {i: orbpair for orbpair, i in self.dict_orbpair_ind.items()}

    def get_overlap(self, sd, deriv=None):
        """Compute the overlap between the AP1roG wavefunction and a Slater determinant.

        The results are cached in self.cache and self.d_cache.

        .. math::

            \big| \Psi \big>
            &= \prod_{p=1}^{N_{gem}} \sum_{pq} C_{pq} a^\dagger_p a^\dagger_q \big| \theta \big>\\
            &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} |C(\mathbf{m})|^+
            \big| \mathbf{m} \big>

        where :math:`N_{gem}` is the number of geminals, :math:`\mathbf{m}` is a Slater determinant.

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant as a bitstring
        deriv : None, int
            Index of the paramater with respect to which the overlap is derivatized
            Default is no derivatization

        Returns
        -------
        overlap : float
        """
        sd = slater.internal_sd(sd)
        try:
            if deriv is None:
                return self.cache[sd]
            else:
                return self.d_cache[(sd, deriv)]
        except KeyError:
            # get indices of the occupied orbitals
            orbs_annihilated, orbs_created = slater.diff(self.ref_sd, sd)

            # if different number of electrons
            if len(orbs_annihilated) != len(orbs_created):
                return 0.0
            # if different seniority
            if slater.get_seniority(sd, self.nspatial) != 0:
                return 0.0

            # convert to spatial orbitals
            # NOTE: these variables are essentially the same as the output of
            #       generate_possible_orbpairs
            # ASSUMES: each orbital pair is a spatial orbital (alpha and beta orbitals)
            # FIXME: code will fail if an alternative orbpair is used
            inds_annihilated = np.array([self.dict_reforbpair_ind[(i, i+self.nspatial)]
                                         for i in orbs_annihilated if i < self.nspatial])
            inds_created = np.array([self.dict_orbpair_ind[(i, i+self.nspatial)]
                                     for i in orbs_created if i < self.nspatial])

            # if no derivatization
            if deriv is None:
                if inds_annihilated.size == inds_created.size == 0:
                    return 1.0
                val = self.compute_permanent(row_inds=inds_annihilated, col_inds=inds_created)
                if val != 0:
                    self.cache[sd] = val
                return val
            # if derivatization
            else:
                row_to_remove = deriv // self.norbpair
                col_to_remove = deriv % self.norbpair
                if inds_annihilated.size == inds_created.size == 0:
                    return 0.0
                val = self.compute_permanent(row_inds=inds_annihilated, col_inds=inds_created,
                                             deriv_row_col=(row_to_remove, col_to_remove))
                if val != 0:
                    self.d_cache[(sd, deriv)] = val
                return val
