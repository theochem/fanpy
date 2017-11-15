"""Antisymmetric Product of One-Reference-Orbital (AP1roG) Geminals wavefunction."""
import functools
import numpy as np
from wfns.backend import slater
from wfns.wfn.base import BaseWavefunction
from wfns.wfn.geminal.apig import APIG


class AP1roG(APIG):
    r"""Antisymmetric Product of One-Reference-Orbital (AP1roG) Geminals Wavefunction

    APIG wavefunction where a part of the geminal coefficient matrix, :math:`C`, is constrained to
    be an identity matrix. This part corresponds uniquely to a reference Slater determinant. Then,
    the wavefunction can be redescribed with respect to the reference Slater determinant, similar to
    Coupled Cluster wavefunctions.

    .. math::
        \ket{\Psi_{\mathrm{AP1roG}}}
        &= \prod_{q=1}^P T_q^\dagger \ket{\Phi_{\mathrm{ref}}}\\
        &= \prod_{q=1}^P \left( a_i^\dagger a_{\bar{i}}^\dagger +
           \sum_{i=P+1}^B c_{q;i}^{(\mathrm{AP1roG})} a_i^\dagger a_{\bar{i}}^\dagger \right)
           \ket{\Phi_{\mathrm{ref}}}\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}}
           | C(\mathbf{m})_{\mathrm{AP1roG}} |^+ \ket{\mathbf{m}}

    where :math:`P` is the number of electron pairs and :math:`\mathbf{m}` is a seniority-zero
    Slater determinant.

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
        Dictionary of orbital pair that are not occupied in the reference Slater determinant to the
        column index of the geminal coefficient matrix (after removal of the identity submatrix).
    dict_reforbpair_ind : dict of int to 2-tuple of int
        Dictionary of orbital pair that correspond to the reference Slater determinant to the column
        index of the identity submatrix.
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix (after removal of the identity
        submatrix) to the orbital pair that are no occupied in the reference Slater determinant.

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
    __init__(self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, ref_sd=None,
             params=None)
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
    assign_ref_sd(self, sd=None)
        Assign the reference Slater determinant.

    """
    def __init__(self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, ref_sd=None,
                 params=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type.
            Default is `np.float64`.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).
        ngem : {int, None}
            Number of geminals.
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal.
        params : np.ndarray
            Geminal coefficient matrix.
        ref_sd : {int, gmpy2.mpz, None}
            Reference Slater determinant.
            Default is the HF ground state.

        Notes
        -----
        Need to skip over APIG.__init__ because `assign_ref_sd` must come before `assign_params`.

        """
        BaseWavefunction.__init__(self, nelec, nspin, dtype=dtype)
        self.assign_ngem(ngem=ngem)
        self.assign_ref_sd(sd=ref_sd)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_params(params=params)
        self.load_cache()

    @property
    def template_params(self):
        """Return the template of the parameters of the given wavefunction.

        Since part of the coefficient matrix is constrained, these parts will be removed from the
        coefficient matrix.

        Returns
        -------
        template_params : np.ndarray(ngem, norbpair)
            Default parameters of the geminal wavefunction.

        """
        params = np.zeros((self.ngem, self.norbpair), dtype=self.dtype)
        return params

    def assign_ref_sd(self, sd=None):
        """Assign the reference Slater determinant.

        Parameters
        ----------
        sd : {int, None}
            Slater determinant to use as a reference.
            Default is the HF ground state.

        Raises
        ------
        TypeError
            If given `sd` cannot be turned into a Slater determinant (i.e. not integer or list of
            integers).
        ValueError
            If given `sd` does not have the correct spin.
            If given `sd` does not have the correct seniority.

        Notes
        -----
        This method depends on `nelec`, `nspin`, `spin`, and `seniority`.

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
        """Assign the number of geminals.

        Parameters
        ----------
        ngem : {int, None}
            Number of geminals.
            Default is the number of electron pairs.

        Raises
        ------
        TypeError
            If number of geminals is not an integer.
        ValueError
            If number of geminals is less than the number of electron pairs.
        NotImplementedError
            If number of geminals is not equal to the number of electron pairs.

        """
        super().assign_ngem(ngem)
        if self.ngem != self.npair:
            raise NotImplementedError('Currently, AP1roG does not support more than the exactly '
                                      ' right number of geminals (i.e. number of electron pairs).')

    def assign_orbpairs(self, orbpairs=None):
        """Assign the orbital pairs that will be used to construct the geminals.

        Since a part of the coefficient matrix is constrained, the column indices that correspond to
        these parts will be ignored. Instead, the column indices of the coefficient matrix after the
        removal of the constrained parts will be used.

        Parameters
        ----------
        orbpairs : iterable of 2-tuple/list of ints
            Indices of the orbital pairs that will be used to construct each geminal.
            Default is all possible orbital pairs.

        Raises
        ------
        TypeError
            If `orbpairs` is not an iterable.
            If an orbital pair is not given as a list or a tuple.
            If an orbital pair does not contain exactly two elements.
            If an orbital index is not an integer.
        ValueError
            If an orbital pair has the same integer.
            If an orbital pair occurs more than once.

        Notes
        -----
        Must have `ref_sd` and `nspin` defined for the default option.

        """
        super().assign_orbpairs(orbpairs=orbpairs)
        # removing orbital indices that correspond to the reference Slater determinant
        dict_orbpair_ind = {}
        dict_reforbpair_ind = {}
        for orbpair in self.dict_ind_orbpair.values():
            # if orbital pair is occupied in the reference Slater determinant
            if slater.occ(self.ref_sd, orbpair[0]) and slater.occ(self.ref_sd, orbpair[1]):
                dict_reforbpair_ind[orbpair] = len(dict_reforbpair_ind)
            # otherwise
            else:
                dict_orbpair_ind[orbpair] = len(dict_orbpair_ind)

        self.dict_orbpair_ind = dict_orbpair_ind
        self.dict_reforbpair_ind = dict_reforbpair_ind
        self.dict_ind_orbpair = {i: orbpair for orbpair, i in self.dict_orbpair_ind.items()}

    def load_cache(self):
        """Load the functions whose values will be cached.

        To minimize the cache size, the input is made as small as possible. Therefore, the cached
        function is not a method of an instance (because the instance is an input) and the smallest
        representation of the Slater determinant (an integer) is used as the only input. However,
        the functions must access other properties/methods of the instance, so they are defined
        within this method so that the instance is available within the namespace w/o use of
        `global` or `local`.

        Since the bitstring is used to represent the Slater determinant, they need to be processed,
        which may result in repeated processing depending on when the cached function is accessed.

        It is assumed that the cached functions will not be used to calculate redundant results. All
        simplifications that can be made is assumed to have already been made. For example, it is
        assumed that the overlap derivatized with respect to a parameter that is not associated with
        the given Slater determinant will never need to be evaluated because these conditions are
        caught before calling the cached functions.

        Notes
        -----
        Needs to access `memory` and `params`.

        """
        # assign memory allocated to cache
        if self.memory == np.inf:
            memory = None
        else:
            memory = int((self.memory - 5*8*self.params.size) / (self.params.size + 1))

        # create function that will be cached
        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp(sd):
            """Calculate the overlap with the Slater determinant.

            Parameters
            ----------
            sd : gmpy2.mpz
                Occupation vector of a Slater determinant given as a bitstring.

            Returns
            -------
            olp : {float, complex}
                Overlap of the current instance with the given Slater determinant.

            """
            # NOTE: Need to recreate spatial_ref_sd, inds_annihilated and inds_created
            spatial_sd, _ = slater.split_spin(sd, self.nspatial)
            spatial_ref_sd, _ = slater.split_spin(self.ref_sd, self.nspatial)
            orbs_annihilated, orbs_created = slater.diff_orbs(spatial_ref_sd, spatial_sd)
            inds_annihilated = np.array([self.dict_reforbpair_ind[(i, i+self.nspatial)]
                                         for i in orbs_annihilated])
            inds_created = np.array([self.dict_orbpair_ind[(i, i+self.nspatial)]
                                     for i in orbs_created])

            # FIXME: missing signature. see apig. Not a problem if alpha beta spin pairing
            return self.compute_permanent(row_inds=inds_annihilated, col_inds=inds_created)

        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp_deriv(sd, deriv):
            """Calculate the derivative of the overlap with the Slater determinant.

            Parameters
            ----------
            sd : gmpy2.mpz
                Occupation vector of a Slater determinant given as a bitstring.
            deriv : int
                Index of the parameter with respect to which the overlap is derivatized.

            Returns
            -------
            olp : {float, complex}
                Derivative of the overlap with respect to the given parameter.

            """
            spatial_sd, _ = slater.split_spin(sd, self.nspatial)
            spatial_ref_sd, _ = slater.split_spin(self.ref_sd, self.nspatial)
            orbs_annihilated, orbs_created = slater.diff_orbs(spatial_ref_sd, spatial_sd)
            inds_annihilated = np.array([self.dict_reforbpair_ind[(i, i+self.nspatial)]
                                         for i in orbs_annihilated])
            inds_created = np.array([self.dict_orbpair_ind[(i, i+self.nspatial)]
                                     for i in orbs_created])

            # FIXME: missing signature. see apig. Not a problem if alpha beta spin pairing
            return self.compute_permanent(row_inds=inds_annihilated, col_inds=inds_created,
                                          deriv=deriv)

        # create cache
        if not hasattr(self, '_cache_fns'):
            self._cache_fns = {}

        # store the cached function
        self._cache_fns['overlap'] = _olp
        self._cache_fns['overlap derivative'] = _olp_deriv

    # FIXME: allow other pairing schemes.
    def get_overlap(self, sd, deriv=None):
        """Return the overlap of the wavefunction with a Slater determinant.

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : int
            Index of the parameter to derivatize.
            Default does not derivatize.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        Notes
        -----
        Pairing scheme is assumed to be the alpha-beta spin orbital pair of each spatial orbital.
        This code will fail if another pairing scheme is used.

        """
        sd = slater.internal_sd(sd)

        # cut off beta part (for just the alpha/spatial part)
        spatial_ref_sd, _ = slater.split_spin(self.ref_sd, self.nspatial)
        spatial_sd, _ = slater.split_spin(sd, self.nspatial)
        # get indices of the occupied orbitals
        orbs_annihilated, orbs_created = slater.diff_orbs(spatial_ref_sd, spatial_sd)

        # if different number of electrons
        if len(orbs_annihilated) != len(orbs_created):
            return 0.0
        # if different seniority
        if slater.get_seniority(sd, self.nspatial) != 0:
            return 0.0

        # convert to spatial orbitals
        # NOTE: these variables are essentially the same as the output of
        #       generate_possible_orbpairs
        inds_annihilated = np.array([self.dict_reforbpair_ind[(i, i+self.nspatial)]
                                     for i in orbs_annihilated])
        inds_created = np.array([self.dict_orbpair_ind[(i, i+self.nspatial)]
                                 for i in orbs_created])

        # if no derivatization
        if deriv is None:
            if inds_annihilated.size == inds_created.size == 0:
                return 1.0

            return self._cache_fns['overlap'](sd)
        # if derivatization
        elif isinstance(deriv, (int, np.int64)):
            if deriv >= self.nparams:
                return 0.0
            if inds_annihilated.size == inds_created.size == 0:
                return 0.0

            return self._cache_fns['overlap derivative'](sd, deriv)
