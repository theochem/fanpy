"""Antisymmetric Product of One-Reference-Orbital (AP1roG) Geminals wavefunction."""
from __future__ import absolute_import, division, print_function
import functools
import numpy as np
from wfns.backend import slater
from wfns.wavefunction.base_wavefunction import BaseWavefunction
from wfns.wavefunction.geminals.apig import APIG
from pydocstring.wrapper import docstring_class

__all__ = []


@docstring_class(indent_level=1)
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
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair that are not occupied in the reference Slater determinant to the
        column index of the geminal coefficient matrix (after removal of the identity submatrix).
    dict_reforbpair_ind : dict of int to 2-tuple of int
        Dictionary of orbital pair that correspond to the reference Slater determinant to the column
        index of the identity submatrix.
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix (after removal of the identity
        submatrix) to the orbital pair that are no occupied in the reference Slater determinant.

    """
    def __init__(self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, ref_sd=None,
                 params=None):
        """

        Parameters
        ----------
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

    @property
    def template_params(self):
        """

        Since part of the coefficient matrix is constrained, these parts will be removed from the
        coefficient matrix.

        """
        params = np.zeros((self.ngem, self.norbpair), dtype=self.dtype)
        return params

    def assign_ref_sd(self, sd=None):
        """Set the reference Slater determinant.

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

        Multiple Slater determinant references are not allowed.
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
        """

        Raises
        ------
        NotImplementedError
            If number of geminals is not equal to the number of electron pairs.

        """
        super().assign_ngem(ngem)
        if self.ngem != self.npair:
            raise NotImplementedError('Currently, AP1roG does not support more than the exactly '
                                      ' right number of geminals (i.e. number of electron pairs).')

    def assign_orbpairs(self, orbpairs=None):
        """

        Since a part of the coefficient matrix is constrained, the column indices that correspond to
        these parts will be ignored. Instead, the column indices of the coefficient matrix after the
        removal of the constrained parts will be used.

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

    def get_overlap(self, sd, deriv=None):
        if not slater.is_internal_sd(sd):
            sd = slater.internal_sd(sd)

        # ASSUMES pairing scheme (alpha-beta spin orbital of same spatial orbital)
        # cut off beta part (for just the alpha/spatial part)
        spatial_ref_sd, _ = slater.split_spin(self.ref_sd, self.nspatial)
        spatial_sd, _ = slater.split_spin(sd, self.nspatial)
        # get indices of the occupied orbitals
        orbs_annihilated, orbs_created = slater.diff(spatial_ref_sd, spatial_sd)

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
                                     for i in orbs_annihilated])
        inds_created = np.array([self.dict_orbpair_ind[(i, i+self.nspatial)]
                                 for i in orbs_created])

        # if no derivatization
        if deriv is None:
            if inds_annihilated.size == inds_created.size == 0:
                return 1.0

            # if cached function has not been created yet
            if 'overlap' not in self._cache_fns:
                # assign memory allocated to cache
                if self.memory == np.inf:
                    memory = None
                else:
                    memory = int((self.memory - 5*8*self.params.size) / (self.params.size + 1))

                # create function that will be cached
                @functools.lru_cache(maxsize=memory, typed=False)
                def _olp(sd):
                    # FIXME: ugly, repeats code
                    # NOTE: sd is used as the key because it uses less memory
                    # NOTE: Need to recreate inds_annihilated and inds_created
                    spatial_sd, _ = slater.split_spin(sd, self.nspatial)
                    orbs_annihilated, orbs_created = slater.diff(spatial_ref_sd, spatial_sd)
                    inds_annihilated = np.array([self.dict_reforbpair_ind[(i, i+self.nspatial)]
                                                 for i in orbs_annihilated])
                    inds_created = np.array([self.dict_orbpair_ind[(i, i+self.nspatial)]
                                             for i in orbs_created])

                    # FIXME: missing signature. see apig
                    return self.compute_permanent(row_inds=inds_annihilated, col_inds=inds_created)

                # store the cached function
                self._cache_fns['overlap'] = _olp
            # if cached function already exists
            else:
                # reload cached function
                _olp = self._cache_fns['overlap']

            return _olp(sd)
        # if derivatization
        elif isinstance(deriv, int):
            if deriv >= self.nparams:
                return 0.0
            if inds_annihilated.size == inds_created.size == 0:
                return 0.0

            # if cached function has not been created yet
            if 'overlap derivative' not in self._cache_fns:
                # assign memory allocated to cache
                if self.memory == np.inf:
                    memory = None
                else:
                    memory = int((self.memory - 5*8*self.params.size)
                                 / (self.params.size + 1) * self.params.size)

                # create function that will be cached
                @functools.lru_cache(maxsize=memory, typed=False)
                def _olp_deriv(sd, deriv):
                    # FIXME: ugly, repeats code
                    # NOTE: sd and deriv is used as the key because it uses less memory
                    # NOTE: Need to recreate inds_annihilated and inds_created, row_removed,
                    #       col_removed
                    spatial_sd, _ = slater.split_spin(sd, self.nspatial)
                    orbs_annihilated, orbs_created = slater.diff(spatial_ref_sd, spatial_sd)
                    row_removed = deriv // self.norbpair
                    col_removed = deriv % self.norbpair
                    inds_annihilated = np.array([self.dict_reforbpair_ind[(i, i+self.nspatial)]
                                                 for i in orbs_annihilated])
                    inds_created = np.array([self.dict_orbpair_ind[(i, i+self.nspatial)]
                                             for i in orbs_created])

                    # FIXME: missing signature. see apig
                    return self.compute_permanent(row_inds=inds_annihilated, col_inds=inds_created,
                                                  deriv_row_col=(row_removed, col_removed))

                # store the cached function
                self._cache_fns['overlap derivative'] = _olp_deriv
            # if cached function already exists
            else:
                # reload cached function
                _olp_deriv = self._cache_fns['overlap derivative']

            return _olp_deriv(sd, deriv)
