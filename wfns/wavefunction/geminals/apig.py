"""APIG wavefunction."""
from __future__ import absolute_import, division, print_function
import numpy as np
from .base_geminal import BaseGeminal

__all__ = []


class APIG(BaseGeminal):
    r"""Antisymmetric Product of Interacting Geminals (APIG) Wavefunction.

    .. math::
        \big| \Psi_{\mathrm{APIG}} \big>
        &= \prod_{p=1}^P a^\dagger_p \big| \theta \big>\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} |C(\mathbf{m})|^+
        \big| \mathbf{m} \big>

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
    def spin(self):
        """Spin of the APIG wavefunction

        Note
        ----
        Seniority is not zero if you change the pairing scheme
        """
        return 0.0

    @property
    def seniority(self):
        """Seniority of the APIG wavefunction

        Note
        ----
        Seniority is not zero if you change the pairing scheme
        """
        return 0

    def assign_orbpairs(self, orbpairs=None):
        """Set the orbital pairs that will be used to construct the geminals.

        Parameters
        ----------
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal
            Default is all spatial orbital (alpha beta) pair

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
            If any two orbital pair shares an orbital
            If any orbital is not included in any orbital pair

        Note
        ----
        Must have `nspin` defined for the default option
        """
        if orbpairs is None:
            orbpairs = ((i, i+self.nspatial) for i in range(self.nspatial))
        super().assign_orbpairs(orbpairs)

        all_orbs = [j for i in self.dict_orbpair_ind.keys() for j in i]
        if len(all_orbs) != len(set(all_orbs)):
            raise ValueError('At least two orbital pairs share an orbital')
        elif len(all_orbs) != self.nspin:
            raise ValueError('Not all of the orbitals are included in orbital pairs')

    def generate_possible_orbpairs(self, occ_indices):
        """Yield the possible orbital pairs that can construct the given Slater determinant.

        The APIG wavefunction only contains one pairing scheme for each Slater determinant (because
        the orbital pairs are disjoint of one another)

        Parameters
        ----------
        occ_indices : N-tuple of int
            Indices of the orbitals from which the Slater determinant is constructed
            Must be strictly increasing.

        Yields
        ------
        orbpairs : P-tuple of 2-tuple of ints
            Indices of the creation operators (grouped by orbital pairs) that construct the Slater
            determinant.
        sign : int
            Signature of the transpositions required to shuffle the `orbitalpairs` back into the
            original order in `occ_indices`.

        Raises
        ------
        ValueError
            If the number of electrons in the Slater determinant does not match up with the number
            of electrons in the wavefunction
            If the Slater determinant cannot be constructed using the APIG pairing scheme
        """
        if len(occ_indices) != self.nelec:
            raise ValueError('The number of electrons in the Slater determinant does not match up '
                             'with the number of electrons in the wavefunction.')
        orbpairs = tuple((i, i+self.nspatial) for i in occ_indices if i < self.nspatial)

        if set(occ_indices) != set(j for i in orbpairs for j in i):
            raise ValueError('This Slater determinant cannot be created using the pairing scheme of'
                             ' APIG wavefunction.')

        # signature to turn orbpairs into strictly INCREASING order.
        sign = (-1)**((self.npair//2) % 2)

        yield orbpairs, sign

    # TODO: refactor when APG is set up
    def to_apg(self):
        """Convert APIG wavefunction to APG wavefunction.

        Returns
        -------
        apg : APG instance
            APG wavefunction that corresponds to the given APIG wavefunction
        """
        # import APG (because nasty cyclic import business)
        from .apg import APG

        # make apig coefficients
        apig_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)
        # make apg coefficients
        apg = APG(self.nelec, self.one_int, self.two_int, dtype=self.dtype, nuc_nuc=self.nuc_nuc,
                  orbtype=self.orbtype, ref_sds=self.ref_sds, ngem=self.ngem)
        apg_coeffs = apg.params[:-1].reshape(apg.template_coeffs.shape)
        # assign apg coefficients
        for orbpair, ind in self.dict_orbpair_ind.items():
            apg_coeffs[:, apg.dict_orbpair_ind[orbpair]] = apig_coeffs[:, ind]
        apg.assign_params(np.hstack((apg_coeffs.flat, self.get_energy())))

        # make wavefunction
        return apg
