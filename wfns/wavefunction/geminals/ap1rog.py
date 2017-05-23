""" AP1roG wavefunction
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .apig import APIG
from ..backend import slater
from ..backend.math_tools import permanent_combinatoric

__all__ = []

class AP1roG(APIG):
    """Antisymmetric Product of One-Reference-Orbital Geminals

    ..math::
        \ket{\Psi_{\mathrm{AP1roG}}}
        &= \prod_{q=1}^P T_q^\dagger \ket{\theta}\\
        &= \prod_{q=1}^P \left( a_q^\dagger a_{\bar{q}}^\dagger +
           \sum_{i=P+1}^B c_{q;i}^{(\mathrm{AP1roG})} a_i^\dagger a_{\bar{i}}^\dagger \right)
           \ket{\theta}\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}}
           | C(\mathbf{m})_{\mathrm{AP1roG}} |^+ \ket{\mathbf{m}}

    where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a Slater determinant
    (DOCI).

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
    template_coeffs : np.ndarray
        Initial guess coefficient matrix for the given reference Slater determinants
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
        Function in FancyCI
    """
    _nconstraints = 0

    @property
    def template_coeffs(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_coeffs : np.ndarray
            Geminal coefficient matrix (excluding the reference orbital part)
        """
        # FIXME: weird behaviour when more than npair geminals
        # FIXME: when there is more than npair geminals, then the parts of the identity matrix
        #        should repeat e.g.
        #        1 0 0 0 c c c c c
        #        1 0 0 0 c c c c c
        #        0 1 0 0 c c c c c
        #        0 1 0 0 c c c c c
        #        0 0 1 0 c c c c c
        #        0 0 0 1 c c c c c
        # NOTE: we need to know which rows (of identity matrix) are repeated
        return np.zeros((self.ngem, self.nspatial-self.npair), dtype=self.dtype)


    def assign_ngem(self, ngem=None):
        """ Assigns the number of geminals

        Parameters
        ----------
        ngem : int, None
            Number of geminals

        Raises
        ------
        TypeError
            If number of geminals is not an integer or long
        ValueError
            If number of geminals is less than the number of electron pairs
        NotImplementedError
            If number of geminals is not equal to the number of electron pairs
        """
        super(self.__class__, self).assign_ngem(ngem=ngem)
        if self.ngem != self.npair:
            raise NotImplementedError('AP1roG (as it is right now) does not support over projection'
                                      ' i.e. more than exactly the right number of geminals')


    def assign_ref_sds(self, ref_sds=None):
        """ Assigns the reference Slater determinants

        Reference Slater determinants are used to calculate the `energy`, `norm`, and
        `template_coeffs`.

        Parameters
        ----------
        ref_sds : int/long/gmpy2.mpz, list/tuple of ints, None
            Slater determinants that will be used as a reference for the wavefunction (e.g. for
            initial guess, energy calculation, normalization, etc)
            If `int` or `gmpy2.mpz`, then the equivalent Slater determinant (see `wfns.slater`) is
            used as a reference
            If `list` or `tuple` of Slater determinants, then multiple Slater determinants will be
            used as a reference. Note that multiple references require an initial guess
            Default is the first element of the `self.pspace`

        Raises
        ------
        TypeError
            If Slater determinants in a list or tuple are not compatible with the format used
            internally
            If Slater determinants are given in a form that is not int/long/gmpy2.mpz, list/tuple of
            ints or None
        """
        super(self.__class__, self).assign_ref_sds(ref_sds=ref_sds)
        if len(self.ref_sds) != 1:
            raise NotImplementedError('AP1roG (as it is right now) only supports one reference'
                                      ' Slater determinant')


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
        """
        super(AP1roG, self).assign_orbpairs(orbpairs=orbpairs)
        # NOTE: it's possible for two different orbital pair to share the same index b/c index for
        #       occupied orbital pair refers to the row index and the index for virtual orbital pair
        #       refers to the column index
        # NOTE: if there is more than one reference SD, we don't know which is "occupied" and
        #       "virtual"
        counter_occ = 0
        counter_vir = 0
        # Since only the virtual orbitals (wrt reference SD) are relevant
        for i, j in sorted(self.dict_orbpair_ind.iterkeys(), key=lambda x: x[0]):
            if not slater.occ(self.ref_sds[0], i) and not slater.occ(self.ref_sds[0], j):
                self.dict_orbpair_ind[(i, j)] = counter_vir
                counter_vir += 1
            else:
                self.dict_orbpair_ind[(i, j)] = counter_occ
                counter_occ += 1


    def compute_overlap(self, sd, deriv=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.
        ..math::
            \braket{\Phi_q^i | \Psi_{\mathrm{AP1roG}}}
            &= \bra{\Phi_q^i} \prod_{q=1}^P
            \left(a_q^\dagger a_{\bar{q}}^\dagger +
            \sum_{i=P+1}^B c_{q;i}^{(\mathrm{AP1roG})} a_i^\dagger a_{\bar{i}}^\dagger \right)
            \ket{\theta } = c_{q;i}^{(\mathrm{AP1roG})} \\
            &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}}
            |C(\mathbf{m})_{\mathrm{AP1roG}}|^+ \braket{\Phi_q^i| \mathbf{m}} = |C(\Phi_q^i)|^+

        where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a Slater determinant
        (DOCI).

        The geminal coefficient matrix :math:`\mathbf{C}` of AP1roG links the geminals with the
        underlying one-particle basis functions and has the following form,
        .. math::
            \mathbf{C}_{\mathrm{AP1roG}}  =
            \begin{pmatrix}
                1      & 0       & \dots & 0       & c_{1;P+1} & c_{1;P+2} & \cdots & c_{1;K}\\
                0      & 1       & \dots & 0       & c_{2;P+1} & c_{2;P+2} & \cdots & c_{2;K}\\
                \vdots & \vdots  & \ddots& \vdots  & \vdots    & \vdots    & \ddots & \vdots\\
                0      & 0       & \dots & 1       & c_{P;P+1} & c_{P;P+2} & \cdots & c_{P;K}
            \end{pmatrix}


        Parameters
        ----------
        sd : int, gmpy2.mpz, None
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        overlap : float

        Raises
        ------
        ValueError
            If `sd` does not have same number of electrons as ground state HF
            If `sd` does not have seniority zero
        """
        # caching is done wrt mpz objects, so you should convert sd to mpz first
        sd = slater.internal_sd(sd)
        # get indices of the occupied orbitals
        # NOTE: self.dict_orbpair_ind breaks if reference SD is not wrt first ref_sds (there can
        #       only be one)
        orbs_annihilated, orbs_created = slater.diff(self.ref_sds[0], sd)
        if len(orbs_annihilated) != len(orbs_created):
            raise ValueError('Given Slater determinant, {0}, does not have the same number of'
                             ' electrons as ground state HF wavefunction'.format(bin(sd)))
        if slater.get_seniority(sd, self.nspatial) != 0:
            raise ValueError('Given Slater determinant, {0}, does not belong to the DOCI Slater'
                             ' determinants'.format(bin(sd)))
        # convert to spatial orbitals
        orbs_annihilated = [self.dict_orbpair_ind[(i, i+self.nspatial)] for i in orbs_annihilated
                            if i < self.nspatial]
        orbs_created = [self.dict_orbpair_ind[(i, i+self.nspatial)] for i in orbs_created
                        if i < self.nspatial]

        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)

        val = 0.0
        # if no derivatization
        if deriv is None:
            if len(orbs_annihilated) == 0:
                val = 1.0
            else:
                val = permanent_combinatoric(gem_coeffs[orbs_annihilated][:, orbs_created])
            self.cache[sd] = val
        # if derivatization
        elif (isinstance(deriv, int) and 0 <= deriv < self.params.size - 1 and
              len(orbs_annihilated) > 0):
            row_to_remove = deriv // self.template_coeffs.shape[1]
            col_to_remove = deriv %  self.template_coeffs.shape[1]
            # find indices that excludes the specified row and column
            row_inds = [i for i in orbs_annihilated if i != row_to_remove]
            col_inds = [i for i in orbs_created if i != col_to_remove]
            # compute
            #  if the coefficient submatrix was eliminated (means there was only one spatial orbital
            #  difference)
            if len(row_inds) == 0 and len(col_inds) == 0:
                val = 1.0
            #  if the coefficient submatrix was derivatized wrt element inside of submatrix (i.e.
            #  atleast one row and column was removed)
            elif len(row_inds) < len(orbs_annihilated) and len(col_inds) < len(orbs_created):
                val = permanent_combinatoric(gem_coeffs[row_inds][:, col_inds])
            self.d_cache[(sd, deriv)] = val
        return val

    def normalize(self):
        """ Normalizes wavefunction

        Note
        ----
        AP1roG wavefunction (by definition) is always normalized (wrt reference determinant)
        """
        pass
