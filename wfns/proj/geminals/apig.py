""" APIG wavefunction
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .geminal import Geminal
from ..proj_hamiltonian import sen0_hamiltonian
from ... import slater
from ...sd_list import sd_list
from ...math_tools import permanent_ryser


class APIG(Geminal):
    """ Antisymmetric Product of Interacting Geminals

    ..math::
        \ket{\Psi_{\mathrm{APIG}}}
        &= \prod_{p=1}^P two_int_p^\dagger \ket{\theta}\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} |C(\mathbf{m})|^+
        \ket{\mathbf{m}}
    where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a
    Slater determinant (DOCI).

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
    @property
    def template_orbpairs(self):
        """ Default orbital pairs used to construct the wavefunction
        """
        return tuple((i, i+self.nspatial) for i in range(self.nspatial))

    @property
    def template_coeffs(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_coeffs : np.ndarray

        """
        return np.eye(self.npair, self.nspatial, dtype=self.dtype)

    def generate_pspace(self, num_sd):
        """ Generates Slater determinants to project onto

        # FIXME: wording
        Since APIG wavefunction only consists of Slater determinants with orbitals that are
        paired (alpha and beta orbitals corresponding to the same spatial orbital are occupied),
        the Slater determinants used correspond to those in DOCI wavefunction

        Parameters
        ----------
        num_sd : int
            Number of Slater determinants to generate

        Returns
        -------
        pspace : list of gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        """
        return sd_list(self.nelec, self.nspatial, num_limit=num_sd, seniority=0)

    def compute_overlap(self, sd, deriv=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.
        ..math::
            \big< \Phi_k \big| \Psi_{\mathrm{APIG}} \big>
            &= \big< \Phi_k \big| \prod_{p=1}^P two_int_p^\dagger \big| \theta \big>\\
            &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} | C(\mathbf{m}) |^+ \big< \Phi_k \big| \mathbf{m} \big>
            &= | C(\Phi_k) |^+
        where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a
        Slater determinant (DOCI).

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        overlap : float
        """
        # caching is done wrt mpz objects, so you should convert sd to mpz first
        sd = mpz(sd)
        # get indices of the occupied orbitals
        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
        occ_alpha_indices = slater.occ_indices(alpha_sd)
        occ_beta_indices = slater.occ_indices(beta_sd)
        if occ_alpha_indices != occ_beta_indices:
            raise ValueError('Given Slater determinant, {0}, does not belong'
                             ' to the DOCI Slater determinants'.format(bin(sd)))

        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)

        val = 0.0
        # if no derivatization
        if deriv is None:
            val = permanent_ryser(gem_coeffs[:, occ_alpha_indices])
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.params.size - 1:
            row_to_remove = deriv // self.nspatial
            col_to_remove = deriv % self.nspatial
            if col_to_remove in occ_alpha_indices:
                row_inds = [i for i in range(self.npair) if i != row_to_remove]
                col_inds = [i for i in occ_alpha_indices if i != col_to_remove]
                if len(row_inds) == 0 and len(col_inds) == 0:
                    val = 1
                else:
                    val = permanent_ryser(gem_coeffs[row_inds][:, col_inds])
                # construct new gmpy2.mpz to describe the slater determinant and
                # derivation index
                self.d_cache[(sd, deriv)] = val
        return val

    def compute_hamiltonian(self, sd, deriv=None):
        """ Computes the hamiltonian of the wavefunction with respect to a Slater
        determinant

        ..math::
            \big< \Phi_i \big| H \big| \Psi_{mathrm{APIG}} \big>

        Since only Slater determinants from DOCI will be used, we can use the DOCI
        Hamiltonian

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        float
        """
        return sum(sen0_hamiltonian(self, sd, self.orbtype, deriv=deriv))

    # FIXME: remove
    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in
        ProjectedWavefunction.compute_norm

        Some of the cache are emptied because the parameters are rewritten
        """
        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)
        # normalize the wavefunction
        norm = self.compute_norm()
        gem_coeffs *= norm**(-0.5 / self.npair)
        # set attributes
        self.params[:-1] = gem_coeffs.flatten()
        # empty cache
        for sd in self.default_ref_sds:
            del self.cache[sd]
            for i in (j for j in self.d_cache.keys() if j[0] == sd):
                del self.d_cache[i]
            # Following requires d_cache to be a dictionary of dictionary
            # self.d_cache[sd] = {}
