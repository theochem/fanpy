from __future__ import absolute_import, division, print_function

import numpy as np
from gmpy2 import mpz

from ..proj_wavefunction import ProjectedWavefunction
from ... import slater
from ...sd_list import sd_list
from ..proj_hamiltonian import sen0_hamiltonian
from ...math_tools import permanent_combinatoric

class AP1roG(ProjectedWavefunction):
    """ Antisymmetric Product of One-Reference-Orbital Geminals

    ..math::
        \big| \Psi_{\mathrm{AP1roG}} \big>
        &= \prod_{q=1}^P T_q^\dagger \big| \theta \big>\\
        &= \prod_{q=1}^P \left( a_q^\dagger a_{\bar{q}}^\dagger + \sum_{i=P+1}^B c_{q;i}^{(\mathrm{AP1roG})} a_i^\dagger a_{\bar{i}}^\dagger \right) \big| \theta \big> \\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} | C(\mathbf{m})_{\mathrm{AP1roG}} |^+ \big| \mathbf{m} \big>
    where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a
    Slater determinant (DOCI).

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    one_int : tuple of np.ndarray(K,K)
        One electron integrals for the spatial orbitals
    two_int : tuple of np.ndarray(K,K,K,K)
        Two electron integrals for the spatial orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nspatial : int
        Number of spatial orbitals
    nspin : int
        Number of spin orbitals (alpha and beta)
    nelec : int
        Number of electrons
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    params : np.ndarray(K)
        Guess for the parameters
        Iteratively updated during convergence
        Initial guess before convergence
        Coefficients after convergence
    cache : dict of mpz to float
        Cache of the Slater determinant to the overlap of the wavefunction with this
        Slater determinant
    d_cache : dict of (mpz, int) to float
        Cache of the Slater determinant to the derivative(with respect to some index)
        of the overlap of the wavefunction with this Slater determinan

    Properties
    ----------
    _methods : dict
        Default dimension of projection space
    _energy : float
        Electronic energy
    _nci : int
        Number of Slater determinants
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals
    npair : int
        Number of electron pairs (rounded down)
    nparams : int
        Number of parameters used to define the wavefunction
    nproj : int
        Number of Slater determinants to project against
    ref_sd : int or list of int
        Reference Slater determinants with respect to which the norm and the energy
        are calculated
        Integer that describes the occupation of a Slater determinant as a bitstring
        Or list of integers
    template_coeffs : np.ndarray(K)
        Default numpy array of parameters.
        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

    Methods
    -------
    __init__(nelec=None, one_int=None, two_int=None, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    __call__(method="default", **kwargs)
        Solves the wavefunction
    assign_dtype(dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_integrals(one_int, two_int, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
        (and the wavefunction)
    assign_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_nelec(nelec)
        Assigns the number of electrons
    _solve_least_squares(**kwargs)
        Solves the system of nonliear equations (and the wavefunction) using
        least squares method
    assign_params(params=None)
        Assigns the parameters used to describe the wavefunction.
        Adds random noise from the template if necessary
    assign_pspace(pspace=None)
        Assigns projection space
    overlap(sd, deriv=None)
        Retrieves overlap from the cache if available, otherwise compute overlap
    compute_norm(sd=None, deriv=None)
        Computes the norm of the wavefunction
    compute_energy(include_nuc=False, sd=None, deriv=None)
        Computes the energy of the wavefunction
    objective(x)
        The objective (system of nonlinear equations) associated with the projected
        Schrodinger equation
    jacobian(x)
        The Jacobian of the objective
    generate_pspace
        Generates a tuple of Slater determinants onto which the wavefunction is projected
    compute_overlap
        Computes the overlap of the wavefunction with one or more Slater determinants
    compute_hamiltonian
        Computes the hamiltonian of the wavefunction with respect to one or more Slater
        determinants
        By default, the energy is determined with respect to ref_sd
    normalize
        Normalizes the wavefunction (different definitions available)
        By default, the norm should the projection against the ref_sd squared
    """
    @property
    def template_coeffs(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_coeffs : np.ndarray(K, )

        """
        return np.zeros((self.npair, self.nspatial-self.npair), dtype=self.dtype)

    @property
    def nconstraints(self):
        """Number of constraints on the sollution of the projected wavefunction.

        For AP1roG this is 0 because the intermediate normalization is satisfied by construction.

        Returns
        -------
        nconstraints : int
        """

        self._nconstraints = 0

        return self._nconstraints

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
            \big< \Phi_q^i \big| \Psi_{\mathrm{AP1roG}} \big>
            &= \big< \Phi_q^i \big| \prod_{q=1}^P T_q^\dagger \big| \theta \big>\\
            &= \big< \Phi_q^i \big| \prod_{q=1}^P \left( a_q^\dagger a_{\bar{q}}^\dagger + \sum_{i=P+1}^B c_{q;i}^{(\mathrm{AP1roG})} a_i^\dagger a_{\bar{i}}^\dagger \right) \big| \theta \big> = c_{q;i}^{(\mathrm{AP1roG})} \\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} | C(\mathbf{m})_{\mathrm{AP1roG}} |^+ \big < \Phi_q^i \big| \mathbf{m} \big> = | C(\Phi_q^i) |^+
    where :math:`P` is the number of electron pairs, :math:`\mathbf{m}` is a
        Slater determinant (DOCI).
    The geminal coefficient matrix :math:`\mathbf{C}`) of AP1roG links the geminals with the underlying one-particle basis functions and has the following form,

    .. math::
        :label: cia

        \mathbf{C}_{\mathrm{AP1roG}}  =
        \begin{pmatrix}
            1      & 0       & \cdots & 0       & c_{1;P+1} & c_{1;P+2}&\cdots &c_{1;K}\\
            0      & 1       & \cdots & 0       & c_{2;P+1} & c_{2;P+2}&\cdots &c_{2;K}\\
            \vdots & \vdots  & \ddots & \vdots  & \vdots    & \vdots   &\ddots &\vdots\\
            0      & 0       & \cdots & 1       & c_{P;P+1} & c_{P;P+2}&\cdots & c_{P;K}
        \end{pmatrix}


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
        # if the alpha and the beta parts are the same, then the orbitals
        # correspond to the spatial orbitals
        occ_indices = occ_alpha_indices
        # get indices of the virtual orbitals
        vir_indices = slater.vir_indices(alpha_sd, self.nspatial)

        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)

        # get the indices that need to be swapped from virtual to occupied
        vo_col = [i - self.npair for i in range(self.npair, self.nspatial) if i in occ_indices]
        vo_row = [j for j in range(self.npair) if j not in occ_indices]
        assert len(vo_row) == len(vo_col)

        val = 0.0
        # if no derivatization
        if deriv is None:
            if len(vo_row) == 0:
                val = 1
            else:
                val = permanent_combinatoric(gem_coeffs[vo_row][:, vo_col])
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.params.size - 1:
            if len(vo_row) > 0:
                row_to_remove = deriv // self.template_coeffs.shape[1]
                col_to_remove = deriv %  self.template_coeffs.shape[1]
                # find indices that excludes the specified row and column
                row_inds = [i for i in vo_row if i != row_to_remove]
                col_inds = [i for i in vo_col if i != col_to_remove]
                # compute
                if len(row_inds) == 0 and len(col_inds) == 0:
                    val = 1
                elif len(row_inds) < len(vo_row) and len(col_inds) < len(vo_col):
                    val = permanent_combinatoric(gem_coeffs[row_inds][:, col_inds])
                self.d_cache[(sd, deriv)] = val
        return val

    def compute_hamiltonian(self, sd, deriv=None):
        """ Computes the hamiltonian of the wavefunction with respect to a Slater
        determinant

        ..math::
            \big< \Phi_i \big| H \big| \Psi_{mathrm{AP1roG}} \big>

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

    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in
        ProjectedWavefunction.compute_norm

        Some of the cache are emptied because the parameters are rewritten

        AP1roG is normalized by contruction.
        """
        pass
