from __future__ import absolute_import, division, print_function

import numpy as np
from gmpy2 import mpz

from .proj_wavefunction import ProjectionWavefunction
from . import slater
from .sd_list import doci_sd_list
from .proj_hamiltonian import doci_hamiltonian
from .math_tools import permanent_ryser


class APseqG(ProjectionWavefunction):
    """ Antisymmetric Product of Sequantially Interacting Geminals

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : np.ndarray(K,K) or tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        If tuple of np.ndarray (length 2), one electron integrals for the (alpha, alpha)
        and the (beta, beta) unrestricted orbitals
    G : np.ndarray(K,K,K,K) or tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        If tuple of np.ndarray (length 3), two electron integrals for the
        (alpha, alpha, alpha, alpha), (alpha, beta, alpha, beta), and
        (beta, beta, beta, beta) unrestricted orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nelec : int
        Number of electrons
    orb_type : {'restricted', 'unrestricted', 'generalized'}
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
    _methods : dict of func
        Dictionary of methods that are used to solve the wavefunction
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals
    npair : int
        Number of electron pairs (rounded down)
    nparam : int
        Number of parameters used to define the wavefunction
    nproj : int
        Number of Slater determinants to project against
    energy_index : int
        Index of the energy in the list of parameters
    ref_sd : int or list of int
        Reference Slater determinants with respect to which the norm and the energy
        are calculated
        Integer that describes the occupation of a Slater determinant as a bitstring
        Or list of integers
    template_params : np.ndarray(K)
        Default numpy array of parameters.
        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

    Method
    ------
    __init__(nelec=None, H=None, G=None, dtype=None, nuc_nuc=None, orb_type=None)
        Initializes wavefunction
    __call__(method="default", **kwargs)
        Solves the wavefunction
    assign_dtype(dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_integrals(H, G, orb_type=None)
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
    compute_pspace
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
    def template_params(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_params : np.ndarray(K, )

        """
        self.seqmax = self.npair - 1
        gem_coeffs = np.eye(self.nspatial, self.nspatial, dtype=self.dtype)
        params = gem_coeffs.flatten()
        return params

    def compute_pspace(self, num_sd):
        """ Generates Slater determinants to project onto

        The Slater determinants used correspond to those in FCI wavefunction

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
        return ci_sd_list(self, num_sd)

    def compute_overlap(self, sd, deriv=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.

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
        if len(occ_alpha_indices) != len(occ_beta_indices):
            raise ValueError('Given Slater determinant, {0}, should have same 
                                number of alpha and beta electrons'.format(bin(sd)))

        # build geminal coefficient
        if self.energy_is_param:
            gem_coeffs = self.params[:-1].reshape(self.npair, self.nspatial)
        else:
            gem_coeffs = self.params.reshape(self.npair, self.nspatial)

        val = 0.0


        # order indices of sequentially occupied orbitals
        # put all the pairs that are going to be used in the first npair rows 
        fainds = []
        fbinds = []
        for j,a in enumerate(range(self.npair)):
            seqlsta = [[]]*self.npair
            seqlstb = [[]]*self.npair
            if occ_beta_indices[j] == occ_alpha_indices[j]:
                b = occ_beta_indices[j]
                seqlsta[0].extend([a])
                seqlstb[0].extend([b])
            else:
                for seq in range(1, self.seqmax):
                    if occ_beta_indices[j] == occ_alpha_indices[j]+seq:
                        b = occ_beta_indices[j]
                        seqlsta[seq].extend([a])
                        seqlstb[seq].extend([b])
                    elif occ_beta_indices[j] == occ_alpha_indices[j]-seq:
                        b = occ_beta_indices[j]
                        seqlsta[seq].extend([a])
                        seqlstb[seq].extend([b])
            ainds = [i for k in seqlsta for i in k]
            binds = [i for l in seqlstb for i in l] 
            assert len(ainds) == self.npair and len(binds) == len(ainds)
            fainds.append(ainds) ; fbinds.append(binds)

        # if no derivatization
        if deriv is None:
            val = permanent_ryser(gem_coeffs[fainds, fbinds])
            self.cache[sd] = val
        # if derivatization
        # TODO:Modify the derivatives 
        elif isinstance(deriv, int) and deriv < self.energy_index:
            return NotImplementedError
        return val

    def compute_hamiltonian(self, sd, deriv=None):
        """ Computes the hamiltonian of the wavefunction with respect to a Slater
        determinant

        ..math::
            \big< \Phi_i \big| H \big| \Psi_{mathrm{APseqG}} \big>

        Since only Slater determinants from FCI will be used
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
        return sum(hamiltonian(self, sd, self.orb_type, deriv=deriv))

    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in
        ProjectionWavefunction.compute_norm

        Some of the cache are emptied because the parameters are rewritten
        """
        pass
