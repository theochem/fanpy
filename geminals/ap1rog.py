from __future__ import absolute_import, division, print_function

import numpy as np
from gmpy2 import mpz

from .proj_wavefunction import ProjectionWavefunction
from . import slater
from .sd_list import doci_sd_list
from .proj_hamiltonian import doci_hamiltonian
from .math_tools import permanent_ryser

class AP1roG(ProjectionWavefunction):
    """ Antisymmetric Product of One-Reference-Orbital Geminals

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : tuple of np.ndarray(K,K)
        One electron integrals for the spatial orbitals
    G : tuple of np.ndarray(K,K,K,K)
        Two electron integrals for the spatial orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nspatial : int
        Number of spatial orbitals
    nspin : int
        Number of spin orbitals (alpha and beta)
    nelec : int
        Number of electrons
    npair : int
        Number of electron pairs
        Assumes that the number of electrons is even
    nparticle : int
        Number of quasiparticles (electrons)
    ngeminal : int
        Number of geminals
    nvirtual : int
        Number of virtual orbitals

    Private
    -------
    _methods : dict
        Default dimension of projection space
    _energy : float
        Electronic energy
    _nci : int
        Number of Slater determinants

    Methods
    -------
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
        self.nvirtual = self.nspatial - self.npair
        gem_coeffs = np.eye(self.npair, self.nvirtual, dtype=self.dtype)
        params = gem_coeffs.flatten()
        return params

    def compute_pspace(self, num_sd):
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
        return doci_sd_list(self, num_sd)

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
        if occ_alpha_indices != occ_beta_indices:
            raise ValueError('Given Slater determinant, {0}, does not belong'
                             ' to the DOCI Slater determinants'.format(bin(sd)))
        # get indices of the virtual orbitals
        vir_alpha_indices = slater.vir_indices(alpha_sd, self.nspatial)

        # build geminal coefficient
        if self.energy_is_param:
            gem_coeffs = self.params[:-1].reshape(self.npair, self.nvirtual)
        else:
            gem_coeffs = self.params.reshape(self.npair, self.nvirtual)

        val = 0.0
        # get the indices that need to be swapped from virtual to occupied 
        vo_col = [i - self.npair for i in occ_alpha_indices if i > vir_alpha_indices[0]]
        vo_row = range(len(vo_col))
        # if no derivatization
        if deriv is None:
            if len(vo_row) == 0:
                val = 1
            else:
                val = permanent_ryser(gem_coeffs[vo_row][:, vo_col])
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.energy_index:
            if len(vo_row) == 0:
                val = 1
            else:
                row_to_remove = deriv // self.nspatial
                col_to_remove = deriv %  self.nspatial
                if col_to_remove in vir_alpha_indices:
                    row_inds = [i for i in vo_row if i != row_to_remove]
                    col_inds = [i for i in vo_col if i != col_to_remove]
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
            \big< \Phi_i \big| H \big| \Psi \big>

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
        return sum(doci_hamiltonian(self, sd, self.orb_type, deriv=deriv))

    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in
        ProjectionWavefunction.compute_norm

        Some of the cache are emptied because the parameters are rewritten
        
        AP1roG is normalized by contruction.
        """
        pass
